import CarbocationLocalLLM
import CarbocationLlamaCommonBridge
import Foundation
import llama

enum LlamaCommonChatError: Error, LocalizedError {
    case native(status: Int32, detail: String)
    case invalidResponse(String)

    var errorDescription: String? {
        switch self {
        case .native(let status, let detail):
            return "llama.cpp common-chat error \(status): \(detail)"
        case .invalidResponse(let detail):
            return "Invalid llama.cpp common-chat response: \(detail)"
        }
    }
}

struct LlamaCommonChatGrammarTrigger: Codable, Equatable, Sendable {
    enum Kind: String, Codable, Sendable {
        case token
        case word
        case pattern
        case patternFull = "pattern_full"
        case unknown
    }

    var type: Kind
    var value: String
    var token: Int32?
}

struct LlamaCommonChatRenderMetadata: Decodable, Equatable, Sendable {
    var prompt: String
    var format: String
    var formatCode: Int
    var grammar: String
    var grammarLazy: Bool
    var generationPrompt: String
    var supportsThinking: Bool
    var thinkingStartTag: String
    var thinkingEndTags: [String]
    var grammarTriggers: [LlamaCommonChatGrammarTrigger]
    var preservedTokens: [String]
    var additionalStops: [String]
    var hasParser: Bool
    var hasExplicitTemplate: Bool

    var outputProfile: OutputSanitizationProfile {
        let pairs = thinkingEndTags.compactMap { end -> OutputDelimiterPair? in
            guard !thinkingStartTag.isEmpty, !end.isEmpty else { return nil }
            return OutputDelimiterPair(open: thinkingStartTag, close: end)
        }
        return OutputSanitizationProfile(thinkingPairs: pairs)
    }
}

struct LlamaCommonChatParsedToolCall: Decodable, Equatable, Sendable {
    var name: String
    var arguments: String
    var id: String?

    func materialized(triggerPhase: LLMStreamContentPhase? = nil) -> LLMToolCall? {
        guard !name.isEmpty,
              let data = arguments.data(using: .utf8),
              let value = try? JSONDecoder().decode(LLMJSONValue.self, from: data),
              case .object = value else {
            return nil
        }
        return LLMToolCall(
            executionID: id ?? "",
            rawID: id,
            name: name,
            arguments: value,
            triggerPhase: triggerPhase
        )
    }
}

struct LlamaCommonChatParsedMessage: Decodable, Equatable, Sendable {
    var role: String
    var content: String
    var reasoningContent: String
    var toolCalls: [LlamaCommonChatParsedToolCall]
    var toolName: String?
    var toolCallID: String?
}

struct LlamaCommonChatParsedDiff: Decodable, Equatable, Sendable {
    var reasoningContentDelta: String
    var contentDelta: String
    var toolCallIndex: Int?
    var toolCallDelta: LlamaCommonChatParsedToolCall?
}

struct LlamaCommonChatParseUpdate: Decodable, Equatable, Sendable {
    var message: LlamaCommonChatParsedMessage
    var diffs: [LlamaCommonChatParsedDiff]
    var isPartial: Bool
    var usedRawContentFallback: Bool
}

private struct LlamaCommonChatRenderRequest: Encodable {
    var messages: [LlamaCommonChatMessage]
    var tools: [LlamaCommonChatTool]
    var toolChoice: LlamaCommonChatToolChoice
    var parallelToolCalls: Bool
    var addGenerationPrompt: Bool
    var continueFinalMessage: String?
    var enableThinking: Bool
    var reasoningEffort: String?
    var preserveThinking: Bool?
    var grammar: String?
}

private struct LlamaCommonChatMessage: Encodable {
    struct FunctionCall: Encodable {
        struct Function: Encodable {
            var name: String
            var arguments: String
        }
        var id: String?
        var type = "function"
        var function: Function
    }

    var role: String
    var content: LLMJSONValue
    var reasoningContent: String?
    var toolCalls: [FunctionCall]?
    var toolCallID: String?
    var name: String?

    init(_ message: ChatTemplateMessage) throws {
        role = message.role
        content = message.content
        reasoningContent = message.reasoningContent
        toolCallID = message.toolCallID
        name = message.name
        toolCalls = try message.toolCalls.isEmpty ? nil : message.toolCalls.map { call in
            FunctionCall(
                id: call.rawID ?? call.executionID.nonEmpty,
                function: FunctionCall.Function(
                    name: call.name,
                    arguments: try call.arguments.jsonString(prettyPrinted: false)
                )
            )
        }
    }
}

private struct LlamaCommonChatTool: Encodable {
    struct Function: Encodable {
        var name: String
        var description: String
        var parameters: LLMJSONValue
    }
    var type = "function"
    var function: Function

    init(_ definition: LLMToolDefinition) {
        function = Function(
            name: definition.name,
            description: definition.description,
            parameters: definition.parameters
        )
    }
}

private enum LlamaCommonChatToolChoice: Encodable {
    case value(String)
    case named(String)

    func encode(to encoder: Encoder) throws {
        switch self {
        case .value(let value):
            var container = encoder.singleValueContainer()
            try container.encode(value)
        case .named(let name):
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode("named", forKey: .type)
            try container.encode(name, forKey: .name)
        }
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case name
    }

    init(_ choice: LLMToolChoice) {
        switch choice {
        case .auto: self = .value("auto")
        case .none: self = .value("none")
        case .required: self = .value("required")
        case .named(let name): self = .named(name)
        }
    }
}

final class LlamaCommonChatTemplates: @unchecked Sendable {
    private let handle: OpaquePointer
    let hasExplicitTemplate: Bool

    init(
        model: OpaquePointer?,
        templateOverride: String? = nil,
        bosTokenOverride: String? = nil,
        eosTokenOverride: String? = nil
    ) throws {
        var created: OpaquePointer?
        var error: UnsafeMutablePointer<CChar>?
        let status = templateOverride?.withCString { templatePointer in
            Self.withOptionalCString(bosTokenOverride) { bosPointer in
                Self.withOptionalCString(eosTokenOverride) { eosPointer in
                    carbocation_llama_chat_templates_create(
                        model,
                        templatePointer,
                        bosPointer,
                        eosPointer,
                        &created,
                        &error
                    )
                }
            }
        } ?? Self.withOptionalCString(bosTokenOverride) { bosPointer in
            Self.withOptionalCString(eosTokenOverride) { eosPointer in
                carbocation_llama_chat_templates_create(
                    model,
                    nil,
                    bosPointer,
                    eosPointer,
                    &created,
                    &error
                )
            }
        }
        try Self.check(status, error: error)
        guard let created else {
            throw LlamaCommonChatError.invalidResponse("template creation returned no handle")
        }
        handle = created
        hasExplicitTemplate = carbocation_llama_chat_templates_has_explicit_template(created)
    }

    deinit {
        carbocation_llama_chat_templates_free(handle)
    }

    func render(
        messages: [ChatTemplateMessage],
        tools: [LLMToolDefinition],
        toolChoice: LLMToolChoice = .auto,
        options: GenerationOptions,
        addGenerationPrompt: Bool = true,
        continueFinalMessage: String? = nil
    ) throws -> LlamaCommonChatPlan {
        let request = LlamaCommonChatRenderRequest(
            messages: try messages.map(LlamaCommonChatMessage.init),
            tools: tools.map(LlamaCommonChatTool.init),
            toolChoice: LlamaCommonChatToolChoice(toolChoice),
            parallelToolCalls: !tools.isEmpty && toolChoice != .none,
            addGenerationPrompt: addGenerationPrompt,
            continueFinalMessage: continueFinalMessage,
            enableThinking: options.enableThinking,
            reasoningEffort: options.reasoningEffort?.rawValue,
            preserveThinking: options.preserveThinking,
            grammar: options.grammar
        )
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let requestData = try encoder.encode(request)
        guard let requestJSON = String(data: requestData, encoding: .utf8) else {
            throw LlamaCommonChatError.invalidResponse("render request was not UTF-8")
        }

        var plan: OpaquePointer?
        var result: UnsafeMutablePointer<CChar>?
        var error: UnsafeMutablePointer<CChar>?
        let status = requestJSON.withCString { requestPointer in
            carbocation_llama_chat_templates_render(
                handle,
                requestPointer,
                &plan,
                &result,
                &error
            )
        }
        try Self.check(status, error: error)
        guard let plan, let result else {
            throw LlamaCommonChatError.invalidResponse("render returned no plan or metadata")
        }
        var adoptedPlan = false
        defer {
            if !adoptedPlan {
                carbocation_llama_chat_plan_free(plan)
            }
        }
        defer { carbocation_llama_chat_string_free(result) }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let metadata = try decoder.decode(
            LlamaCommonChatRenderMetadata.self,
            from: Data(String(cString: result).utf8)
        )
        adoptedPlan = true
        return LlamaCommonChatPlan(handle: plan, metadata: metadata)
    }

    fileprivate static func check(
        _ status: carbocation_llama_chat_status,
        error: UnsafeMutablePointer<CChar>?
    ) throws {
        defer {
            if let error {
                carbocation_llama_chat_string_free(error)
            }
        }
        guard status == CARBOCATION_LLAMA_CHAT_STATUS_OK else {
            throw LlamaCommonChatError.native(
                status: Int32(status.rawValue),
                detail: error.map { String(cString: $0) } ?? "Unknown native error"
            )
        }
    }

    private static func withOptionalCString<Result>(
        _ value: String?,
        _ body: (UnsafePointer<CChar>?) -> Result
    ) -> Result {
        guard let value else { return body(nil) }
        return value.withCString(body)
    }
}

final class LlamaCommonChatPlan: @unchecked Sendable {
    fileprivate let handle: OpaquePointer
    let metadata: LlamaCommonChatRenderMetadata

    fileprivate init(handle: OpaquePointer, metadata: LlamaCommonChatRenderMetadata) {
        self.handle = handle
        self.metadata = metadata
    }

    deinit {
        carbocation_llama_chat_plan_free(handle)
    }

    func makeParser() throws -> LlamaCommonChatParser {
        var parser: OpaquePointer?
        var error: UnsafeMutablePointer<CChar>?
        let status = carbocation_llama_chat_parser_create(handle, &parser, &error)
        try LlamaCommonChatTemplates.check(status, error: error)
        guard let parser else {
            throw LlamaCommonChatError.invalidResponse("parser creation returned no handle")
        }
        return LlamaCommonChatParser(handle: parser)
    }
}

final class LlamaCommonChatParser: @unchecked Sendable {
    private let handle: OpaquePointer

    fileprivate init(handle: OpaquePointer) {
        self.handle = handle
    }

    deinit {
        carbocation_llama_chat_parser_free(handle)
    }

    func append(_ text: String, isPartial: Bool) throws -> LlamaCommonChatParseUpdate {
        var result: UnsafeMutablePointer<CChar>?
        var error: UnsafeMutablePointer<CChar>?
        let status = text.withCString { textPointer in
            carbocation_llama_chat_parser_update(
                handle,
                textPointer,
                isPartial,
                &result,
                &error
            )
        }
        try LlamaCommonChatTemplates.check(status, error: error)
        guard let result else {
            throw LlamaCommonChatError.invalidResponse("parser returned no update")
        }
        defer { carbocation_llama_chat_string_free(result) }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(
            LlamaCommonChatParseUpdate.self,
            from: Data(String(cString: result).utf8)
        )
    }
}

private extension String {
    var nonEmpty: String? { isEmpty ? nil : self }
}
