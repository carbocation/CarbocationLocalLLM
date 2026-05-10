import CarbocationLocalLLM
import Foundation
import Jinja

struct ChatTemplateMessage: Equatable {
    var role: String
    var content: LLMJSONValue
    var toolCalls: [LLMToolCall]
    var toolCallID: String?
    var name: String?

    init(
        role: String,
        content: String = "",
        toolCalls: [LLMToolCall] = [],
        toolCallID: String? = nil,
        name: String? = nil
    ) {
        self.role = role
        self.content = .string(content)
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.name = name
    }

    init(
        role: String,
        content: LLMJSONValue,
        toolCalls: [LLMToolCall] = [],
        toolCallID: String? = nil,
        name: String? = nil
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.name = name
    }
}

struct ChatTemplatePromptFormatter {
    enum Error: Swift.Error, LocalizedError {
        case notJinjaTemplate
        case missingUserContent

        var errorDescription: String? {
            switch self {
            case .notJinjaTemplate:
                return "Embedded chat template is not a Jinja template."
            case .missingUserContent:
                return "Applied chat template did not include the user message."
            }
        }
    }

    private let template: Template

    init(template source: String) throws {
        guard source.contains("{%") || source.contains("{{") || source.contains("{#") else {
            throw Error.notJinjaTemplate
        }

        self.template = try Template(
            source,
            with: Template.Options(lstripBlocks: true, trimBlocks: true)
        )
    }

    static func format(
        template source: String,
        system: String,
        user: String,
        bosToken: String,
        eosToken: String,
        enableThinking: Bool = false
    ) throws -> String {
        try Self(template: source).format(
            messages: Self.defaultMessages(system: system, user: user),
            tools: [],
            bosToken: bosToken,
            eosToken: eosToken,
            enableThinking: enableThinking
        )
    }

    func format(
        system: String,
        user: String,
        bosToken: String,
        eosToken: String,
        enableThinking: Bool = false
    ) throws -> String {
        try format(
            messages: Self.defaultMessages(system: system, user: user),
            tools: [],
            bosToken: bosToken,
            eosToken: eosToken,
            enableThinking: enableThinking
        )
    }

    func format(
        messages: [ChatTemplateMessage],
        tools: [LLMToolDefinition] = [],
        bosToken: String,
        eosToken: String,
        enableThinking: Bool = false
    ) throws -> String {
        let context: [String: Value] = [
            "messages": .array(messages.map(Self.value(for:))),
            "bos_token": .string(bosToken),
            "eos_token": .string(eosToken),
            "add_generation_prompt": true,
            "enable_thinking": .boolean(enableThinking),
            "tools": .array(tools.map(Self.value(for:)))
        ]

        let formatted = try template.render(context)
        let requiredUserContent = messages
            .filter { $0.role == "user" }
            .compactMap(\.content.stringValue)
            .filter { !$0.isEmpty }
        guard requiredUserContent.allSatisfy({ formatted.contains($0) }) else {
            throw Error.missingUserContent
        }
        return formatted
    }

    private static func defaultMessages(system: String, user: String) -> [ChatTemplateMessage] {
        [
            ChatTemplateMessage(role: "system", content: system),
            ChatTemplateMessage(role: "user", content: user)
        ]
    }

    private static func value(for message: ChatTemplateMessage) -> Value {
        var object: OrderedDictionary<String, Value> = [
            "role": .string(message.role),
            "content": value(for: message.content)
        ]
        if !message.toolCalls.isEmpty {
            object["tool_calls"] = .array(message.toolCalls.map(value(for:)))
        }
        if let toolCallID = message.toolCallID {
            object["tool_call_id"] = .string(toolCallID)
        }
        if let name = message.name {
            object["name"] = .string(name)
        }
        return .object(object)
    }

    private static func value(for call: LLMToolCall) -> Value {
        .object([
            "id": .string(call.id),
            "type": .string("function"),
            "function": .object([
                "name": .string(call.name),
                "arguments": value(for: call.arguments)
            ])
        ])
    }

    private static func value(for tool: LLMToolDefinition) -> Value {
        .object([
            "type": .string("function"),
            "function": .object([
                "name": .string(tool.name),
                "description": .string(tool.description),
                "parameters": value(for: tool.parameters)
            ])
        ])
    }

    private static func value(for json: LLMJSONValue) -> Value {
        switch json {
        case .null:
            return .null
        case .bool(let value):
            return .boolean(value)
        case .number(let value):
            return .double(value)
        case .string(let value):
            return .string(value)
        case .array(let values):
            return .array(values.map(value(for:)))
        case .object(let object):
            var ordered: OrderedDictionary<String, Value> = [:]
            for key in object.keys.sorted() {
                ordered[key] = object[key].map(value(for:)) ?? .null
            }
            return .object(ordered)
        }
    }
}
