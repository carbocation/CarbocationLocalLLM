import CarbocationLocalLLM
import Foundation

struct ChatTemplateMessage: Equatable {
    var role: String
    var content: LLMJSONValue
    var reasoningContent: String?
    var toolCalls: [LLMToolCall]
    var toolCallID: String?
    var name: String?

    init(
        role: String,
        content: String = "",
        reasoningContent: String? = nil,
        toolCalls: [LLMToolCall] = [],
        toolCallID: String? = nil,
        name: String? = nil
    ) {
        self.role = role
        self.content = .string(content)
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.name = name
    }

    init(
        role: String,
        content: LLMJSONValue,
        reasoningContent: String? = nil,
        toolCalls: [LLMToolCall] = [],
        toolCallID: String? = nil,
        name: String? = nil
    ) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.name = name
    }
}

/// Test-fixture renderer for exercising explicit template strings. Production
/// rendering retains a model-lifetime `LlamaCommonChatTemplates` handle.
struct LlamaChatTemplateRenderer {
    enum Error: Swift.Error, LocalizedError, Equatable {
        case notJinjaTemplate
        case emptyPrompt

        var errorDescription: String? {
            switch self {
            case .notJinjaTemplate: return "Embedded chat template is not a Jinja template."
            case .emptyPrompt: return "Applied chat template produced an empty prompt."
            }
        }
    }

    private let source: String

    init(template source: String) throws {
        guard source.contains("{%") || source.contains("{{") || source.contains("{#") else {
            throw Error.notJinjaTemplate
        }
        self.source = source
    }

    static func format(
        template source: String,
        system: String,
        user: String,
        bosToken: String,
        eosToken: String,
        enableThinking: Bool = false,
        reasoningEffort: LLMReasoningEffort? = nil,
        preserveThinking: Bool? = nil
    ) throws -> String {
        try Self(template: source).format(
            messages: defaultMessages(system: system, user: user),
            tools: [],
            bosToken: bosToken,
            eosToken: eosToken,
            enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            preserveThinking: preserveThinking
        )
    }

    func format(
        system: String,
        user: String,
        bosToken: String,
        eosToken: String,
        enableThinking: Bool = false,
        reasoningEffort: LLMReasoningEffort? = nil,
        preserveThinking: Bool? = nil
    ) throws -> String {
        try format(
            messages: Self.defaultMessages(system: system, user: user),
            tools: [],
            bosToken: bosToken,
            eosToken: eosToken,
            enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            preserveThinking: preserveThinking
        )
    }

    func format(
        messages: [ChatTemplateMessage],
        tools: [LLMToolDefinition] = [],
        bosToken: String,
        eosToken: String,
        enableThinking: Bool = false,
        reasoningEffort: LLMReasoningEffort? = nil,
        preserveThinking: Bool? = nil
    ) throws -> String {
        let templates = try LlamaCommonChatTemplates(
            model: nil,
            templateOverride: source,
            bosTokenOverride: bosToken,
            eosTokenOverride: eosToken
        )
        do {
            let plan = try templates.render(
                messages: messages,
                tools: tools,
                options: GenerationOptions(
                    enableThinking: enableThinking,
                    reasoningEffort: reasoningEffort,
                    preserveThinking: preserveThinking
                )
            )
            guard !plan.metadata.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw Error.emptyPrompt
            }
            return plan.metadata.prompt
        } catch let error as Error {
            throw error
        } catch {
            if error.localizedDescription.localizedCaseInsensitiveContains("empty prompt") {
                throw Error.emptyPrompt
            }
            throw error
        }
    }

    static func defaultMessages(system: String, user: String) -> [ChatTemplateMessage] {
        [
            ChatTemplateMessage(role: "system", content: system),
            ChatTemplateMessage(role: "user", content: user)
        ]
    }
}
