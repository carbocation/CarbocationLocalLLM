import CarbocationLocalLLM
import Foundation

extension LlamaEngine {
    public func generateToolCandidate(
        system: String,
        originalPrompt: String,
        tools: [LLMToolDefinition],
        toolChoice: LLMToolChoice,
        history: [LLMToolInteractionRound],
        options: GenerationOptions,
        onPhaseAwareEvent: @Sendable (LLMPhaseAwareStreamEvent) -> Void
    ) async throws -> String {
        if supportsGemmaNativeToolTemplate,
           let promptFormatting = try nativeToolPromptFormatting(
                system: Self.nativeToolCandidateSystemPrompt(
                    system: system,
                    toolChoice: toolChoice
                ),
                originalPrompt: originalPrompt,
                history: history,
                tools: tools,
                options: options
           ) {
            return try await generate(
                promptFormatting: promptFormatting,
                options: options,
                onPhaseAwareEvent: onPhaseAwareEvent
            )
        }

        return try await generate(
            system: LLMToolPromptRenderer.toolAwareSystemPrompt(
                system: system,
                tools: tools,
                toolChoice: toolChoice
            ),
            prompt: LLMToolPromptRenderer.toolAwareUserPrompt(
                originalPrompt: originalPrompt,
                history: history
            ),
            options: options,
            onPhaseAwareEvent: onPhaseAwareEvent
        )
    }

    public func generateToolFinalAnswer(
        system: String,
        originalPrompt: String,
        history: [LLMToolInteractionRound],
        unexecutedCalls: [LLMToolCall],
        maxToolRoundsReached: Bool,
        options: GenerationOptions,
        onPhaseAwareEvent: @Sendable (LLMPhaseAwareStreamEvent) -> Void
    ) async throws -> String {
        if supportsGemmaNativeToolTemplate,
           !history.isEmpty,
           let promptFormatting = try nativeToolPromptFormatting(
                system: Self.nativeToolFinalSystemPrompt(system: system),
                originalPrompt: originalPrompt,
                history: history,
                tools: [],
                finalUserInstruction: Self.nativeToolFinalUserInstruction(
                    unexecutedCalls: unexecutedCalls,
                    maxToolRoundsReached: maxToolRoundsReached
                ),
                options: options
           ) {
            return try await generate(
                promptFormatting: promptFormatting,
                options: options,
                onPhaseAwareEvent: onPhaseAwareEvent
            )
        }

        return try await generate(
            system: system,
            prompt: LLMToolPromptRenderer.toolFinalUserPrompt(
                originalPrompt: originalPrompt,
                history: history,
                unexecutedCalls: unexecutedCalls,
                maxToolRoundsReached: maxToolRoundsReached
            ),
            options: options,
            onPhaseAwareEvent: onPhaseAwareEvent
        )
    }

    var supportsGemmaNativeToolTemplate: Bool {
        guard case .swiftJinja = preparedChatTemplate,
              let chatTemplate else {
            return false
        }
        return chatTemplate.contains("<|tool>")
            && chatTemplate.contains("<|tool_call>call:")
            && chatTemplate.contains("<|tool_response>")
            && chatTemplate.contains("<tool_response|>")
    }

    private func nativeToolPromptFormatting(
        system: String,
        originalPrompt: String,
        history: [LLMToolInteractionRound],
        tools: [LLMToolDefinition],
        finalUserInstruction: String? = nil,
        options: GenerationOptions
    ) throws -> PromptFormattingResult? {
        do {
            return try applyChatTemplate(
                messages: Self.nativeToolMessages(
                    system: system,
                    originalPrompt: originalPrompt,
                    history: history,
                    finalUserInstruction: finalUserInstruction
                ),
                tools: tools,
                options: options
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            llamaRuntimeLog.info(
                "Native Gemma tool template render failed; falling back to text tool prompt: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }

    private static func nativeToolMessages(
        system: String,
        originalPrompt: String,
        history: [LLMToolInteractionRound],
        finalUserInstruction: String?
    ) -> [ChatTemplateMessage] {
        var messages: [ChatTemplateMessage] = []
        let trimmedSystem = system.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedSystem.isEmpty {
            messages.append(ChatTemplateMessage(role: "system", content: trimmedSystem))
        }
        messages.append(ChatTemplateMessage(role: "user", content: originalPrompt))

        for round in history {
            messages.append(ChatTemplateMessage(
                role: "assistant",
                content: "",
                toolCalls: round.calls
            ))
            for output in round.outputs {
                messages.append(ChatTemplateMessage(
                    role: "tool",
                    content: output.content,
                    toolCallID: output.callID,
                    name: output.name
                ))
            }
        }

        if let finalUserInstruction,
           !finalUserInstruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            messages.append(ChatTemplateMessage(role: "user", content: finalUserInstruction))
        }
        return messages
    }

    private static func nativeToolCandidateSystemPrompt(
        system: String,
        toolChoice: LLMToolChoice
    ) -> String {
        var parts: [String] = []
        let trimmedSystem = system.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedSystem.isEmpty {
            parts.append(trimmedSystem)
        }

        var instruction = "You can call the provided tools when they help answer the user. If you call a tool, use the model's native tool-call format and output no prose in that turn."
        switch toolChoice {
        case .auto:
            instruction += " Use tools only when they help answer the user."
        case .none:
            instruction += " Do not call tools."
        case .required:
            instruction += " You must call at least one tool before giving a final answer."
        case .named(let name):
            instruction += " If you call a tool, call only \(name)."
        }
        instruction += " When no more tool calls are needed, respond with only this JSON object and no prose: {\"tool_calls\":[]}."
        parts.append(instruction)
        return parts.joined(separator: "\n\n")
    }

    private static func nativeToolFinalSystemPrompt(system: String) -> String {
        var parts: [String] = []
        let trimmedSystem = system.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedSystem.isEmpty {
            parts.append(trimmedSystem)
        }
        parts.append("Answer the user using the available tool outputs. Do not call tools. Do not output tool-call JSON or native tool-call markup.")
        return parts.joined(separator: "\n\n")
    }

    private static func nativeToolFinalUserInstruction(
        unexecutedCalls: [LLMToolCall],
        maxToolRoundsReached: Bool
    ) -> String? {
        guard maxToolRoundsReached else { return nil }

        let calls = LLMJSONValue.array(unexecutedCalls.map(LLMToolPromptRenderer.jsonValue))
        let serializedCalls = (try? calls.jsonString(prettyPrinted: false)) ?? "[]"
        return """
        Additional tool calls were requested but were not executed because the maximum tool round limit was reached:
        \(serializedCalls)

        Answer the original request using only the available information. Do not call tools.
        """
    }
}
