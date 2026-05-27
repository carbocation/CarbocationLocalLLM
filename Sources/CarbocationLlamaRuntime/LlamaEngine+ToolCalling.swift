import CarbocationLlamaCommonBridge
import CarbocationLocalLLM
import Foundation
import llama

extension LlamaEngine {
    public func generateWithTools(
        _ request: LLMToolGenerationRequest,
        onPhaseAwareEvent: @escaping @Sendable (LLMToolPhaseAwareStreamEvent) -> Void = { _ in }
    ) async throws -> LLMToolGenerationResult {
        try await generateWithTools(
            request,
            control: nil,
            onPhaseAwareEvent: onPhaseAwareEvent
        )
    }

    public func generateWithTools(
        _ request: LLMToolGenerationRequest,
        control: LLMGenerationControl? = nil,
        onPhaseAwareEvent: @escaping @Sendable (LLMToolPhaseAwareStreamEvent) -> Void = { _ in }
    ) async throws -> LLMToolGenerationResult {
        guard vocabulary != nil,
              context != nil || toolAwareGenerationSegmentOverride != nil else {
            throw LLMEngineError.noModelLoaded
        }

        beginGenerationLease()
        defer { endGenerationLease() }

        try LLMToolRuntime.validate(request)
        let stats = LlamaToolGenerationStatsAccumulator()

        func emitAggregateStats(stopReason: String) {
            let snapshot = stats.snapshot(fallbackStopReason: stopReason)
            onPhaseAwareEvent(.aggregateGenerationStats(
                promptTokens: snapshot.promptTokens,
                generatedTokens: snapshot.generatedTokens,
                stopReason: snapshot.stopReason
            ))
            if let accelerationStats = snapshot.accelerationStats {
                onPhaseAwareEvent(.aggregateAccelerationStats(accelerationStats))
            }
        }

        guard !request.tools.isEmpty, request.toolChoice != .none else {
            let text = try await generate(
                system: request.system,
                prompt: request.prompt,
                options: request.options,
                control: control,
                onPhaseAwareEvent: { event in
                    stats.record(event)
                    onPhaseAwareEvent(.finalAnswerEvent(event))
                }
            )
            let snapshot = stats.snapshot(fallbackStopReason: "complete")
            emitAggregateStats(stopReason: snapshot.stopReason)
            return LLMToolGenerationResult(finalText: text, stopReason: snapshot.stopReason)
        }

        let controlGenerationID = control?.beginGeneration()
        defer {
            if let controlGenerationID {
                control?.finishGeneration(controlGenerationID)
            }
        }

        let toolsByName = LLMToolRuntime.index(request.tools)
        var idAllocator = LLMToolCallIDAllocator()
        var history: [LLMToolInteractionRound] = []
        var allCalls: [LLMToolCall] = []
        var allOutputs: [LLMToolOutput] = []
        var roundsCompleted = 0
        let streamState = LlamaToolFinalAnswerStreamState(onPhaseAwareEvent: onPhaseAwareEvent)
        var continuation: LlamaToolContinuation?

        while true {
            let segmentOptions = continuation.map {
                Self.toolContinuationOptions(base: request.options, continuation: $0)
            } ?? request.options
            let promptFormatting = try toolPromptFormatting(
                system: request.system,
                originalPrompt: request.prompt,
                tools: request.tools.map(\.definition),
                toolChoice: request.toolChoice,
                history: history,
                assistantTextSoFar: streamState.text,
                options: segmentOptions
            )
            let segment = try await generateToolAwareSegment(
                promptFormatting: promptFormatting,
                options: segmentOptions,
                streamState: streamState,
                stats: stats,
                interceptTools: true,
                isInternalContinuation: continuation != nil,
                phaseLock: continuation?.triggerPhase == .final ? .final : nil,
                emitDoneOnCompletion: true,
                control: control,
                controlGenerationID: controlGenerationID,
                onPhaseAwareEvent: onPhaseAwareEvent
            )

            let calls = idAllocator.materialize(segment.toolCalls)
            guard !calls.isEmpty else {
                emitAggregateStats(stopReason: "complete")
                return LLMToolGenerationResult(
                    finalText: streamState.text,
                    toolCalls: allCalls,
                    toolOutputs: allOutputs,
                    roundsCompleted: roundsCompleted,
                    stopReason: "complete"
                )
            }

            guard roundsCompleted < request.maxToolRounds else {
                allCalls.append(contentsOf: calls)
                let finalFormatting = try noToolContinuationPromptFormatting(
                    system: request.system,
                    originalPrompt: request.prompt,
                    history: history,
                    assistantTextSoFar: streamState.text,
                    unexecutedCalls: calls,
                    maxToolRoundsReached: true,
                    options: Self.toolContinuationOptions(
                        base: request.options,
                        continuation: LlamaToolContinuation(
                            triggerPhase: segment.triggerPhase,
                            remainingThinkingBudgetTokens: segment.remainingThinkingBudgetTokens
                        )
                    )
                )
                _ = try await generateToolAwareSegment(
                    promptFormatting: finalFormatting,
                    options: Self.toolContinuationOptions(
                        base: request.options,
                        continuation: LlamaToolContinuation(
                            triggerPhase: segment.triggerPhase,
                            remainingThinkingBudgetTokens: segment.remainingThinkingBudgetTokens
                        )
                    ),
                    streamState: streamState,
                    stats: stats,
                    interceptTools: true,
                    isInternalContinuation: true,
                    phaseLock: segment.triggerPhase == .final ? .final : nil,
                    emitDoneOnCompletion: true,
                    emitDoneOnToolInterception: true,
                    control: control,
                    controlGenerationID: controlGenerationID,
                    onPhaseAwareEvent: onPhaseAwareEvent
                )
                emitAggregateStats(stopReason: "max-tool-rounds")
                return LLMToolGenerationResult(
                    finalText: streamState.text,
                    toolCalls: allCalls,
                    toolOutputs: allOutputs,
                    roundsCompleted: roundsCompleted,
                    stopReason: "max-tool-rounds"
                )
            }

            let round = roundsCompleted + 1
            onPhaseAwareEvent(.toolRoundStarted(round: round))
            let outputs = try await LLMToolRuntime.execute(
                calls: calls,
                toolsByName: toolsByName,
                onPhaseAwareEvent: onPhaseAwareEvent
            )

            roundsCompleted = round
            allCalls.append(contentsOf: calls)
            allOutputs.append(contentsOf: outputs)
            history.append(LLMToolInteractionRound(calls: calls, outputs: outputs))
            continuation = LlamaToolContinuation(
                triggerPhase: segment.triggerPhase,
                remainingThinkingBudgetTokens: segment.remainingThinkingBudgetTokens
            )
        }
    }

    public func generateWithToolsPhased(
        _ request: LLMToolGenerationRequest,
        control: LLMGenerationControl? = nil,
        onEvent: @escaping @Sendable (LLMToolPhasedStreamEvent) -> Void = { _ in }
    ) async throws -> LLMToolPhasedGenerationResult {
        guard vocabulary != nil,
              context != nil || toolAwareGenerationSegmentOverride != nil else {
            throw LLMEngineError.noModelLoaded
        }

        beginGenerationLease()
        defer { endGenerationLease() }

        try LLMToolRuntime.validate(request)
        let stats = LlamaToolGenerationStatsAccumulator()

        func emitAggregateStats(stopReason: String) {
            let snapshot = stats.snapshot(fallbackStopReason: stopReason)
            onEvent(.aggregateGenerationStats(
                promptTokens: snapshot.promptTokens,
                generatedTokens: snapshot.generatedTokens,
                stopReason: snapshot.stopReason
            ))
            if let accelerationStats = snapshot.accelerationStats {
                onEvent(.aggregateAccelerationStats(accelerationStats))
            }
        }

        guard !request.tools.isEmpty, request.toolChoice != .none else {
            let generation = try await generatePhased(
                system: request.system,
                prompt: request.prompt,
                options: request.options,
                control: control,
                onEvent: { event in
                    onEvent(.generationEvent(event))
                }
            )
            onEvent(.aggregateGenerationStats(
                promptTokens: generation.promptTokens,
                generatedTokens: generation.generatedTokens,
                stopReason: generation.stopReason
            ))
            if let accelerationStats = generation.accelerationStats {
                onEvent(.aggregateAccelerationStats(accelerationStats))
            }
            return LLMToolPhasedGenerationResult(
                generation: generation,
                stopReason: generation.stopReason
            )
        }

        let controlGenerationID = control?.beginGeneration()
        defer {
            if let controlGenerationID {
                control?.finishGeneration(controlGenerationID)
            }
        }

        let forwardToolEvent: @Sendable (LLMToolPhaseAwareStreamEvent) -> Void = { event in
            switch event {
            case .finalAnswerEvent(let phaseEvent):
                let generationEvent = LLMGenerationStreamEvent(adapting: phaseEvent)
                switch generationEvent {
                case .contentDelta, .contentSnapshot:
                    break
                default:
                    onEvent(.generationEvent(generationEvent))
                }
            case .toolRoundStarted(let round):
                onEvent(.toolRoundStarted(round: round))
            case .toolCallStarted(let call):
                onEvent(.toolCallStarted(call))
            case .toolCallCompleted(let output):
                onEvent(.toolCallCompleted(output))
            case .toolCallFailed(let output):
                onEvent(.toolCallFailed(output))
            case .aggregateGenerationStats(let promptTokens, let generatedTokens, let stopReason):
                onEvent(.aggregateGenerationStats(
                    promptTokens: promptTokens,
                    generatedTokens: generatedTokens,
                    stopReason: stopReason
                ))
            case .aggregateAccelerationStats(let stats):
                onEvent(.aggregateAccelerationStats(stats))
            }
        }

        let toolsByName = LLMToolRuntime.index(request.tools)
        var idAllocator = LLMToolCallIDAllocator()
        var history: [LLMToolInteractionRound] = []
        var allCalls: [LLMToolCall] = []
        var allOutputs: [LLMToolOutput] = []
        var roundsCompleted = 0
        let streamState = LlamaToolFinalAnswerStreamState(onPhaseAwareEvent: forwardToolEvent)
        let phasedStreamState = LlamaToolPhasedGenerationStreamState(onEvent: onEvent)
        var continuation: LlamaToolContinuation?

        func phasedResult(stopReason: String) -> LLMToolPhasedGenerationResult {
            let snapshot = stats.snapshot(fallbackStopReason: stopReason)
            let generation = phasedStreamState.result(
                stopReason: snapshot.stopReason,
                promptTokens: snapshot.promptTokens,
                generatedTokens: snapshot.generatedTokens,
                templateMode: snapshot.templateMode,
                accelerationStats: snapshot.accelerationStats
            )
            return LLMToolPhasedGenerationResult(
                generation: generation,
                toolCalls: allCalls,
                toolOutputs: allOutputs,
                roundsCompleted: roundsCompleted,
                stopReason: stopReason
            )
        }

        while true {
            let segmentOptions = continuation.map {
                Self.toolContinuationOptions(base: request.options, continuation: $0)
            } ?? request.options
            let promptFormatting = try toolPromptFormatting(
                system: request.system,
                originalPrompt: request.prompt,
                tools: request.tools.map(\.definition),
                toolChoice: request.toolChoice,
                history: history,
                assistantTextSoFar: streamState.text,
                options: segmentOptions
            )
            let segment = try await generateToolAwareSegment(
                promptFormatting: promptFormatting,
                options: segmentOptions,
                streamState: streamState,
                phasedStreamState: phasedStreamState,
                stats: stats,
                interceptTools: true,
                isInternalContinuation: continuation != nil,
                phaseLock: continuation?.triggerPhase == .final ? .final : nil,
                emitDoneOnCompletion: true,
                control: control,
                controlGenerationID: controlGenerationID,
                onPhaseAwareEvent: forwardToolEvent
            )

            let calls = idAllocator.materialize(segment.toolCalls)
            guard !calls.isEmpty else {
                emitAggregateStats(stopReason: segment.stopReason)
                return phasedResult(stopReason: segment.stopReason)
            }

            guard roundsCompleted < request.maxToolRounds else {
                allCalls.append(contentsOf: calls)
                let finalFormatting = try noToolContinuationPromptFormatting(
                    system: request.system,
                    originalPrompt: request.prompt,
                    history: history,
                    assistantTextSoFar: streamState.text,
                    unexecutedCalls: calls,
                    maxToolRoundsReached: true,
                    options: Self.toolContinuationOptions(
                        base: request.options,
                        continuation: LlamaToolContinuation(
                            triggerPhase: segment.triggerPhase,
                            remainingThinkingBudgetTokens: segment.remainingThinkingBudgetTokens
                        )
                    )
                )
                _ = try await generateToolAwareSegment(
                    promptFormatting: finalFormatting,
                    options: Self.toolContinuationOptions(
                        base: request.options,
                        continuation: LlamaToolContinuation(
                            triggerPhase: segment.triggerPhase,
                            remainingThinkingBudgetTokens: segment.remainingThinkingBudgetTokens
                        )
                    ),
                    streamState: streamState,
                    phasedStreamState: phasedStreamState,
                    stats: stats,
                    interceptTools: true,
                    isInternalContinuation: true,
                    phaseLock: segment.triggerPhase == .final ? .final : nil,
                    emitDoneOnCompletion: true,
                    emitDoneOnToolInterception: true,
                    control: control,
                    controlGenerationID: controlGenerationID,
                    onPhaseAwareEvent: forwardToolEvent
                )
                emitAggregateStats(stopReason: "max-tool-rounds")
                return phasedResult(stopReason: "max-tool-rounds")
            }

            let round = roundsCompleted + 1
            forwardToolEvent(.toolRoundStarted(round: round))
            let outputs = try await LLMToolRuntime.execute(
                calls: calls,
                toolsByName: toolsByName,
                onPhaseAwareEvent: forwardToolEvent
            )

            roundsCompleted = round
            allCalls.append(contentsOf: calls)
            allOutputs.append(contentsOf: outputs)
            history.append(LLMToolInteractionRound(calls: calls, outputs: outputs))
            continuation = LlamaToolContinuation(
                triggerPhase: segment.triggerPhase,
                remainingThinkingBudgetTokens: segment.remainingThinkingBudgetTokens
            )
        }
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

    private func toolPromptFormatting(
        system: String,
        originalPrompt: String,
        tools: [LLMToolDefinition],
        toolChoice: LLMToolChoice,
        history: [LLMToolInteractionRound],
        assistantTextSoFar: String,
        options: GenerationOptions
    ) throws -> PromptFormattingResult {
        if supportsGemmaNativeToolTemplate,
           let promptFormatting = try nativeToolPromptFormatting(
                system: Self.nativeToolSystemPrompt(
                    system: system,
                    toolChoice: toolChoice
                ),
                originalPrompt: originalPrompt,
                history: history,
                tools: tools,
                finalUserInstruction: Self.nativeToolContinuationInstruction(
                    assistantTextSoFar: assistantTextSoFar,
                    hasToolHistory: !history.isEmpty
                ),
                options: options
           ) {
            return promptFormatting
        }

        return try applyChatTemplate(
            system: LLMToolPromptRenderer.toolAwareSystemPrompt(
                system: system,
                tools: tools,
                toolChoice: toolChoice
            ),
            user: LLMToolPromptRenderer.toolAwareUserPrompt(
                originalPrompt: originalPrompt,
                history: history,
                assistantTextSoFar: assistantTextSoFar
            ),
            options: options
        )
    }

    private func noToolContinuationPromptFormatting(
        system: String,
        originalPrompt: String,
        history: [LLMToolInteractionRound],
        assistantTextSoFar: String,
        unexecutedCalls: [LLMToolCall],
        maxToolRoundsReached: Bool,
        options: GenerationOptions
    ) throws -> PromptFormattingResult {
        if supportsGemmaNativeToolTemplate,
           let promptFormatting = try nativeToolPromptFormatting(
                system: Self.nativeToolFinalSystemPrompt(system: system),
                originalPrompt: originalPrompt,
                history: history,
                tools: [],
                finalUserInstruction: Self.nativeToolFinalUserInstruction(
                    assistantTextSoFar: assistantTextSoFar,
                    unexecutedCalls: unexecutedCalls,
                    maxToolRoundsReached: maxToolRoundsReached
                ),
                options: options
           ) {
            return promptFormatting
        }

        return try applyChatTemplate(
            system: system,
            user: LLMToolPromptRenderer.toolFinalUserPrompt(
                originalPrompt: originalPrompt,
                history: history,
                unexecutedCalls: unexecutedCalls,
                maxToolRoundsReached: maxToolRoundsReached,
                assistantTextSoFar: assistantTextSoFar
            ),
            options: options
        )
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

    private static func nativeToolSystemPrompt(
        system: String,
        toolChoice: LLMToolChoice
    ) -> String {
        var parts: [String] = []
        let trimmedSystem = system.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedSystem.isEmpty {
            parts.append(trimmedSystem)
        }

        var instruction = "You can answer normally. When a provided tool would help, use the model's native tool-call format at the point where the tool is needed."
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
        instruction += " If no tool is needed, answer directly without tool-call markup."
        parts.append(instruction)
        return parts.joined(separator: "\n\n")
    }

    private static func nativeToolContinuationInstruction(
        assistantTextSoFar: String,
        hasToolHistory: Bool
    ) -> String? {
        guard hasToolHistory
                || !assistantTextSoFar.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }

        var instruction = ""
        let trimmedAssistantText = assistantTextSoFar.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedAssistantText.isEmpty {
            instruction += """
            Assistant answer already shown to the user:
            \(trimmedAssistantText)

            """
        }
        instruction += "Use the tool outputs above to continue the answer. If another tool is needed, use native tool-call markup; otherwise continue naturally."
        return instruction
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
        assistantTextSoFar: String,
        unexecutedCalls: [LLMToolCall],
        maxToolRoundsReached: Bool
    ) -> String? {
        var parts: [String] = []
        let trimmedAssistantText = assistantTextSoFar.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedAssistantText.isEmpty {
            parts.append("""
            Assistant answer already shown to the user:
            \(trimmedAssistantText)
            """)
        }

        if maxToolRoundsReached {
            let calls = LLMJSONValue.array(unexecutedCalls.map(LLMToolPromptRenderer.jsonValue))
            let serializedCalls = (try? calls.jsonString(prettyPrinted: false)) ?? "[]"
            parts.append("""
            Additional tool calls were requested but were not executed because the maximum tool round limit was reached:
            \(serializedCalls)
            """)
        }

        parts.append("Continue the answer using only the available information. Do not call tools.")
        return parts.joined(separator: "\n\n")
    }

    private static func toolContinuationOptions(
        base: GenerationOptions,
        continuation: LlamaToolContinuation
    ) -> GenerationOptions {
        var options = base
        var phaseConfiguration = options.streamPhaseConfiguration
        switch continuation.triggerPhase {
        case .thinking:
            options.enableThinking = true
            if let remainingThinkingBudgetTokens = continuation.remainingThinkingBudgetTokens {
                options.thinkingBudgetTokens = max(0, remainingThinkingBudgetTokens)
            }
            phaseConfiguration.startsInThinking = true
        case .final:
            options.enableThinking = false
            options.thinkingBudgetTokens = nil
            phaseConfiguration.startsInThinking = false
        case .unknown, nil:
            if phaseConfiguration.startsInThinking != false {
                phaseConfiguration.startsInThinking = nil
            }
        }
        options.streamPhaseConfiguration = phaseConfiguration
        return options
    }
}

private extension LlamaEngine {
    struct LlamaToolContinuation {
        var triggerPhase: LLMStreamContentPhase?
        var remainingThinkingBudgetTokens: Int?
    }

    struct ToolAwareGenerationSegment {
        var toolCalls: [LLMToolCall]
        var stopReason: String
        var triggerPhase: LLMStreamContentPhase?
        var remainingThinkingBudgetTokens: Int?
    }

    func generateToolAwareSegment(
        promptFormatting: PromptFormattingResult,
        options: GenerationOptions,
        streamState: LlamaToolFinalAnswerStreamState,
        phasedStreamState: LlamaToolPhasedGenerationStreamState? = nil,
        stats: LlamaToolGenerationStatsAccumulator,
        interceptTools: Bool,
        isInternalContinuation: Bool = false,
        phaseLock: LLMStreamContentPhase? = nil,
        emitDoneOnCompletion: Bool,
        emitDoneOnToolInterception: Bool = false,
        control: LLMGenerationControl? = nil,
        controlGenerationID: UInt64? = nil,
        onPhaseAwareEvent: @Sendable (LLMToolPhaseAwareStreamEvent) -> Void
    ) async throws -> ToolAwareGenerationSegment {
        if let override = toolAwareGenerationSegmentOverride {
            streamState.beginSegment()
            phasedStreamState?.beginSegment()
            return try await generateToolAwareSegmentOverride(
                override,
                promptFormatting: promptFormatting,
                streamState: streamState,
                phasedStreamState: phasedStreamState,
                stats: stats,
                isInternalContinuation: isInternalContinuation,
                emitDoneOnCompletion: emitDoneOnCompletion,
                emitDoneOnToolInterception: emitDoneOnToolInterception,
                onPhaseAwareEvent: onPhaseAwareEvent
            )
        }

        guard let context, let vocabulary else {
            throw LLMEngineError.noModelLoaded
        }
        streamState.beginSegment()
        phasedStreamState?.beginSegment()

        var currentPhase = streamState.phase
        func emit(_ event: LLMPhaseAwareStreamEvent) {
            stats.record(event)
            onPhaseAwareEvent(.finalAnswerEvent(event))
        }

        func updatePhase(_ nextPhase: LLMStreamContentPhase) {
            let nextPhase = phaseLock ?? nextPhase
            guard nextPhase != currentPhase else { return }
            let previousPhase = currentPhase
            currentPhase = nextPhase
            streamState.phase = nextPhase
            emit(.phaseChanged(from: previousPhase, to: nextPhase))
        }

        if !streamState.didEmitRequestSent {
            streamState.didEmitRequestSent = true
            emit(.requestSent(phase: currentPhase))
        }
        let startedAt = Date()

        var templateMode: LLMChatTemplateMode = .unavailable
        var promptTokenCount = 0
        var generatedTokenCount = 0
        var stopReason = "cancelled"
        var emittedStats = false
        var emittedAccelerationStats = false
        var accelerationStats: LLMGenerationAccelerationStats?
        var promptContextPrepared = false
        var promptCacheCommitted = false

        func emitAccelerationStatsIfNeeded() {
            guard !emittedAccelerationStats, let accelerationStats else {
                return
            }
            emittedAccelerationStats = true
            emit(.accelerationStats(accelerationStats))
        }

        defer {
            if promptContextPrepared && !promptCacheCommitted {
                clearPromptCaches()
            }
            emitAccelerationStatsIfNeeded()
            if !emittedStats {
                emit(.generationStats(
                    promptTokens: promptTokenCount,
                    generatedTokens: generatedTokenCount,
                    stopReason: stopReason,
                    templateMode: templateMode,
                    phase: currentPhase
                ))
            }
        }

        let renderedPrompt = promptFormatting.text
        templateMode = promptFormatting.mode

        let promptForTokenization = promptWithAutoAddedSpecialTokensStripped(
            renderedPrompt,
            vocab: vocabulary
        )
        let promptTokens = try tokenize(vocab: vocabulary, text: promptForTokenization, addSpecial: true)
        promptTokenCount = promptTokens.count
        guard !promptTokens.isEmpty else {
            throw LLMEngineError.tokenizationFailed
        }
        guard promptTokens.count < currentContextSize() else {
            throw LLMEngineError.insufficientGenerationBudget(
                contextSize: currentContextSize(),
                promptTokens: promptTokens.count,
                reserve: configuration.promptReserveTokens
            )
        }

        let activeOutputProfile = promptFormatting.outputProfile.merging(options.streamPhaseConfiguration)
        let continuingOpenThinkingPairs = Self.continuingOpenThinkingPairs(
            in: renderedPrompt,
            profile: activeOutputProfile
        )
        let streamPhasePlan = StreamPhasePlan(
            profile: activeOutputProfile,
            continuingOpenThinkingPairs: continuingOpenThinkingPairs,
            startsInThinking: options.streamPhaseConfiguration.startsInThinking
        )
        if !isInternalContinuation {
            updatePhase(Self.streamContentPhase(in: "", plan: streamPhasePlan))
        }
        let grammarMode = Self.generationGrammarMode(
            for: options,
            profile: activeOutputProfile,
            continuingOpenThinkingPairs: continuingOpenThinkingPairs
        )
        let structuredOutputPlan = grammarMode.usesLazyGrammar
            ? StructuredOutputPlan(
                profile: activeOutputProfile,
                continuingOpenThinkingPairs: continuingOpenThinkingPairs,
                grammarMode: grammarMode
            )
            : nil
        let reasoningBudgetPlan = Self.reasoningBudgetPlan(
            for: options,
            profile: activeOutputProfile,
            continuingOpenThinkingPairs: continuingOpenThinkingPairs,
            startsInThinking: streamPhasePlan.startsInThinking == true,
            requiresSampler: control != nil
        )
        llamaRuntimeLog.info(
            "Tool-aware generation grammar mode selected: mode=\(grammarMode.logLabel, privacy: .public) enableThinking=\(options.enableThinking, privacy: .public) continuingOpenThinkingPairs=\(continuingOpenThinkingPairs.count, privacy: .public) thinkingBudgetActive=\(reasoningBudgetPlan != nil, privacy: .public)"
        )

        let samplerRuntime = try buildSampler(
            grammarMode: grammarMode,
            options: options,
            vocab: vocabulary,
            reasoningBudgetPlan: reasoningBudgetPlan
        )
        let sampler = samplerRuntime.chain
        defer { llama_sampler_free(sampler) }

        let mtpRuntime = mtpAccelerationRuntime(hasIncompatibleControlPath: false)
        let activeMTPContext = mtpRuntime.context
        accelerationStats = mtpRuntime.stats

        try preparePromptContext(promptTokens, context: context, mtpContext: activeMTPContext)
        promptContextPrepared = true

        var accumulatedData = Data()
        var accumulatedText = ""
        var reasoningBudgetExhaustionLogged = false
        func logReasoningBudgetExhaustionIfNeeded(
            state: carbocation_llama_reasoning_budget_state,
            generatedTokens: Int
        ) {
            guard !reasoningBudgetExhaustionLogged,
                  Self.reasoningBudgetStateIsExhausted(state),
                  let budgetTokens = reasoningBudgetPlan?.budgetTokens else {
                return
            }

            reasoningBudgetExhaustionLogged = true
            llamaRuntimeLog.info(
                "Reasoning budget exhausted: budgetTokens=\(budgetTokens, privacy: .public) generatedTokens=\(generatedTokens, privacy: .public) rawBytes=\(accumulatedData.count, privacy: .public) state=\(Self.reasoningBudgetStateLogLabel(state), privacy: .public)"
            )
        }

        if let reasoningBudgetSampler = samplerRuntime.reasoningBudgetSampler {
            logReasoningBudgetExhaustionIfNeeded(
                state: carbocation_llama_reasoning_budget_sampler_state(reasoningBudgetSampler),
                generatedTokens: generatedTokenCount
            )
        }

        func emitFirstByteIfNeeded() {
            guard !streamState.didEmitFirstByte else { return }
            streamState.didEmitFirstByte = true
            emit(.firstByteReceived(
                after: Date().timeIntervalSince(startedAt),
                phase: currentPhase
            ))
        }

        func sanitizedVisibleText(raw: String) -> String? {
            try? Self.sanitizedGeneratedText(
                raw,
                profile: activeOutputProfile,
                continuingOpenThinkingPairs: continuingOpenThinkingPairs,
                requiresNonEmptyStructuredOutput: false
            )
        }

        var structuredPhase = structuredOutputPlan.map {
            Self.structuredOutputPhase(in: accumulatedText, plan: $0)
        }

        func emitVisibleProgress(raw: String, snapshotReason: LLMFinalAnswerSnapshotReason) {
            let canEmitFinal = Self.shouldEmitFinalAnswerProgress(
                currentPhase: currentPhase,
                structuredPhase: structuredPhase
            )
            let visible = canEmitFinal ? sanitizedVisibleText(raw: raw) : nil
            if let visible {
                streamState.emit(visible, snapshotReason: snapshotReason)
            }
            phasedStreamState?.emit(
                raw: raw,
                plan: streamPhasePlan,
                finalText: visible,
                allowsParserDerivedFinalText: visible != nil,
                snapshotReason: LLMGenerationContentSnapshotReason(snapshotReason)
            )
        }

        let contextMaxNew = Self.maxGenerationTokens(
            contextSize: currentContextSize(),
            promptTokenCount: promptTokens.count,
            reserve: configuration.promptReserveTokens
        )
        let maxNew: Int
        if let requestedMax = options.maxOutputTokens, requestedMax > 0 {
            maxNew = min(contextMaxNew, requestedMax)
        } else {
            maxNew = contextMaxNew
        }

        guard maxNew > 0 else {
            throw LLMEngineError.insufficientGenerationBudget(
                contextSize: currentContextSize(),
                promptTokens: promptTokens.count,
                reserve: configuration.promptReserveTokens
            )
        }

        let effectiveStopSequences = Self.mergingStopSequences(
            options.stopSequences,
            activeOutputProfile.extraStopStrings
        )

        var interceptedToolCalls: [LLMToolCall] = []
        var activeToolCapture: LlamaToolStreamInterpreter.Capture?
        var activeToolCaptureRemainingBudget: Int?
        var pendingNativeToolInterception: LlamaToolStreamInterpreter.Interception?

        func completeToolInterception(_ interception: LlamaToolStreamInterpreter.Interception) {
            let capture = activeToolCapture ?? LlamaToolStreamInterpreter.Capture(
                range: interception.range,
                phase: Self.streamContentPhase(
                    in: String(accumulatedText[..<interception.range.lowerBound]),
                    plan: streamPhasePlan
                )
            )
            let visibleRaw = String(accumulatedText[..<capture.range.lowerBound])
            updatePhase(capture.phase)
            emitFirstByteIfNeeded()
            emitVisibleProgress(raw: visibleRaw, snapshotReason: .streamCorrection)
            interceptedToolCalls = interception.calls.map { call in
                LLMToolCall(
                    executionID: call.executionID,
                    rawID: call.rawID,
                    name: call.name,
                    arguments: call.arguments,
                    triggerPhase: capture.phase
                )
            }
            stopReason = "tool-call-complete"
        }

        func processGeneratedToolToken(
            _ token: llama_token,
            generatedTokensIncludingThisToken: Int,
            reasoningBudgetState: carbocation_llama_reasoning_budget_state?
        ) -> Bool {
            if llama_vocab_is_eog(vocabulary, token) {
                if interceptTools, let pendingNativeToolInterception {
                    completeToolInterception(pendingNativeToolInterception)
                } else {
                    if let reasoningBudgetState {
                        logReasoningBudgetExhaustionIfNeeded(
                            state: reasoningBudgetState,
                            generatedTokens: max(0, generatedTokensIncludingThisToken - 1)
                        )
                    }
                    stopReason = "eog"
                }
                return false
            }

            let rawPiece = tokenToPiece(vocab: vocabulary, token: token)
            let piece = rawPiece.isEmpty
                ? tokenToPiece(vocab: vocabulary, token: token, special: true)
                : rawPiece
            if !piece.isEmpty {
                accumulatedData.append(piece)
                if let decoded = String(data: accumulatedData, encoding: .utf8) {
                    accumulatedText = decoded
                }
            }

            if let reasoningBudgetState {
                logReasoningBudgetExhaustionIfNeeded(
                    state: reasoningBudgetState,
                    generatedTokens: generatedTokensIncludingThisToken
                )
            }

            if !piece.isEmpty {
                if let plan = structuredOutputPlan {
                    let nextPhase = Self.structuredOutputPhase(in: accumulatedText, plan: plan)
                    if nextPhase != structuredPhase {
                        structuredPhase = nextPhase
                        llamaRuntimeLog.info(
                            "Structured output phase changed: phase=\(nextPhase.rawValue, privacy: .public) rawBytes=\(accumulatedData.count, privacy: .public)"
                        )
                    }
                }

                if interceptTools, let capture = activeToolCapture {
                    let capturedText = String(accumulatedText[capture.range.lowerBound...])
                    if !LlamaToolStreamInterpreter.isPotentialStartedToolCall(in: capturedText) {
                        activeToolCapture = nil
                        activeToolCaptureRemainingBudget = nil
                    }
                }

                if interceptTools, activeToolCapture == nil,
                   let capture = LlamaToolStreamInterpreter.startedToolCall(in: accumulatedText) {
                    let visibleRaw = String(accumulatedText[..<capture.range.lowerBound])
                    let triggerPhase = Self.streamContentPhase(in: visibleRaw, plan: streamPhasePlan)
                    activeToolCapture = LlamaToolStreamInterpreter.Capture(
                        range: capture.range,
                        phase: triggerPhase
                    )
                    activeToolCaptureRemainingBudget = samplerRuntime.reasoningBudgetSampler
                        .map { Int(carbocation_llama_reasoning_budget_sampler_remaining($0)) }
                    updatePhase(triggerPhase)
                    emitFirstByteIfNeeded()
                    emitVisibleProgress(raw: visibleRaw, snapshotReason: .streamCorrection)
                }

                if interceptTools,
                   let interception = LlamaToolStreamInterpreter.completedToolCallForStreaming(in: accumulatedText) {
                    pendingNativeToolInterception = nil
                    completeToolInterception(interception)
                    return false
                }

                if interceptTools,
                   let pending = LlamaToolStreamInterpreter.pendingNativeToolCallBatch(in: accumulatedText) {
                    pendingNativeToolInterception = pending
                } else {
                    pendingNativeToolInterception = nil
                }

                let boundary = if let plan = structuredOutputPlan {
                    Self.firstStructuredGenerationBoundary(
                        in: accumulatedText,
                        stopSequences: effectiveStopSequences,
                        stopAtBalancedJSON: options.stopAtBalancedJSON,
                        plan: plan
                    )
                } else {
                    Self.firstGenerationBoundary(
                        in: accumulatedText,
                        stopSequences: effectiveStopSequences,
                        stopAtBalancedJSON: options.stopAtBalancedJSON
                    )
                }

                if let boundary,
                   !(interceptTools
                       && boundary.reason == "tool-call-complete"
                       && pendingNativeToolInterception != nil) {
                    accumulatedText = boundary.text
                    stopReason = boundary.reason
                }

                let safeRaw = interceptTools
                    ? LlamaToolStreamInterpreter.visibleRawPrefix(in: accumulatedText)
                    : accumulatedText
                updatePhase(Self.streamContentPhase(in: safeRaw, plan: streamPhasePlan))

                if stopReason == "json-complete" || stopReason == "stop-sequence" || stopReason == "tool-call-complete" {
                    emitFirstByteIfNeeded()
                    emitVisibleProgress(raw: safeRaw, snapshotReason: .streamCorrection)
                    if stopReason == "json-complete", structuredOutputPlan != nil {
                        structuredPhase = .complete
                    }
                    return false
                }
            }

            emitFirstByteIfNeeded()

            let now = Date()
            if now.timeIntervalSince(streamState.lastHeartbeat) >= configuration.heartbeatInterval {
                streamState.lastHeartbeat = now
                let safeRaw = interceptTools
                    ? LlamaToolStreamInterpreter.visibleRawPrefix(in: accumulatedText)
                    : accumulatedText
                emitVisibleProgress(raw: safeRaw, snapshotReason: .streamCorrection)
                emit(.tokenChunk(
                    preview: String(safeRaw.suffix(60)),
                    bytesSoFar: safeRaw.utf8.count,
                    phase: currentPhase
                ))
            }

            return true
        }

        func decodeSingleToken(_ token: llama_token) throws {
            var oneToken: [llama_token] = [token]
            let decodeResult = oneToken.withUnsafeMutableBufferPointer { buffer -> Int32 in
                let batch = llama_batch_get_one(buffer.baseAddress, 1)
                return llama_decode(context, batch)
            }
            if decodeResult != 0 {
                clearPromptCaches()
                throw LLMEngineError.decodeFailed
            }
        }

        func decodeMTPVerificationTargetTokens(
            _ tokens: [llama_token],
            startPosition: Int,
            mtpContext: UnsafeMutableRawPointer
        ) throws {
            var tokens = tokens
            let decodeResult = tokens.withUnsafeMutableBufferPointer { buffer -> Int32 in
                carbocation_llama_mtp_decode_verification_target_tokens_bridge(
                    mtpContext,
                    buffer.baseAddress,
                    Int32(buffer.count),
                    Int32(startPosition)
                )
            }
            if decodeResult != 0 {
                clearPromptCaches()
                throw LLMEngineError.decodeFailed
            }
        }

        func processLastMTPVerificationBatch(
            mtpContext: UnsafeMutableRawPointer
        ) throws {
            let processResult = carbocation_llama_mtp_process_last_target_batch_bridge(mtpContext)
            if processResult != 0 {
                clearPromptCaches()
                throw LLMEngineError.decodeFailed
            }
        }

        struct PendingMTPToken {
            var token: llama_token
            var reasoningBudgetState: carbocation_llama_reasoning_budget_state?
            var alreadyProcessed: Bool
        }

        var pendingNextToken: PendingMTPToken?
        var generatedTokenHistory: [llama_token] = []

        func sampleTargetToken(at index: Int32) -> llama_token {
            llama_synchronize(context)
            return llama_sampler_sample(sampler, context, index)
        }

        let mtpDiagnosticsEnabled = configuration.emitsMTPDiagnostics
            && activeMTPContext != nil
            && (accelerationStats?.maxDraftTokens ?? 0) > 2
        func emitMTPDiagnostic(_ message: String) {
            guard mtpDiagnosticsEnabled else { return }
            emit(.diagnostic(message: "mtp-diagnostic \(message)"))
        }

        while generatedTokenCount < maxNew {
            try Task.checkCancellation()
            if accumulatedData.isEmpty || String(data: accumulatedData, encoding: .utf8) != nil {
                applyThinkingTerminationIfRequested(
                    control: control,
                    generationID: controlGenerationID,
                    currentPhase: currentPhase,
                    samplerRuntime: samplerRuntime,
                    reasoningBudgetPlan: reasoningBudgetPlan,
                    vocab: vocabulary
                )
            }

            let tokenPosition = promptTokens.count + generatedTokenHistory.count
            let next: llama_token
            let reasoningBudgetState: carbocation_llama_reasoning_budget_state?
            let nextAlreadyProcessed: Bool
            if let pending = pendingNextToken {
                next = pending.token
                reasoningBudgetState = pending.reasoningBudgetState
                nextAlreadyProcessed = pending.alreadyProcessed
                pendingNextToken = nil
            } else {
                next = sampleTargetToken(at: -1)
                reasoningBudgetState = samplerRuntime.reasoningBudgetSampler.map {
                    carbocation_llama_reasoning_budget_sampler_state($0)
                }
                nextAlreadyProcessed = false
            }
            var emittedNextTokenCount = 0
            if !nextAlreadyProcessed {
                guard processGeneratedToolToken(
                    next,
                    generatedTokensIncludingThisToken: generatedTokenCount + 1,
                    reasoningBudgetState: reasoningBudgetState
                ) else {
                    break
                }
                emittedNextTokenCount = 1
            }

            if let activeMTPContext {
                let remainingDraftCapacity = maxNew - generatedTokenCount - emittedNextTokenCount
                let draftLimit = min(accelerationStats?.maxDraftTokens ?? 0, remainingDraftCapacity)
                var draftCallIndex: Int?
                let draftTokens: [llama_token]
                if draftLimit > 0 {
                    draftCallIndex = (accelerationStats?.draftCalls ?? 0) + 1
                    var draftBuffer = [llama_token](repeating: 0, count: draftLimit)
                    let draftCount = draftBuffer.withUnsafeMutableBufferPointer { buffer in
                        carbocation_llama_mtp_draft_bridge(
                            activeMTPContext,
                            next,
                            Int32(tokenPosition),
                            buffer.baseAddress,
                            Int32(buffer.count)
                        )
                    }
                    draftTokens = Array(draftBuffer.prefix(min(draftBuffer.count, max(0, Int(draftCount)))))
                    accelerationStats?.draftCalls += 1
                    accelerationStats?.draftTokensGenerated += draftTokens.count
                    emitMTPDiagnostic(
                        "draft call=\(draftCallIndex ?? 0) pos=\(tokenPosition) next=\(diagnosticTokenDescription(next, vocab: vocabulary)) draft=\(diagnosticTokenList(draftTokens, vocab: vocabulary))"
                    )
                } else {
                    draftTokens = []
                }

                var emittedAcceptedDraftCount = 0
                var emittedCorrectionToken: llama_token?
                var shouldContinue = true

                if draftTokens.isEmpty {
                    try decodeMTPVerificationTargetTokens(
                        [next],
                        startPosition: tokenPosition,
                        mtpContext: activeMTPContext
                    )
                    if generatedTokenCount + emittedNextTokenCount < maxNew {
                        let sampledPrefetch = sampleTargetToken(at: 0)
                        pendingNextToken = PendingMTPToken(
                            token: sampledPrefetch,
                            reasoningBudgetState: samplerRuntime.reasoningBudgetSampler.map {
                                carbocation_llama_reasoning_budget_sampler_state($0)
                            },
                            alreadyProcessed: false
                        )
                    }
                    try processLastMTPVerificationBatch(mtpContext: activeMTPContext)
                } else {
                    let verificationPrefixTokens = promptTokens + generatedTokenHistory
                    let verificationTokens = [next] + draftTokens
                    try decodeMTPVerificationTargetTokens(
                        verificationTokens,
                        startPosition: tokenPosition,
                        mtpContext: activeMTPContext
                    )

                    var correctionToken: llama_token?
                    var correctionReasoningBudgetState: carbocation_llama_reasoning_budget_state?
                    var verifierSamples: [llama_token] = []
                    for (verifierIndex, draftToken) in draftTokens.enumerated() {
                        let sampled = sampleTargetToken(at: Int32(verifierIndex))
                        verifierSamples.append(sampled)
                        let acceptedReasoningBudgetState = samplerRuntime.reasoningBudgetSampler.map {
                            carbocation_llama_reasoning_budget_sampler_state($0)
                        }

                        if sampled != draftToken {
                            correctionToken = sampled
                            correctionReasoningBudgetState = acceptedReasoningBudgetState
                            break
                        }

                        emittedAcceptedDraftCount += 1
                        shouldContinue = processGeneratedToolToken(
                            draftToken,
                            generatedTokensIncludingThisToken: generatedTokenCount + emittedNextTokenCount + emittedAcceptedDraftCount,
                            reasoningBudgetState: acceptedReasoningBudgetState
                        )
                        if !shouldContinue {
                            break
                        }
                    }

                    if emittedAcceptedDraftCount < draftTokens.count {
                        let retainedTokens = [next] + Array(draftTokens.prefix(emittedAcceptedDraftCount))
                        try processLastMTPVerificationBatch(mtpContext: activeMTPContext)
                        _ = carbocation_llama_mtp_accept_bridge(
                            activeMTPContext,
                            Int32(emittedAcceptedDraftCount)
                        )
                        let rollbackPosition = tokenPosition + retainedTokens.count
                        let rollbackResult = carbocation_llama_mtp_rollback_bridge(
                            activeMTPContext,
                            Int32(rollbackPosition)
                        )
                        if rollbackResult == 0 {
                            clearPromptCaches()
                            throw LLMEngineError.decodeFailed
                        }

                        if let correctionToken, shouldContinue {
                            emittedCorrectionToken = correctionToken
                            shouldContinue = processGeneratedToolToken(
                                correctionToken,
                                generatedTokensIncludingThisToken: generatedTokenCount + emittedNextTokenCount + emittedAcceptedDraftCount + 1,
                                reasoningBudgetState: correctionReasoningBudgetState
                            )
                            if shouldContinue {
                                pendingNextToken = PendingMTPToken(
                                    token: correctionToken,
                                    reasoningBudgetState: correctionReasoningBudgetState,
                                    alreadyProcessed: true
                                )
                            }
                        }
                        if let command = diagnosticBatchEquivalenceCommand(
                            modelPath: loadedDescriptor?.url.path,
                            prefixTokens: verificationPrefixTokens,
                            windowTokens: verificationTokens,
                            recurrentStateSnapshots: accelerationStats?.maxDraftTokens ?? 0,
                            samplerPrefixSkip: promptTokens.count
                        ) {
                            emitMTPDiagnostic("repro-command \(command)")
                        }
                        emitMTPDiagnostic(
                            "repro call=\(draftCallIndex ?? 0) pos=\(tokenPosition) prefix-token-count=\(verificationPrefixTokens.count) prefix-token-ids=\(diagnosticTokenIDList(verificationPrefixTokens)) window-token-ids=\(diagnosticTokenIDList(verificationTokens))"
                        )
                        emitMTPDiagnostic(
                            "verify call=\(draftCallIndex ?? 0) pos=\(tokenPosition) accepted=\(emittedAcceptedDraftCount)/\(draftTokens.count) draft=\(diagnosticTokenList(draftTokens, vocab: vocabulary)) sampled=\(diagnosticTokenList(verifierSamples, vocab: vocabulary)) correction=\(correctionToken.map { diagnosticTokenDescription($0, vocab: vocabulary) } ?? "none")"
                        )
                    } else {
                        var prefetchedDiagnosticToken: llama_token?
                        if shouldContinue,
                           generatedTokenCount + emittedNextTokenCount + emittedAcceptedDraftCount < maxNew {
                            let sampledPrefetch = sampleTargetToken(at: Int32(draftTokens.count))
                            prefetchedDiagnosticToken = sampledPrefetch
                            pendingNextToken = PendingMTPToken(
                                token: sampledPrefetch,
                                reasoningBudgetState: samplerRuntime.reasoningBudgetSampler.map {
                                    carbocation_llama_reasoning_budget_sampler_state($0)
                                },
                                alreadyProcessed: false
                            )
                        }

                        try processLastMTPVerificationBatch(mtpContext: activeMTPContext)

                        _ = carbocation_llama_mtp_accept_bridge(
                            activeMTPContext,
                            Int32(emittedAcceptedDraftCount)
                        )
                        emitMTPDiagnostic(
                            "verify call=\(draftCallIndex ?? 0) pos=\(tokenPosition) accepted=full/\(draftTokens.count) draft=\(diagnosticTokenList(draftTokens, vocab: vocabulary)) sampled=\(diagnosticTokenList(verifierSamples, vocab: vocabulary)) prefetch=\(prefetchedDiagnosticToken.map { diagnosticTokenDescription($0, vocab: vocabulary) } ?? "none")"
                        )
                    }
                }

                accelerationStats?.draftTokensAccepted += emittedAcceptedDraftCount
                let retainedTokenCount = emittedNextTokenCount + emittedAcceptedDraftCount + (emittedCorrectionToken == nil ? 0 : 1)
                generatedTokenHistory.append(next)
                if emittedAcceptedDraftCount > 0 {
                    generatedTokenHistory.append(contentsOf: draftTokens.prefix(emittedAcceptedDraftCount))
                }
                generatedTokenCount += retainedTokenCount
                if !shouldContinue {
                    break
                }
                continue
            }

            try decodeSingleToken(next)

            generatedTokenHistory.append(next)
            generatedTokenCount += 1
        }

        if interceptedToolCalls.isEmpty,
           interceptTools,
           let pendingNativeToolInterception {
            completeToolInterception(pendingNativeToolInterception)
        }

        if stopReason == "cancelled" {
            stopReason = "max-tokens"
        }

        if accumulatedText.isEmpty, !accumulatedData.isEmpty {
            accumulatedText = String(decoding: accumulatedData, as: UTF8.self)
        }

        let rawForReturn: String
        if !interceptedToolCalls.isEmpty,
           let interception = LlamaToolStreamInterpreter.completedToolCall(in: accumulatedText) {
            rawForReturn = String(accumulatedText[..<interception.range.lowerBound])
        } else {
            rawForReturn = interceptTools
                ? LlamaToolStreamInterpreter.visibleRawPrefix(in: accumulatedText)
                : accumulatedText
        }

        updatePhase(Self.streamContentPhase(in: rawForReturn, plan: streamPhasePlan))
        if stopReason == "max-tokens", currentPhase == .thinking {
            stopReason = "thinking-not-closed"
        }
        let shouldReturnIncompleteThinking = stopReason == "thinking-not-closed"

        if interceptedToolCalls.isEmpty, let plan = structuredOutputPlan {
            let finalPhase = structuredPhase ?? Self.structuredOutputPhase(in: rawForReturn, plan: plan)
            if finalPhase == .thinking || finalPhase == .awaitingFinal {
                stopReason = finalPhase == .thinking
                    ? "thinking-not-closed"
                    : "structured-output-not-started"
                llamaRuntimeLog.error(
                    "Structured output phase failed before sanitization: phase=\(finalPhase.rawValue, privacy: .public) rawBytes=\(rawForReturn.utf8.count, privacy: .public) stopReason=\(stopReason, privacy: .public)"
                )
                if !(shouldReturnIncompleteThinking && finalPhase == .thinking) {
                    throw LLMEngineError.structuredOutputPhaseFailed(
                        "Generation ended before final structured output began."
                    )
                }
            }
        }

        let phasedText = Self.phasedGeneratedText(rawForReturn, plan: streamPhasePlan)
        let returnedText: String
        if stopReason == "thinking-not-closed" {
            returnedText = phasedText.finalText
        } else {
            do {
                returnedText = try Self.sanitizedGeneratedText(
                    rawForReturn,
                    profile: activeOutputProfile,
                    continuingOpenThinkingPairs: continuingOpenThinkingPairs,
                    requiresNonEmptyStructuredOutput: options.grammar != nil && !activeOutputProfile.isEmpty
                )
            } catch let error as LLMEngineError {
                if case .structuredOutputPhaseFailed = error {
                    stopReason = "structured-sanitization-empty"
                    llamaRuntimeLog.error(
                        "Structured output sanitization failed: rawBytes=\(rawForReturn.utf8.count, privacy: .public) stopReason=\(stopReason, privacy: .public)"
                    )
                }
                throw error
            }
        }

        if interceptedToolCalls.isEmpty {
            streamState.emit(returnedText, snapshotReason: .completed)
            phasedStreamState?.emit(
                phasedText: phasedText.replacingPhaseText(.final, with: returnedText),
                snapshotReason: .completed
            )
        } else if Self.shouldEmitFinalAnswerProgress(
            currentPhase: currentPhase,
            structuredPhase: structuredPhase
        ) {
            streamState.emit(returnedText, snapshotReason: .streamCorrection)
            phasedStreamState?.emit(
                phasedText: phasedText.replacingPhaseText(.final, with: returnedText),
                snapshotReason: .streamCorrection
            )
        } else {
            phasedStreamState?.emit(
                phasedText: phasedText.replacingPhaseText(.final, with: ""),
                snapshotReason: .streamCorrection
            )
        }

        llamaRuntimeLog.info(
            "Tool-aware generation sanitized output: grammarMode=\(grammarMode.logLabel, privacy: .public) rawBytes=\(rawForReturn.utf8.count, privacy: .public) sanitizedBytes=\(returnedText.utf8.count, privacy: .public) stopReason=\(stopReason, privacy: .public)"
        )

        commitPromptCache(promptTokens, mtpSynchronized: activeMTPContext != nil)
        promptCacheCommitted = true
        emitAccelerationStatsIfNeeded()
        emittedStats = true
        emit(.generationStats(
            promptTokens: promptTokens.count,
            generatedTokens: generatedTokenCount,
            stopReason: stopReason,
            templateMode: templateMode,
            phase: currentPhase
        ))
        if emitDoneOnCompletion,
           interceptedToolCalls.isEmpty || emitDoneOnToolInterception {
            emit(.done(
                totalBytes: streamState.text.utf8.count,
                duration: Date().timeIntervalSince(startedAt),
                phase: currentPhase
            ))
        }
        return ToolAwareGenerationSegment(
            toolCalls: interceptedToolCalls,
            stopReason: stopReason,
            triggerPhase: interceptedToolCalls.first?.triggerPhase,
            remainingThinkingBudgetTokens: activeToolCaptureRemainingBudget.flatMap { $0 >= 0 ? $0 : nil }
        )
    }

    func generateToolAwareSegmentOverride(
        _ override: ToolAwareGenerationSegmentOverride,
        promptFormatting: PromptFormattingResult,
        streamState: LlamaToolFinalAnswerStreamState,
        phasedStreamState: LlamaToolPhasedGenerationStreamState?,
        stats: LlamaToolGenerationStatsAccumulator,
        isInternalContinuation: Bool,
        emitDoneOnCompletion: Bool,
        emitDoneOnToolInterception: Bool,
        onPhaseAwareEvent: @Sendable (LLMToolPhaseAwareStreamEvent) -> Void
    ) async throws -> ToolAwareGenerationSegment {
        let startedAt = Date()
        var currentPhase = streamState.phase

        func emit(_ event: LLMPhaseAwareStreamEvent) {
            stats.record(event)
            onPhaseAwareEvent(.finalAnswerEvent(event))
        }

        if !streamState.didEmitRequestSent {
            streamState.didEmitRequestSent = true
            emit(.requestSent(phase: currentPhase))
        }

        let output = try await override(ToolAwareGenerationSegmentOverrideInput(
            renderedPrompt: promptFormatting.text,
            templateMode: promptFormatting.mode,
            isInternalContinuation: isInternalContinuation
        ))

        if let triggerPhase = output.triggerPhase {
            currentPhase = triggerPhase
            streamState.phase = triggerPhase
        }

        if output.finalText != nil, !streamState.didEmitFirstByte {
            streamState.didEmitFirstByte = true
            emit(.firstByteReceived(
                after: Date().timeIntervalSince(startedAt),
                phase: currentPhase
            ))
        }

        if let finalText = output.finalText {
            streamState.emit(
                finalText,
                snapshotReason: output.toolCalls.isEmpty ? .completed : .streamCorrection
            )
            phasedStreamState?.emitFinalText(
                finalText,
                rawGeneratedText: finalText,
                snapshotReason: output.toolCalls.isEmpty ? .completed : .streamCorrection
            )
        }

        emit(.generationStats(
            promptTokens: 0,
            generatedTokens: output.generatedTokens,
            stopReason: output.stopReason,
            templateMode: promptFormatting.mode,
            phase: currentPhase
        ))

        if emitDoneOnCompletion,
           output.toolCalls.isEmpty || emitDoneOnToolInterception {
            emit(.done(
                totalBytes: streamState.text.utf8.count,
                duration: Date().timeIntervalSince(startedAt),
                phase: currentPhase
            ))
        }

        return ToolAwareGenerationSegment(
            toolCalls: output.toolCalls,
            stopReason: output.stopReason,
            triggerPhase: output.triggerPhase,
            remainingThinkingBudgetTokens: output.remainingThinkingBudgetTokens
        )
    }
}

private final class LlamaToolFinalAnswerStreamState: @unchecked Sendable {
    private let lock = NSLock()
    private let onPhaseAwareEvent: @Sendable (LLMToolPhaseAwareStreamEvent) -> Void
    private var committedFinalAnswer = ""
    private var currentSegmentText = ""
    private var storedPhase = LLMStreamContentPhase.unknown
    private var emittedRequestSent = false
    private var emittedFirstByte = false
    var lastHeartbeat = Date()

    init(onPhaseAwareEvent: @escaping @Sendable (LLMToolPhaseAwareStreamEvent) -> Void) {
        self.onPhaseAwareEvent = onPhaseAwareEvent
    }

    var text: String {
        lock.lock()
        defer { lock.unlock() }
        return committedFinalAnswer + currentSegmentText
    }

    var phase: LLMStreamContentPhase {
        get {
            lock.lock()
            defer { lock.unlock() }
            return storedPhase
        }
        set {
            lock.lock()
            storedPhase = newValue
            lock.unlock()
        }
    }

    var didEmitRequestSent: Bool {
        get {
            lock.lock()
            defer { lock.unlock() }
            return emittedRequestSent
        }
        set {
            lock.lock()
            emittedRequestSent = newValue
            lock.unlock()
        }
    }

    var didEmitFirstByte: Bool {
        get {
            lock.lock()
            defer { lock.unlock() }
            return emittedFirstByte
        }
        set {
            lock.lock()
            emittedFirstByte = newValue
            lock.unlock()
        }
    }

    func beginSegment() {
        lock.lock()
        committedFinalAnswer += currentSegmentText
        currentSegmentText = ""
        lock.unlock()
    }

    func emit(_ segmentText: String, snapshotReason: LLMFinalAnswerSnapshotReason) {
        lock.lock()
        let previousSegmentText = currentSegmentText
        let previousFinalAnswer = committedFinalAnswer + previousSegmentText
        let nextFinalAnswer = committedFinalAnswer + segmentText
        guard nextFinalAnswer != previousFinalAnswer else {
            lock.unlock()
            return
        }
        currentSegmentText = segmentText
        lock.unlock()

        if segmentText.hasPrefix(previousSegmentText) {
            let delta = String(segmentText.dropFirst(previousSegmentText.count))
            if !delta.isEmpty {
                onPhaseAwareEvent(.finalAnswerEvent(.finalAnswerDelta(
                    text: delta,
                    bytesSoFar: nextFinalAnswer.utf8.count
                )))
            }
        } else {
            onPhaseAwareEvent(.finalAnswerEvent(.finalAnswerSnapshot(
                text: nextFinalAnswer,
                bytesSoFar: nextFinalAnswer.utf8.count,
                reason: snapshotReason
            )))
        }
    }
}

final class LlamaToolPhasedGenerationStreamState: @unchecked Sendable {
    private let lock = NSLock()
    private let onEvent: @Sendable (LLMToolPhasedStreamEvent) -> Void
    private var committedSegments: [LLMGenerationPhaseSegment] = []
    private var currentSegments: [LLMGenerationPhaseSegment] = []
    private var committedRawGeneratedText: [String] = []
    private var currentRawGeneratedText: String?
    private var emittedPhaseText: [LLMStreamContentPhase: String] = [:]

    init(onEvent: @escaping @Sendable (LLMToolPhasedStreamEvent) -> Void) {
        self.onEvent = onEvent
    }

    func beginSegment() {
        lock.lock()
        committedSegments.append(contentsOf: currentSegments)
        currentSegments = []
        if let currentRawGeneratedText, !currentRawGeneratedText.isEmpty {
            committedRawGeneratedText.append(currentRawGeneratedText)
        }
        currentRawGeneratedText = nil
        lock.unlock()
    }

    func emit(
        raw: String,
        plan: LlamaEngine.StreamPhasePlan,
        finalText: String?,
        allowsParserDerivedFinalText: Bool = true,
        snapshotReason: LLMGenerationContentSnapshotReason
    ) {
        var phasedText = LlamaEngine.phasedGeneratedText(raw, plan: plan)
        if let finalText {
            phasedText = phasedText.replacingPhaseText(.final, with: finalText)
        } else if !allowsParserDerivedFinalText {
            phasedText = phasedText.replacingPhaseText(.final, with: "")
        }
        emit(phasedText: phasedText, snapshotReason: snapshotReason)
    }

    func emitFinalText(
        _ finalText: String,
        rawGeneratedText: String,
        snapshotReason: LLMFinalAnswerSnapshotReason
    ) {
        emit(
            phasedText: LlamaEngine.PhasedGenerationText(
                thinkingText: "",
                finalText: finalText,
                phaseSegments: finalText.isEmpty
                    ? []
                    : [LLMGenerationPhaseSegment(phase: .final, text: finalText)],
                rawGeneratedText: rawGeneratedText
            ),
            snapshotReason: LLMGenerationContentSnapshotReason(snapshotReason)
        )
    }

    func emit(
        phasedText: LlamaEngine.PhasedGenerationText,
        snapshotReason: LLMGenerationContentSnapshotReason
    ) {
        var events: [LLMGenerationStreamEvent] = []
        lock.lock()
        currentSegments = phasedText.phaseSegments
        currentRawGeneratedText = phasedText.rawGeneratedText
        let allSegments = committedSegments + currentSegments

        for phase in [LLMStreamContentPhase.thinking, .final] {
            let nextText = Self.joinedText(for: phase, in: allSegments)
            let previousText = emittedPhaseText[phase] ?? ""
            guard nextText != previousText else { continue }

            if nextText.hasPrefix(previousText) {
                let delta = String(nextText.dropFirst(previousText.count))
                if !delta.isEmpty {
                    events.append(.contentDelta(
                        phase: phase,
                        text: delta,
                        bytesSoFar: nextText.utf8.count
                    ))
                }
            } else {
                events.append(.contentSnapshot(
                    phase: phase,
                    text: nextText,
                    bytesSoFar: nextText.utf8.count,
                    reason: snapshotReason
                ))
            }
            emittedPhaseText[phase] = nextText
        }
        lock.unlock()

        for event in events {
            onEvent(.generationEvent(event))
        }
    }

    func result(
        stopReason: String,
        promptTokens: Int,
        generatedTokens: Int,
        templateMode: LLMChatTemplateMode,
        accelerationStats: LLMGenerationAccelerationStats?
    ) -> LLMGenerationResult {
        lock.lock()
        defer { lock.unlock() }

        let allSegments = committedSegments + currentSegments
        var rawSegments = committedRawGeneratedText
        if let currentRawGeneratedText, !currentRawGeneratedText.isEmpty {
            rawSegments.append(currentRawGeneratedText)
        }
        let rawGeneratedText = rawSegments.isEmpty ? nil : rawSegments.joined(separator: "\n")

        return LLMGenerationResult(
            thinkingText: Self.joinedText(for: .thinking, in: allSegments),
            finalText: Self.joinedText(for: .final, in: allSegments),
            phaseSegments: allSegments,
            stopReason: stopReason,
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            templateMode: templateMode,
            accelerationStats: accelerationStats,
            rawGeneratedText: rawGeneratedText
        )
    }

    private static func joinedText(
        for phase: LLMStreamContentPhase,
        in segments: [LLMGenerationPhaseSegment]
    ) -> String {
        segments
            .filter { $0.phase == phase }
            .map(\.text)
            .joined(separator: "\n")
    }
}

private final class LlamaToolGenerationStatsAccumulator: @unchecked Sendable {
    private let lock = NSLock()
    private var promptTokens = 0
    private var generatedTokens = 0
    private var stopReason: String?
    private var templateMode = LLMChatTemplateMode.unavailable
    private var accelerationStats: LLMGenerationAccelerationStats?

    func record(_ event: LLMPhaseAwareStreamEvent) {
        switch event {
        case .generationStats(let promptTokens, let generatedTokens, let stopReason, let templateMode, _):
            lock.lock()
            defer { lock.unlock() }
            self.promptTokens += promptTokens
            self.generatedTokens += generatedTokens
            self.stopReason = stopReason
            self.templateMode = templateMode
        case .accelerationStats(let stats):
            lock.lock()
            defer { lock.unlock() }
            if accelerationStats == nil {
                accelerationStats = stats
            } else {
                accelerationStats?.merge(stats)
            }
        default:
            break
        }
    }

    func snapshot(
        fallbackStopReason: String
    ) -> (
        promptTokens: Int,
        generatedTokens: Int,
        stopReason: String,
        templateMode: LLMChatTemplateMode,
        accelerationStats: LLMGenerationAccelerationStats?
    ) {
        lock.lock()
        defer { lock.unlock() }
        return (
            promptTokens,
            generatedTokens,
            stopReason ?? fallbackStopReason,
            templateMode,
            accelerationStats
        )
    }
}

struct LlamaToolStreamInterpreter {
    struct Capture {
        var range: Range<String.Index>
        var phase: LLMStreamContentPhase
    }

    struct Interception {
        var range: Range<String.Index>
        var calls: [LLMToolCall]
    }

    private static let nativeStartMarker = "<|tool_call>call:"
    private static let nativeEndMarker = "<tool_call|>"
    private static let jsonToolObjectKeys = [
        #""tool_calls""#,
        #""function""#,
        #""name""#,
        #""tool_name""#
    ]

    static func completedToolCall(in text: String) -> Interception? {
        completedToolCall(in: text, deferPendingNativeBatch: false)
    }

    static func completedToolCallForStreaming(in text: String) -> Interception? {
        completedToolCall(in: text, deferPendingNativeBatch: true)
    }

    static func pendingNativeToolCallBatch(in text: String) -> Interception? {
        guard let nativeRange = LlamaEngine.nativeToolCallEnvelopeRange(in: text),
              isPendingNativeBatch(in: text, range: nativeRange) else {
            return nil
        }

        let calls = LLMToolCallParser.parseToolCalls(in: String(text[nativeRange]))
        guard !calls.isEmpty else { return nil }
        return Interception(range: nativeRange, calls: calls)
    }

    private static func completedToolCall(
        in text: String,
        deferPendingNativeBatch: Bool
    ) -> Interception? {
        var earliest: Interception?
        var pendingNativeRange: Range<String.Index>?
        if let nativeRange = LlamaEngine.nativeToolCallEnvelopeRange(in: text) {
            let calls = LLMToolCallParser.parseToolCalls(in: String(text[nativeRange]))
            if !calls.isEmpty {
                if deferPendingNativeBatch, isPendingNativeBatch(in: text, range: nativeRange) {
                    pendingNativeRange = nativeRange
                } else {
                    earliest = Interception(range: nativeRange, calls: calls)
                }
            }
        }

        var searchStart = text.startIndex
        while let jsonRange = LlamaEngine.balancedJSONValueRange(
            in: text,
            searchRange: searchStart..<text.endIndex
        ) {
            if let pendingNativeRange,
               pendingNativeRange.lowerBound < jsonRange.lowerBound {
                break
            }
            let calls = LLMToolCallParser.parseToolCalls(in: String(text[jsonRange]))
            if !calls.isEmpty,
               earliest.map({ jsonRange.lowerBound < $0.range.lowerBound }) ?? true {
                earliest = Interception(range: jsonRange, calls: calls)
                break
            }
            searchStart = jsonRange.upperBound
        }

        return earliest
    }

    private static func isPendingNativeBatch(
        in text: String,
        range: Range<String.Index>
    ) -> Bool {
        var index = range.upperBound
        while index < text.endIndex, text[index].isWhitespace {
            index = text.index(after: index)
        }
        guard index < text.endIndex else {
            return true
        }

        let suffix = String(text[index...])
        return nativeStartMarker.hasPrefix(suffix) || suffix.hasPrefix(nativeStartMarker)
    }

    static func completedToolCall(
        in text: String,
        phasePlan: LlamaEngine.StreamPhasePlan
    ) -> Interception? {
        guard let interception = completedToolCall(in: text) else { return nil }
        let prefix = String(text[..<interception.range.lowerBound])
        let phase = LlamaEngine.streamContentPhase(in: prefix, plan: phasePlan)
        return Interception(
            range: interception.range,
            calls: interception.calls.map { call in
                LLMToolCall(
                    executionID: call.executionID,
                    rawID: call.rawID,
                    name: call.name,
                    arguments: call.arguments,
                    triggerPhase: phase
                )
            }
        )
    }

    static func startedToolCall(in text: String) -> Capture? {
        guard !text.isEmpty else { return nil }
        var start: String.Index?
        if let complete = completedToolCall(in: text) {
            start = minIndex(start, complete.range.lowerBound, in: text)
        }
        if let unclosedNativeStart = unclosedNativeToolCallStart(in: text) {
            start = minIndex(start, unclosedNativeStart, in: text)
        }
        if let possibleJSONStart = possibleIncompleteToolJSONStart(in: text) {
            start = minIndex(start, possibleJSONStart, in: text)
        }
        if let suffixStart = possibleToolMarkerSuffixStart(in: text) {
            start = minIndex(start, suffixStart, in: text)
        }
        return start.map { Capture(range: $0..<text.endIndex, phase: .unknown) }
    }

    static func isPotentialStartedToolCall(in text: String) -> Bool {
        guard !text.isEmpty else { return false }
        if nativeStartMarker.hasPrefix(text) || text.hasPrefix(nativeStartMarker) {
            return true
        }
        if let complete = completedToolCall(in: text),
           complete.range.lowerBound == text.startIndex {
            return true
        }
        if text.first == "{" {
            if jsonToolObjectStartMarkers.contains(where: { $0.hasPrefix(text) || text.hasPrefix($0) }) {
                return true
            }
            return isPotentialToolJSONObjectPrefix(text)
        }
        return false
    }

    static func visibleRawPrefix(in text: String) -> String {
        guard !text.isEmpty else { return text }
        var safeEnd = text.endIndex

        if let complete = completedToolCall(in: text) {
            safeEnd = minIndex(safeEnd, complete.range.lowerBound, in: text)
        }

        if let unclosedNativeStart = unclosedNativeToolCallStart(in: text) {
            safeEnd = minIndex(safeEnd, unclosedNativeStart, in: text)
        }

        if let possibleJSONStart = possibleIncompleteToolJSONStart(in: text) {
            safeEnd = minIndex(safeEnd, possibleJSONStart, in: text)
        }

        if let suffixStart = possibleToolMarkerSuffixStart(in: text) {
            safeEnd = minIndex(safeEnd, suffixStart, in: text)
        }

        return String(text[..<safeEnd])
    }

    private static func unclosedNativeToolCallStart(in text: String) -> String.Index? {
        guard let lastStart = text.range(of: nativeStartMarker, options: .backwards) else {
            return nil
        }
        if text.range(of: nativeEndMarker, range: lastStart.upperBound..<text.endIndex) == nil {
            return lastStart.lowerBound
        }
        return nil
    }

    private static func possibleIncompleteToolJSONStart(in text: String) -> String.Index? {
        var index = text.startIndex
        var earliest: String.Index?
        while index < text.endIndex {
            guard let brace = text[index..<text.endIndex].firstIndex(of: "{") else {
                break
            }
            let range = brace..<text.endIndex
            if LlamaEngine.balancedJSONValueRange(in: text, searchRange: range) == nil,
               isPotentialToolJSONObjectPrefix(String(text[range])) {
                earliest = earliest.map { minIndex($0, brace, in: text) } ?? brace
            }
            index = text.index(after: brace)
        }
        return earliest
    }

    private static func isPotentialToolJSONObjectPrefix(_ text: String) -> Bool {
        guard text.first == "{" else { return false }
        let afterBrace = text.dropFirst().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !afterBrace.isEmpty else { return false }
        return jsonToolObjectKeys.contains { key in
            key.hasPrefix(afterBrace) || afterBrace.hasPrefix(key)
        }
    }

    private static func possibleToolMarkerSuffixStart(in text: String) -> String.Index? {
        let markers = [nativeStartMarker] + jsonToolObjectStartMarkers
        for marker in markers {
            let maxCount = min(marker.count - 1, text.count)
            guard maxCount > 0 else { continue }
            for count in stride(from: maxCount, through: 1, by: -1) {
                let start = text.index(text.endIndex, offsetBy: -count)
                let suffix = String(text[start...])
                if marker.hasPrefix(suffix) {
                    return start
                }
            }
        }
        return nil
    }

    private static var jsonToolObjectStartMarkers: [String] {
        jsonToolObjectKeys.flatMap { key in
            ["{\(key)", "{ \(key)"]
        }
    }

    private static func minIndex(
        _ lhs: String.Index?,
        _ rhs: String.Index,
        in text: String
    ) -> String.Index {
        guard let lhs else { return rhs }
        return minIndex(lhs, rhs, in: text)
    }

    private static func minIndex(
        _ lhs: String.Index,
        _ rhs: String.Index,
        in text: String
    ) -> String.Index {
        lhs <= rhs ? lhs : rhs
    }
}
