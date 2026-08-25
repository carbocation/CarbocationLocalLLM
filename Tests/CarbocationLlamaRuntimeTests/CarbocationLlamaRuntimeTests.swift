import AVFoundation
import CarbocationLocalLLM
import XCTest
@testable import CarbocationLlamaRuntime

final class CarbocationLlamaRuntimeTests: XCTestCase {
    func testRuntimeTargetImportsAndLinksLlamaSymbols() {
        let summary = LlamaRuntimeSmoke.defaultModelParameterSummary()
        XCTAssertTrue(summary.contains("load_mode="))
        XCTAssertTrue(summary.contains("n_gpu_layers="))
    }

    func testRuntimeCanReadDefaultContextParams() {
        XCTAssertGreaterThan(LlamaRuntimeSmoke.defaultContextBatchSize(), 0)
    }

    func testDefaultConfigurationDisablesAccelerationAndUsesSingleTokenDraftWidth() {
        let configuration = LlamaEngineConfiguration()

        XCTAssertEqual(configuration.accelerationPolicy, .disabled)
        XCTAssertEqual(configuration.mtpMaxDraftTokens, 1)
        XCTAssertNil(configuration.microBatchSizeLimit)
    }

    func testRuntimeImportsAndLinksMTMDSymbols() {
        XCTAssertEqual(LlamaRuntimeSmoke.defaultMediaMarker(), "<__media__>")
        XCTAssertEqual(LlamaRuntimeSmoke.audioBitmapBridgeSummary(), "is_audio=true;samples=1")
        XCTAssertEqual(LlamaRuntimeSmoke.imageBufferBitmapBridgeSummary(), "is_audio=false;width=1")
    }

    func testLoadedModelInfoReportsMultimodalCapabilities() {
        let textOnly = LlamaLoadedModelInfo(
            modelID: nil,
            modelPath: "/tmp/model.gguf",
            displayName: nil,
            filename: "model.gguf",
            contextSize: 4_096,
            trainingContextSize: 4_096,
            hasEmbeddedChatTemplate: true
        )
        XCTAssertEqual(textOnly.supportedInputModalities, [.text])
        XCTAssertFalse(textOnly.supportsVision)
        XCTAssertFalse(textOnly.supportsAudio)

        let vision = LlamaLoadedModelInfo(
            modelID: nil,
            modelPath: "/tmp/model.gguf",
            displayName: nil,
            filename: "model.gguf",
            contextSize: 4_096,
            trainingContextSize: 4_096,
            hasEmbeddedChatTemplate: true,
            supportedInputModalities: [.text, .image]
        )
        XCTAssertTrue(vision.supportsVision)
        XCTAssertFalse(vision.supportsAudio)

        let audio = LlamaLoadedModelInfo(
            modelID: nil,
            modelPath: "/tmp/model.gguf",
            displayName: nil,
            filename: "model.gguf",
            contextSize: 4_096,
            trainingContextSize: 4_096,
            hasEmbeddedChatTemplate: true,
            supportedInputModalities: [.text, .audio]
        )
        XCTAssertFalse(audio.supportsVision)
        XCTAssertTrue(audio.supportsAudio)
    }

    func testMultimodalPromptRenderingUsesOrderedMediaMarkersAndKeepsPayloadsSeparate() async throws {
        let engine = LlamaEngine()
        let rendered = try await engine.publicChatTemplateMessages(
            from: [
                LLMChatMessage(role: .user, content: [
                    .text("before "),
                    .image(.rgb8(width: 1, height: 1, data: Data([1, 2, 3]))),
                    .text(" between "),
                    .audio(.pcmFloat32Mono(sampleRate: 16_000, data: Data([0, 0, 0, 0]))),
                    .text(" after")
                ])
            ],
            mediaMarker: "<__media__>"
        )

        XCTAssertEqual(rendered.messages.count, 1)
        XCTAssertEqual(
            rendered.messages[0].content.stringValue,
            "before <__media__> between <__media__> after"
        )
        XCTAssertEqual(rendered.media.count, 2)
        XCTAssertEqual(rendered.images.count, 1)
        XCTAssertEqual(rendered.images[0].location, LLMContentLocation(messageIndex: 0, partIndex: 1))
        XCTAssertEqual(rendered.images[0].image.data, Data([1, 2, 3]))
        XCTAssertEqual(rendered.audio.count, 1)
        XCTAssertEqual(rendered.audio[0].location, LLMContentLocation(messageIndex: 0, partIndex: 3))
    }

    func testPromptRenderingForwardsAssistantReasoningSeparately() async throws {
        let engine = LlamaEngine()
        let rendered = try await engine.publicChatTemplateMessages(
            from: [
                LLMChatMessage(
                    role: .assistant,
                    text: "Visible answer",
                    reasoningContent: "Historical reasoning"
                )
            ],
            mediaMarker: nil
        )

        XCTAssertEqual(rendered.messages.first?.content, .string("Visible answer"))
        XCTAssertEqual(rendered.messages.first?.reasoningContent, "Historical reasoning")
    }

    func testVisionPromptRenderingRejectsAssistantImagesWithLocation() async throws {
        let engine = LlamaEngine()

        do {
            _ = try await engine.publicChatTemplateMessages(
                from: [
                    LLMChatMessage(role: .assistant, content: [
                        .image(.rgb8(width: 1, height: 1, data: Data([0, 0, 0])))
                    ])
                ],
                mediaMarker: "<__media__>"
            )
            XCTFail("Expected assistant image placement rejection.")
        } catch let error as LLMEngineError {
            guard case .unsupportedImagePlacement(let location) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(location, LLMContentLocation(messageIndex: 0, partIndex: 0))
        }
    }

    func testMultimodalPromptRenderingRejectsAssistantAudioWithLocation() async throws {
        let engine = LlamaEngine()

        do {
            _ = try await engine.publicChatTemplateMessages(
                from: [
                    LLMChatMessage(role: .assistant, content: [
                        .audio(.pcmFloat32Mono(sampleRate: 16_000, data: Data([0, 0, 0, 0])))
                    ])
                ],
                mediaMarker: "<__media__>"
            )
            XCTFail("Expected assistant audio placement rejection.")
        } catch let error as LLMEngineError {
            guard case .unsupportedAudioPlacement(let location) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(location, LLMContentLocation(messageIndex: 0, partIndex: 0))
        }
    }

    func testPCMFloat32MonoValidationRejectsBadAudio() throws {
        let valid = try LLMAudioInput
            .pcmFloat32Mono(sampleRate: 16_000, data: Data([0, 0, 0, 0]))
            .validatedPCMFloat32Mono(expectedSampleRate: 16_000)
        XCTAssertEqual(valid, [0])

        do {
            _ = try LLMAudioInput
                .pcmFloat32Mono(sampleRate: 8_000, data: Data([0, 0, 0, 0]))
                .validatedPCMFloat32Mono(
                    expectedSampleRate: 16_000,
                    location: LLMContentLocation(messageIndex: 2, partIndex: 1)
                )
            XCTFail("Expected sample rate mismatch.")
        } catch let error as LLMEngineError {
            guard case .audioSampleRateMismatch(let mismatch, let location) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(mismatch.expected, 16_000)
            XCTAssertEqual(mismatch.actual, 8_000)
            XCTAssertEqual(location, LLMContentLocation(messageIndex: 2, partIndex: 1))
        }

        do {
            _ = try LLMAudioInput
                .pcmFloat32Mono(sampleRate: 16_000, data: Data([0, 0, 0]))
                .validatedPCMFloat32Mono(expectedSampleRate: 16_000)
            XCTFail("Expected invalid byte count.")
        } catch let error as LLMEngineError {
            guard case .invalidAudioData = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testAppleAudioDecoderNormalizesM4A() throws {
        let data = try makeAACEncodedAudioData(fileExtension: "m4a", duration: 0.25)
        XCTAssertEqual(LLMAudioInput.sniffEncodedFormat(data), .m4a)
        let samples = try LlamaAppleAudioDecoder.decodeFloat32Mono(
            data: data,
            format: .m4a,
            sampleRate: 16_000,
            location: LLMContentLocation(messageIndex: 0, partIndex: 0)
        )

        XCTAssertGreaterThan(samples.count, 1_000)
        XCTAssertLessThan(samples.count, 8_000)
        XCTAssertTrue(samples.allSatisfy { $0.isFinite && $0 >= -1 && $0 <= 1 })
    }

    func testAppleAudioDecoderNormalizesAAC() throws {
        let data = try makeAACEncodedAudioData(fileExtension: "aac", duration: 0.25)
        XCTAssertEqual(LLMAudioInput.sniffEncodedFormat(data), .aac)
        let samples = try LlamaAppleAudioDecoder.decodeFloat32Mono(
            data: data,
            format: .aac,
            sampleRate: 16_000,
            location: LLMContentLocation(messageIndex: 0, partIndex: 1)
        )

        XCTAssertGreaterThan(samples.count, 1_000)
        XCTAssertLessThan(samples.count, 8_000)
        XCTAssertTrue(samples.allSatisfy { $0.isFinite && $0 >= -1 && $0 <= 1 })
    }

    func testAppleAudioDecoderEnforcesDurationAfterDecode() throws {
        let data = try makeAACEncodedAudioData(
            fileExtension: "m4a",
            duration: LLMAudioInput.maximumDuration + 0.25
        )

        do {
            _ = try LlamaAppleAudioDecoder.decodeFloat32Mono(
                data: data,
                format: .m4a,
                sampleRate: 16_000,
                location: LLMContentLocation(messageIndex: 3, partIndex: 2)
            )
            XCTFail("Expected decoded audio duration rejection.")
        } catch let error as LLMEngineError {
            guard case .audioDurationExceeded(let limit, let location) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(limit.maxSeconds, LLMAudioInput.maximumDuration)
            XCTAssertGreaterThan(limit.actualSeconds, LLMAudioInput.maximumDuration)
            XCTAssertEqual(location, LLMContentLocation(messageIndex: 3, partIndex: 2))
        }
    }

    func testVisionTokenAccountingUsesContextPositionCostWhenHigher() {
        XCTAssertEqual(LlamaVisionTokenAccounting.contextCost(tokenCount: 12, positionCount: 9), 12)
        XCTAssertEqual(LlamaVisionTokenAccounting.contextCost(tokenCount: 12, positionCount: 20), 20)
    }

    func testMTPAccelerationPolicyAllowsGrammarAndControlObject() {
        let opaqueMTPContext = UnsafeMutableRawPointer(bitPattern: 0x1)
        XCTAssertTrue(LlamaEngine.shouldUseMTPAcceleration(
            policy: .automatic,
            mtpContext: opaqueMTPContext,
            grammarMode: .none,
            control: nil,
            hasIncompatibleControlPath: false
        ))
        XCTAssertTrue(LlamaEngine.shouldUseMTPAcceleration(
            policy: .automatic,
            mtpContext: opaqueMTPContext,
            grammarMode: .eager(grammar: "root ::= object"),
            control: nil,
            hasIncompatibleControlPath: false
        ))
        XCTAssertTrue(LlamaEngine.shouldUseMTPAcceleration(
            policy: .automatic,
            mtpContext: opaqueMTPContext,
            grammarMode: .lazy(grammar: "root ::= object", triggerPatterns: ["{"]),
            control: LLMGenerationControl(),
            hasIncompatibleControlPath: false
        ))
        XCTAssertEqual(LlamaEngine.mtpAccelerationStatus(
            policy: .automatic,
            supportsMTPAcceleration: true,
            mtpContext: opaqueMTPContext,
            hasIncompatibleControlPath: false
        ), .active)
    }

    func testMTPAccelerationPolicyReportsDisabledStates() {
        let opaqueMTPContext = UnsafeMutableRawPointer(bitPattern: 0x1)
        XCTAssertEqual(LlamaEngine.mtpAccelerationStatus(
            policy: .automatic,
            supportsMTPAcceleration: false,
            mtpContext: nil,
            hasIncompatibleControlPath: false
        ), .unsupported)
        XCTAssertEqual(LlamaEngine.mtpAccelerationStatus(
            policy: .disabled,
            supportsMTPAcceleration: true,
            mtpContext: opaqueMTPContext,
            hasIncompatibleControlPath: false
        ), .disabledByPolicy)
        XCTAssertEqual(LlamaEngine.mtpAccelerationStatus(
            policy: .automatic,
            supportsMTPAcceleration: true,
            mtpContext: opaqueMTPContext,
            hasIncompatibleControlPath: true
        ), .disabledByIncompatibleControlPath)
        XCTAssertEqual(LlamaEngine.mtpAccelerationStatus(
            policy: .automatic,
            supportsMTPAcceleration: true,
            mtpContext: nil,
            hasIncompatibleControlPath: false
        ), .runtimeUnavailable)
        XCTAssertFalse(LlamaEngine.shouldUseMTPAcceleration(
            policy: .automatic,
            mtpContext: nil,
            grammarMode: .none,
            control: nil,
            hasIncompatibleControlPath: false
        ))
        XCTAssertFalse(LlamaEngine.shouldUseMTPAcceleration(
            policy: .disabled,
            mtpContext: opaqueMTPContext,
            grammarMode: .none,
            control: nil,
            hasIncompatibleControlPath: false
        ))
    }

    func testMTPDraftWidthUsesRequestedValueWithRangeClamp() {
        XCTAssertEqual(
            LlamaEngine.effectiveMTPMaxDraftTokens(requested: 3),
            3
        )
        XCTAssertEqual(
            LlamaEngine.effectiveMTPMaxDraftTokens(requested: 0),
            1
        )
        XCTAssertEqual(
            LlamaEngine.effectiveMTPMaxDraftTokens(requested: 40),
            32
        )
    }

    func testDisabledMTPDiagnosticDoesNotEvaluateMessage() {
        var evaluationCount = 0
        let disabledMessage = LlamaEngine.mtpDiagnosticMessage(
            enabled: false,
            makeMessage: {
                evaluationCount += 1
                return "disabled"
            }
        )

        XCTAssertNil(disabledMessage)
        XCTAssertEqual(evaluationCount, 0)

        let enabledMessage = LlamaEngine.mtpDiagnosticMessage(
            enabled: true,
            makeMessage: {
                evaluationCount += 1
                return "enabled"
            }
        )

        XCTAssertEqual(enabledMessage, "enabled")
        XCTAssertEqual(evaluationCount, 1)
    }

    func testResolvedSamplerDiagnosticsReportsGreedyRuntimeSettings() {
        let options = LLMSamplingDefaults.extractionSafe.applying(to: GenerationOptions())
        let diagnostics = LlamaEngine.resolvedSamplerDiagnostics(options: options)

        XCTAssertEqual(
            diagnostics.requestLine,
            "temperature=0 topK=40 topP=0.9 minP=nil presencePenalty=nil repetitionPenalty=nil seed=nil"
        )
        XCTAssertEqual(
            diagnostics.resolvedLine,
            "chain=penalties,greedy temperature=0 topK=disabled topP=disabled minP=disabled penaltyLastN=16 repetitionPenalty=1 frequencyPenalty=0 presencePenalty=0 seed=none"
        )
    }

    func testResolvedSamplerDiagnosticsReportsSampledRuntimeSettings() {
        let options = GenerationOptions(
            temperature: 0.7,
            topP: 0.9,
            topK: 40,
            minP: 0.05,
            presencePenalty: 0.1,
            repetitionPenalty: 1.1,
            seed: 123
        )
        let diagnostics = LlamaEngine.resolvedSamplerDiagnostics(
            options: options,
            grammarSampler: "eager",
            usesReasoningBudgetSampler: true
        )

        XCTAssertEqual(
            diagnostics.resolvedLine,
            "chain=reasoning-budget,penalties,grammar:eager,top-k,top-p,min-p,temperature,distribution temperature=0.7 topK=40 topP=0.9 minP=0.05 penaltyLastN=16 repetitionPenalty=1.1 frequencyPenalty=0 presencePenalty=0.1 seed=123"
        )
    }

    func testGenerateWithoutLoadedModelReportsNoModelLoaded() async throws {
        let engine = LlamaEngine()

        do {
            _ = try await engine.generate(
                system: "System",
                prompt: "User",
                options: GenerationOptions(),
                onPhaseAwareEvent: { _ in }
            )
            XCTFail("Expected unloaded generation to fail.")
        } catch let error as LLMEngineError {
            guard case .noModelLoaded = error else {
                return XCTFail("Expected noModelLoaded, got \(error)")
            }
        }
    }

    func testGenerateWithToolsWithoutLoadedModelReportsNoModelLoaded() async throws {
        let engine = LlamaEngine()
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "lookup", description: "Lookup test data.")
        ) { _ in
            ["ok": true]
        }

        do {
            _ = try await engine.generateWithTools(LLMToolGenerationRequest(
                prompt: "Use lookup.",
                tools: [tool]
            ))
            XCTFail("Expected unloaded tool generation to fail.")
        } catch let error as LLMEngineError {
            guard case .noModelLoaded = error else {
                return XCTFail("Expected noModelLoaded, got \(error)")
            }
        }
    }

    func testUnphasedToolGenerationPreservesIncompleteSegmentStopReason() async throws {
        let engine = LlamaEngine()
        try await engine.loadVocabularyOnlyForTesting(
            modelAt: Self.gemma4VocabModelURL,
            displayName: "Tool Stop Reason Vocab",
            requestedContext: 1_024
        )
        await engine.setToolAwareGenerationSegmentOverrideForTesting { _ in
            LlamaEngine.ToolAwareGenerationSegmentOverrideOutput(
                finalText: "Partial response",
                reasoningContent: nil,
                toolCalls: [],
                stopReason: "max-tokens",
                triggerPhase: .final,
                remainingThinkingBudgetTokens: nil,
                generatedTokens: 8
            )
        }
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "lookup", description: "Lookup test data.")
        ) { _ in
            ["ok": true]
        }

        let result = try await engine.generateWithTools(LLMToolGenerationRequest(
            prompt: "Use lookup.",
            tools: [tool]
        ))

        XCTAssertEqual(result.finalText, "Partial response")
        XCTAssertEqual(result.stopReason, "max-tokens")
        XCTAssertTrue(result.toolCalls.isEmpty)
        XCTAssertTrue(result.toolOutputs.isEmpty)
        XCTAssertEqual(result.roundsCompleted, 0)
    }

    func testNativeToolHistoryKeepsAssistantReasoningSeparate() {
        let messages = LlamaEngine.nativeToolMessages(
            system: "System",
            originalPrompt: "Question",
            history: [
                LLMToolInteractionRound(
                    calls: [
                        LLMToolCall(
                            executionID: "call_1",
                            name: "lookup",
                            arguments: ["query": "value"]
                        )
                    ],
                    outputs: [
                        LLMToolOutput(
                            callID: "call_1",
                            name: "lookup",
                            content: ["answer": "result"]
                        )
                    ],
                    reasoningContent: "Reasoning before the tool call"
                )
            ],
            finalUserInstruction: nil
        )

        let assistant = messages.first(where: { $0.role == "assistant" })
        XCTAssertEqual(assistant?.content, .string(""))
        XCTAssertEqual(assistant?.reasoningContent, "Reasoning before the tool call")
    }

    func testStructuredToolRequestPreservesPriorAssistantReasoningInTemplate() async throws {
        let engine = LlamaEngine()
        try await engine.loadVocabularyOnlyForTesting(
            modelAt: Self.gemma4VocabModelURL,
            displayName: "Structured Tool History Vocab",
            requestedContext: 2_048
        )
        await engine.setChatTemplateForTesting("""
        {%- for message in messages -%}
        <{{ message.role }}>{{ message.content }}|{{ message.reasoning_content|default('none') }}|
        {%- endfor -%}
        <assistant>
        """)

        let segmentScript = ToolGenerationSegmentScript()
        await engine.setToolAwareGenerationSegmentOverrideForTesting { input in
            await segmentScript.nextSegment(for: input)
        }
        let lookupTool = LLMTool(
            definition: LLMToolDefinition(name: "lookup", description: "Lookup test data.")
        ) { _ in
            ["answer": "tool output"]
        }

        let result = try await engine.generateWithTools(LLMToolGenerationRequest(
            messages: [
                LLMChatMessage(role: .system, text: "System policy"),
                LLMChatMessage(role: .user, text: "First question"),
                LLMChatMessage(
                    role: .assistant,
                    text: "Prior answer",
                    reasoningContent: "Prior private reasoning"
                ),
                LLMChatMessage(role: .user, text: "Follow-up question")
            ],
            options: GenerationOptions(
                enableThinking: true,
                reasoningEffort: .medium,
                preserveThinking: true
            ),
            tools: [lookupTool],
            maxToolRounds: 1
        ))

        XCTAssertEqual(result.finalText, "Final answer after tool.")
        let snapshot = await segmentScript.snapshot()
        XCTAssertEqual(snapshot.count, 2)
        let firstPrompt = try XCTUnwrap(snapshot.prompts.first)
        XCTAssertTrue(firstPrompt.contains("<user>First question|none|"))
        XCTAssertTrue(firstPrompt.contains("<assistant>Prior answer|Prior private reasoning|"))
        XCTAssertTrue(firstPrompt.contains("<user>Follow-up question|none|"))
        XCTAssertLessThan(
            try XCTUnwrap(firstPrompt.range(of: "First question")?.lowerBound),
            try XCTUnwrap(firstPrompt.range(of: "Prior answer")?.lowerBound)
        )
        XCTAssertLessThan(
            try XCTUnwrap(firstPrompt.range(of: "Prior answer")?.lowerBound),
            try XCTUnwrap(firstPrompt.range(of: "Follow-up question")?.lowerBound)
        )
    }

    func testUnloadDuringSuspendedToolGenerationDefersUntilContinuation() async throws {
        let engine = LlamaEngine()
        try await engine.loadVocabularyOnlyForTesting(
            modelAt: Self.gemma4VocabModelURL,
            displayName: "Gemma 4 Vocab",
            requestedContext: 1_024
        )

        let segmentScript = ToolGenerationSegmentScript()
        await engine.setToolAwareGenerationSegmentOverrideForTesting { input in
            await segmentScript.nextSegment(for: input)
        }

        let toolGate = SuspendedToolGate()
        let lookupTool = LLMTool(
            definition: LLMToolDefinition(
                name: "lookup",
                description: "Lookup test data.",
                parameters: [
                    "type": "object",
                    "properties": [
                        "query": ["type": "string"]
                    ],
                    "required": ["query"]
                ]
            )
        ) { _ in
            await toolGate.suspendUntilReleased()
            return ["answer": "tool output"]
        }
        let control = LLMGenerationControl()
        let request = LLMToolGenerationRequest(
            system: "System",
            prompt: "Use lookup.",
            tools: [lookupTool],
            maxToolRounds: 1
        )

        let generationTask = Task {
            try await engine.generateWithTools(request, control: control)
        }

        try await withTimeout(seconds: 5) {
            await toolGate.waitUntilStarted()
        }
        XCTAssertTrue(control.requestThinkingTermination())
        XCTAssertEqual(control.thinkingTerminationRequestCount, 1)

        await engine.unload()
        let loadedInfoAfterUnloadRequest = await engine.currentLoadedModelInfo()
        XCTAssertNotNil(
            loadedInfoAfterUnloadRequest,
            "unload() should be deferred while a tool generation is active."
        )
        let lifecycleStateAfterUnloadRequest = await engine.modelLifecycleState
        XCTAssertEqual(lifecycleStateAfterUnloadRequest, .unloadPending)

        do {
            _ = try await engine.generateWithTools(request)
            XCTFail("A pending unload must reject a new public tool-generation request.")
        } catch let error as LLMEngineError {
            guard case .noModelLoaded = error else {
                return XCTFail("Unexpected public tool-generation error: \(error)")
            }
        }

        do {
            try await engine.acquireGenerationLease()
            XCTFail("A pending unload must reject new generation work.")
        } catch let error as LLMEngineError {
            guard case .noModelLoaded = error else {
                return XCTFail("Unexpected generation lease error: \(error)")
            }
        }

        do {
            try await engine.requireModelLoadAvailable()
            XCTFail("A pending unload must reject even a same-model load before reuse is considered.")
        } catch let error as LLMEngineError {
            guard case .modelLoadFailed = error else {
                return XCTFail("Unexpected model load error: \(error)")
            }
        }

        await toolGate.release()
        let result = try await withTimeout(seconds: 5) {
            try await generationTask.value
        }

        XCTAssertEqual(result.finalText, "Final answer after tool.")
        XCTAssertEqual(result.stopReason, "eog")
        XCTAssertEqual(result.roundsCompleted, 1)
        XCTAssertEqual(result.toolCalls.map(\.name), ["lookup"])
        XCTAssertEqual(result.toolOutputs.map(\.content), [["answer": "tool output"]])
        XCTAssertEqual(control.thinkingTerminationRequestCount, 0)
        let loadedInfoAfterGeneration = await engine.currentLoadedModelInfo()
        XCTAssertNil(
            loadedInfoAfterGeneration,
            "Deferred unload should run after the active generation completes."
        )
        let lifecycleStateAfterGeneration = await engine.modelLifecycleState
        XCTAssertEqual(lifecycleStateAfterGeneration, .unloaded)

        let snapshot = await segmentScript.snapshot()
        XCTAssertEqual(snapshot.count, 2)
        XCTAssertFalse(snapshot.prompts[0].contains("<|tool_response>"))
        XCTAssertTrue(snapshot.prompts[0].contains("<|tool>"))
        XCTAssertTrue(snapshot.prompts[0].contains("declaration:lookup"))
        XCTAssertTrue(snapshot.prompts[1].contains(#"<|tool_call>call:lookup{query:<|"|>sp500<|"|>}<tool_call|>"#))
        XCTAssertTrue(snapshot.prompts[1].contains("tool output"))
        XCTAssertFalse(snapshot.prompts[1].contains("Use the tool outputs above to continue the answer."))
        XCTAssertEqual(snapshot.templateModes, [.embedded, .embedded])
        XCTAssertEqual(snapshot.isInternalContinuationFlags, [false, true])
    }

    func testManagedLifecycleWatermarkRejectsAnOlderUnload() async throws {
        let engine = LlamaEngine()
        try await engine.loadVocabularyOnlyForTesting(
            modelAt: Self.gemma4VocabModelURL,
            displayName: "Managed lifecycle vocabulary",
            requestedContext: 1_024
        )

        let advanced = await engine.supersedeManagedLifecycleOperations(epoch: 2)
        let staleUnloadAccepted = await engine.unload(managedLifecycleEpoch: 1)
        let loadedInfo = await engine.currentLoadedModelInfo()
        let lifecycleState = await engine.modelLifecycleState

        XCTAssertTrue(advanced)
        XCTAssertFalse(staleUnloadAccepted)
        XCTAssertNotNil(loadedInfo)
        XCTAssertEqual(lifecycleState, .ready)
    }

    func testManagedGenerationAdmissionRejectsAReplacedRuntimeIdentity() async throws {
        let engine = LlamaEngine()
        try await engine.loadVocabularyOnlyForTesting(
            modelAt: Self.gemma4VocabModelURL,
            displayName: "First managed runtime",
            requestedContext: 1_024
        )
        let firstSnapshot = await engine.currentManagedRuntimeSnapshot()
        let first = try XCTUnwrap(firstSnapshot)

        await engine.unload()
        try await engine.loadVocabularyOnlyForTesting(
            modelAt: Self.gemma4VocabModelURL,
            displayName: "Replacement managed runtime",
            requestedContext: 1_024
        )
        let replacementSnapshot = await engine.currentManagedRuntimeSnapshot()
        let replacement = try XCTUnwrap(replacementSnapshot)
        XCTAssertNotEqual(first.identity, replacement.identity)

        do {
            try await LlamaManagedRuntimeContext.$expectedIdentity.withValue(first.identity) {
                try await engine.acquireGenerationLease()
                await engine.endGenerationLease()
            }
            XCTFail("A request routed to the replaced runtime must not be admitted.")
        } catch let error as LlamaManagedRuntimeError {
            guard case .lifecycleOperationSuperseded = error else {
                return XCTFail("Unexpected managed runtime error: \(error)")
            }
        }

        try await LlamaManagedRuntimeContext.$expectedIdentity.withValue(replacement.identity) {
            try await engine.acquireGenerationLease()
            await engine.endGenerationLease()
        }
    }

    func testPrefillRangesSplitByBatchSize() {
        let ranges = LlamaEngine.prefillRanges(tokenCount: 10, maxBatchSize: 4)
            .map { [$0.lowerBound, $0.upperBound] }

        XCTAssertEqual(ranges, [[0, 4], [4, 8], [8, 10]])
        XCTAssertEqual(LlamaEngine.prefillRanges(tokenCount: 0, maxBatchSize: 4).count, 0)
        XCTAssertEqual(LlamaEngine.prefillRanges(tokenCount: 10, maxBatchSize: 0).count, 0)
    }

    func testPromptPrefillChunkPlansRequestLogitsOnlyForFinalChunk() {
        let chunks = LlamaEngine.promptPrefillChunkPlans(tokenCount: 10, maxBatchSize: 4)

        XCTAssertEqual(
            chunks.map { [$0.range.lowerBound, $0.range.upperBound] },
            [[0, 4], [4, 8], [8, 10]]
        )
        XCTAssertEqual(chunks.map(\.requestsLogits), [false, false, true])
        XCTAssertEqual(
            LlamaEngine.promptPrefillChunkPlans(tokenCount: 4, maxBatchSize: 4)
                .map(\.requestsLogits),
            [true]
        )
        XCTAssertTrue(
            LlamaEngine.promptPrefillChunkPlans(tokenCount: 0, maxBatchSize: 4).isEmpty
        )
        XCTAssertEqual(
            LlamaEngine.promptPrefillChunkPlans(
                tokenCount: 6,
                maxBatchSize: 4,
                requestsFinalLogits: false
            ).map(\.requestsLogits),
            [false, false]
        )
    }

    func testPromptCheckpointAnchorUsesLastUserOccurrenceAndTemplateTrim() {
        let rendered = "system repeated value\nuser: repeated value\nassistant:"
        XCTAssertEqual(
            LlamaEngine.promptCheckpointAnchorText(
                renderedPrompt: rendered,
                userContent: "  repeated value  "
            ),
            "system repeated value\nuser: repeated value"
        )
    }

    func testPromptCheckpointBacksOffFromUserBoundaryForExtendedTemplateReuse() {
        let oldPrompt: [Int32] = [1, 2, 3, 4, 5, 6, 20, 21, 22]
        let userBoundary: [Int32] = [1, 2, 3, 4, 5, 6]
        let extendedPrompt: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22]

        let checkpoint = LlamaEngine.promptCheckpointTokenCount(
            promptTokens: oldPrompt,
            anchorTokens: userBoundary,
            boundaryBackoff: 2
        )

        XCTAssertEqual(checkpoint, 4)
        XCTAssertEqual(
            LlamaEngine.commonTokenPrefixCount(
                Array(oldPrompt.prefix(checkpoint ?? 0)),
                extendedPrompt
            ),
            checkpoint
        )
    }

    func testPromptCheckpointRetainsNearbyBoundaryForBranchReuse() {
        XCTAssertTrue(LlamaEngine.shouldRetainPromptCheckpoint(
            savedTokenCount: 399,
            requestedTokenCount: 413
        ))
        XCTAssertTrue(LlamaEngine.shouldRetainPromptCheckpoint(
            savedTokenCount: 413,
            requestedTokenCount: 399
        ))
        XCTAssertFalse(LlamaEngine.shouldRetainPromptCheckpoint(
            savedTokenCount: 399,
            requestedTokenCount: 464
        ))
    }

    func testPromptCheckpointReusesNearbySavedPrefixWithoutReanchoring() {
        let checkpoint = Array(Int32(0)..<399)
        let exact = checkpoint + Array(repeating: Int32(900), count: 13)
        let extended = checkpoint + Array(repeating: Int32(901), count: 27)

        XCTAssertEqual(
            LlamaEngine.reusablePromptCheckpointTokenCount(
                checkpointTokens: checkpoint,
                newPromptTokens: exact
            ),
            399
        )
        XCTAssertEqual(
            LlamaEngine.reusablePromptCheckpointTokenCount(
                checkpointTokens: checkpoint,
                newPromptTokens: extended
            ),
            399
        )

        var divergent = exact
        divergent[200] = -1
        XCTAssertNil(LlamaEngine.reusablePromptCheckpointTokenCount(
            checkpointTokens: checkpoint,
            newPromptTokens: divergent
        ))
        XCTAssertNil(LlamaEngine.reusablePromptCheckpointTokenCount(
            checkpointTokens: checkpoint,
            newPromptTokens: checkpoint + Array(repeating: 1, count: 65)
        ))
        XCTAssertNil(LlamaEngine.reusablePromptCheckpointTokenCount(
            checkpointTokens: checkpoint,
            newPromptTokens: checkpoint
        ))
    }

    func testStableStreamingPrefixHoldsIncompleteControlLiterals() {
        let literals = ["STOP", "</think>", "<|return|>"]

        for literal in literals {
            for split in 1..<literal.count {
                let partial = String(literal.prefix(split))
                XCTAssertEqual(
                    LlamaEngine.stableStreamingPrefix(
                        "visible\(partial)",
                        protecting: literals
                    ),
                    "visible",
                    "Expected split \(split) of \(literal) to remain buffered"
                )
            }
            XCTAssertEqual(
                LlamaEngine.stableStreamingPrefix(
                    "visible\(literal)",
                    protecting: literals
                ),
                "visible\(literal)"
            )
        }

        XCTAssertEqual(
            LlamaEngine.stableStreamingPrefix("visible ordinary", protecting: literals),
            "visible ordinary"
        )
        XCTAssertFalse(LlamaEngine.hasVisibleStreamingContent(" \n\t"))
        XCTAssertTrue(LlamaEngine.hasVisibleStreamingContent(" \nanswer"))
    }

    func testPromptCheckpointRestoreTrimFailureRequiresFullReset() {
        XCTAssertEqual(
            LlamaEngine.promptCheckpointRestoreDisposition(
                bridgeResult: 399,
                expectedTokenCount: 399
            ),
            .restored
        )
        XCTAssertEqual(
            LlamaEngine.promptCheckpointRestoreDisposition(
                bridgeResult: -1,
                expectedTokenCount: 399
            ),
            .unavailable
        )
        XCTAssertEqual(
            LlamaEngine.promptCheckpointRestoreDisposition(
                bridgeResult: -2,
                expectedTokenCount: 399
            ),
            .requiresFullReset
        )
    }

    func testPromptCacheCommitPlanIncludesOnlyDecodedResidentTokens() {
        let sampledButNotDecodedTerminal = Int32(99)
        let synchronized = LlamaEngine.promptCacheCommitPlan(
            promptTokens: [1, 2],
            decodedGeneratedTokens: [3, 4],
            mtpSynchronized: true
        )

        XCTAssertEqual(synchronized.cachedTokens, [1, 2, 3, 4])
        XCTAssertEqual(synchronized.mtpCachedTokens, synchronized.cachedTokens)
        XCTAssertFalse(synchronized.cachedTokens.contains(sampledButNotDecodedTerminal))

        let targetOnly = LlamaEngine.promptCacheCommitPlan(
            promptTokens: [1, 2],
            decodedGeneratedTokens: [3, 4],
            mtpSynchronized: false
        )
        XCTAssertEqual(targetOnly.cachedTokens, synchronized.cachedTokens)
        XCTAssertNil(targetOnly.mtpCachedTokens)
    }

    func testDecodedGeneratedTokensExtendNextPromptReusablePrefix() {
        let committed = LlamaEngine.promptCacheCommitPlan(
            promptTokens: [1, 2],
            decodedGeneratedTokens: [3, 4],
            mtpSynchronized: false
        )
        let reuse = LlamaEngine.promptPrefillPlan(
            cachedPromptTokens: committed.cachedTokens,
            newPromptTokens: [1, 2, 3, 4, 5]
        )

        XCTAssertEqual(reuse.commonPrefixCount, 4)
        XCTAssertEqual(reuse.retainedPrefixCount, 4)
        XCTAssertNil(reuse.removeStartPosition)
        XCTAssertEqual(reuse.decodeStartIndex, 4)
        XCTAssertNil(LlamaEngine.promptCheckpointRestoreTokenCount(
            plan: reuse,
            checkpointTokens: [1, 2],
            newPromptTokens: [1, 2, 3, 4, 5]
        ))
    }

    func testPromptPrefillPlanReusesCommonPrefixAndRedecodesTailForLogits() {
        let partial = LlamaEngine.promptPrefillPlan(
            cachedPromptTokens: [1, 2, 3, 4],
            newPromptTokens: [1, 2, 3, 9]
        )
        XCTAssertEqual(partial.commonPrefixCount, 3)
        XCTAssertEqual(partial.retainedPrefixCount, 3)
        XCTAssertFalse(partial.shouldClearMemory)
        XCTAssertEqual(partial.removeStartPosition, 3)
        XCTAssertEqual(partial.decodeStartIndex, 3)

        let exact = LlamaEngine.promptPrefillPlan(
            cachedPromptTokens: [1, 2, 3],
            newPromptTokens: [1, 2, 3]
        )
        XCTAssertEqual(exact.commonPrefixCount, 3)
        XCTAssertEqual(exact.retainedPrefixCount, 2)
        XCTAssertFalse(exact.shouldClearMemory)
        XCTAssertEqual(exact.removeStartPosition, 2)
        XCTAssertEqual(exact.decodeStartIndex, 2)
        XCTAssertEqual(
            LlamaEngine.promptCheckpointRestoreTokenCount(
                plan: exact,
                checkpointTokens: [1, 2],
                newPromptTokens: [1, 2, 3]
            ),
            2
        )
    }

    func testPromptPrefillPlanFallsBackToFullPrefillWhenNoUsablePrefixCanBeKept() {
        let noPrefix = LlamaEngine.promptPrefillPlan(
            cachedPromptTokens: [1, 2, 3],
            newPromptTokens: [9, 2, 3]
        )
        XCTAssertTrue(noPrefix.shouldClearMemory)
        XCTAssertEqual(noPrefix.decodeStartIndex, 0)
        XCTAssertNil(noPrefix.removeStartPosition)

        let oneTokenExact = LlamaEngine.promptPrefillPlan(
            cachedPromptTokens: [1],
            newPromptTokens: [1]
        )
        XCTAssertTrue(oneTokenExact.shouldClearMemory)
        XCTAssertEqual(oneTokenExact.commonPrefixCount, 1)
        XCTAssertEqual(oneTokenExact.decodeStartIndex, 0)
    }

    func testPromptRuntimeResetPlanClearsLlamaMemoryWhenMTPIsNotLoaded() {
        let plan = LlamaEngine.promptRuntimeResetPlan(mtpContext: nil)

        XCTAssertTrue(plan.clearsPromptCaches)
        XCTAssertEqual(plan.resetPath, .llamaMemory)
    }

    func testPromptRuntimeResetPlanClearsLoadedMTPBridgeWhenPresent() {
        let loadedMTPContext = UnsafeMutableRawPointer(bitPattern: 0x1)
        let plan = LlamaEngine.promptRuntimeResetPlan(mtpContext: loadedMTPContext)

        XCTAssertTrue(plan.clearsPromptCaches)
        XCTAssertEqual(plan.resetPath, .mtpBridge)
    }

    func testContextClampRespectsTrainingLimitAndMinimum() {
        XCTAssertEqual(
            LlamaEngine.clampedContextSize(requestedContext: 32_768, trainingContext: 16_384),
            16_384
        )
        XCTAssertEqual(
            LlamaEngine.clampedContextSize(requestedContext: 128, trainingContext: 32_768),
            LlamaContextPolicy.minimumContext
        )
        XCTAssertEqual(
            LlamaEngine.clampedContextSize(requestedContext: 0, trainingContext: 0),
            LlamaContextPolicy.unknownTrainingFallback
        )
    }

    func testContextBatchCandidatesFallBackByHalving() {
        XCTAssertEqual(
            LlamaEngine.contextBatchCandidates(contextSize: 4_096, batchSizeLimit: 64),
            [64, 32, 16, 8, 4, 2, 1]
        )
        XCTAssertEqual(
            LlamaEngine.contextBatchCandidates(contextSize: 32, batchSizeLimit: 512),
            [32, 16, 8, 4, 2, 1]
        )
        XCTAssertEqual(
            LlamaEngine.contextBatchCandidates(contextSize: 4_096, batchSizeLimit: 2_048),
            [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        )
        XCTAssertEqual(
            LlamaEngine.contextBatchCandidates(
                contextSize: 4_096,
                batchSizeLimit: 2_048,
                microBatchSizeLimit: 2_048
            ),
            [2_048, 1_024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        )
        XCTAssertEqual(
            LlamaEngine.contextBatchCandidates(
                contextSize: 768,
                batchSizeLimit: 2_048,
                microBatchSizeLimit: 1_024
            ),
            [768, 384, 192, 96, 48, 24, 12, 6, 3, 1]
        )
    }

    func testGPUOffloadTreatsNegativeLayerCountAsAllLayers() {
        XCTAssertFalse(LlamaEngine.usesGPUOffload(gpuLayerCount: 0))
        XCTAssertTrue(LlamaEngine.usesGPUOffload(gpuLayerCount: -1))
        XCTAssertTrue(LlamaEngine.usesGPUOffload(gpuLayerCount: 999))
    }

    func testCalibrationMemoryGuardrailAllowsPlausibleCandidate() {
        let gib = UInt64(1_073_741_824)
        let profile = LlamaContextMemoryGuardrail.ModelProfile(
            modelTensorBytes: 4 * gib,
            layerCount: 32,
            embeddingCount: 4_096,
            headCount: 32,
            kvHeadCount: 8,
            keyCacheBytesPerElement: 2,
            valueCacheBytesPerElement: 2
        )

        let estimate = LlamaContextMemoryGuardrail.estimate(
            profile: profile,
            contextSize: 8_192,
            batchSize: 64,
            budgetBytes: 16 * gib
        )

        XCTAssertTrue(LlamaContextMemoryGuardrail.allowsProbe(estimate))
        XCTAssertLessThan(estimate.totalBytes, estimate.budgetBytes)
        XCTAssertGreaterThan(estimate.kvCacheBytes, 0)
        XCTAssertEqual(estimate.modelReserveBytes, profile.modelTensorBytes)
    }

    func testCalibrationMemoryGuardrailDoesNotRejectOnlyBecauseLoadedModelExceedsBudget() {
        let gib = UInt64(1_073_741_824)
        let profile = LlamaContextMemoryGuardrail.ModelProfile(
            modelTensorBytes: 28 * gib,
            layerCount: 32,
            embeddingCount: 4_096,
            headCount: 32,
            kvHeadCount: 8,
            keyCacheBytesPerElement: 2,
            valueCacheBytesPerElement: 2
        )

        let estimate = LlamaContextMemoryGuardrail.estimate(
            profile: profile,
            contextSize: 8_192,
            batchSize: 64,
            budgetBytes: 8 * gib
        )

        XCTAssertTrue(LlamaContextMemoryGuardrail.allowsProbe(estimate))
        XCTAssertGreaterThan(estimate.totalBytes, estimate.budgetBytes)
        XCTAssertLessThan(estimate.modelReserveBytes, estimate.modelTensorBytes)
        XCTAssertLessThanOrEqual(estimate.requiredBytes, estimate.budgetBytes)
    }

    func testCalibrationMemoryGuardrailCountsBoundedLoadedModelReserve() {
        let gib = UInt64(1_073_741_824)
        let profile = LlamaContextMemoryGuardrail.ModelProfile(
            modelTensorBytes: 28 * gib,
            layerCount: 32,
            embeddingCount: 4_096,
            headCount: 32,
            kvHeadCount: 8,
            keyCacheBytesPerElement: 2,
            valueCacheBytesPerElement: 2
        )
        let budgetBytes = 8 * gib

        let estimate = LlamaContextMemoryGuardrail.estimate(
            profile: profile,
            contextSize: 65_536,
            batchSize: 2_048,
            budgetBytes: budgetBytes
        )

        XCTAssertEqual(
            estimate.modelReserveBytes,
            UInt64((Double(budgetBytes) * LlamaContextMemoryGuardrail.maximumModelReserveBudgetFraction).rounded(.up))
        )
        XCTAssertEqual(estimate.requiredBytes, estimate.modelReserveBytes + estimate.incrementalBytes)
        XCTAssertFalse(LlamaContextMemoryGuardrail.allowsProbe(estimate))
    }

    func testCalibrationMemoryGuardrailUsesStricterIncrementalSafetyMargin() {
        let profile = LlamaContextMemoryGuardrail.ModelProfile(
            modelTensorBytes: 4 * UInt64(1_073_741_824),
            layerCount: 1,
            embeddingCount: 1_024,
            headCount: 1,
            kvHeadCount: 1,
            keyCacheBytesPerElement: 2,
            valueCacheBytesPerElement: 2
        )
        let baseline = LlamaContextMemoryGuardrail.estimate(
            profile: profile,
            contextSize: 2_000_000,
            batchSize: 1,
            budgetBytes: 0
        )
        let basis = baseline.kvCacheBytes + baseline.decodeWorkspaceBytes
        let tooLenientBudget = basis + UInt64((Double(basis) * 0.15).rounded(.up))
        let estimate = LlamaContextMemoryGuardrail.estimate(
            profile: profile,
            contextSize: 2_000_000,
            batchSize: 1,
            budgetBytes: tooLenientBudget
        )

        XCTAssertFalse(LlamaContextMemoryGuardrail.allowsProbe(estimate))
        XCTAssertEqual(
            estimate.safetyMarginBytes,
            UInt64((Double(basis) * LlamaContextMemoryGuardrail.incrementalSafetyMarginRatio).rounded(.up))
        )
        XCTAssertGreaterThan(estimate.requiredBytes, estimate.budgetBytes)
    }

    func testCalibrationMemoryGuardrailRejectsClearlyOversizedCandidate() {
        let gib = UInt64(1_073_741_824)
        let profile = LlamaContextMemoryGuardrail.ModelProfile(
            modelTensorBytes: 28 * gib,
            layerCount: 48,
            embeddingCount: 8_192,
            headCount: 64,
            kvHeadCount: 8,
            keyCacheBytesPerElement: 2,
            valueCacheBytesPerElement: 2
        )

        let estimate = LlamaContextMemoryGuardrail.estimate(
            profile: profile,
            contextSize: 262_144,
            batchSize: 2_048,
            budgetBytes: 16 * gib
        )

        XCTAssertFalse(LlamaContextMemoryGuardrail.allowsProbe(estimate))
        XCTAssertGreaterThan(estimate.totalBytes, estimate.budgetBytes)
        XCTAssertGreaterThan(estimate.requiredBytes, estimate.budgetBytes)
    }

    func testContextParamsUseDeploymentSafeMicroBatchDefault() {
        let params = LlamaEngine.contextParams(
            contextSize: 8_192,
            batchSize: 2_048,
            threads: 2
        )

        XCTAssertEqual(params.n_ctx, 8_192)
        XCTAssertEqual(params.n_batch, 2_048)
        XCTAssertEqual(params.n_ubatch, 512)
        XCTAssertEqual(params.n_threads, 2)
        XCTAssertEqual(params.n_threads_batch, 2)
        XCTAssertEqual(params.n_rs_seq, 0)

        let mtpParams = LlamaEngine.contextParams(
            contextSize: 8_192,
            batchSize: 2_048,
            microBatchSize: 256,
            threads: 2,
            recurrentStateSnapshots: 3
        )
        XCTAssertEqual(mtpParams.n_batch, 2_048)
        XCTAssertEqual(mtpParams.n_ubatch, 256)
        XCTAssertEqual(mtpParams.n_rs_seq, 3)
    }

    func testContextCalibrationSearchCancellationDoesNotWriteCacheRecord() async throws {
        let suiteName = "CarbocationLlamaRuntimeCalibrationCancellationTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = LlamaContextCalibrationStore(defaults: defaults)

        do {
            _ = try await LlamaContextCalibrationAlgorithm.search(candidates: [512, 1_024]) { _ in
                throw CancellationError()
            }
            XCTFail("Expected cancellation to propagate.")
        } catch is CancellationError {
            XCTAssertTrue(store.records().isEmpty)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testGenerationBudgetMath() {
        XCTAssertEqual(
            LlamaEngine.maxGenerationTokens(contextSize: 4_096, promptTokenCount: 3_000, reserve: 1_024),
            72
        )
        XCTAssertEqual(
            LlamaEngine.maxGenerationTokens(contextSize: 4_096, promptTokenCount: 4_000, reserve: 1_024),
            0
        )
    }

    func testStopSequenceTrimmingUsesEarliestMatch() {
        XCTAssertEqual(
            LlamaEngine.trimmingAtFirstStopSequence("alpha STOP beta END", stopSequences: ["END", "STOP"]),
            "alpha "
        )
        XCTAssertNil(LlamaEngine.trimmingAtFirstStopSequence("alpha beta", stopSequences: ["STOP"]))
        XCTAssertNil(LlamaEngine.trimmingAtFirstStopSequence("alpha beta", stopSequences: [""]))
    }

    func testMergingStopSequencesPreservesCallerOrderAndDeduplicates() {
        XCTAssertEqual(
            LlamaEngine.mergingStopSequences(["END", "STOP"], ["STOP", "<turn|>", ""]),
            ["END", "STOP", "<turn|>"]
        )
    }

    func testCommonChatAppliesGemma4Template() throws {
        let template = try String(contentsOf: Self.gemma4TemplateURL, encoding: .utf8)
        let prompt = try LlamaChatTemplateRenderer.format(
            template: template,
            system: "System",
            user: "User",
            bosToken: "<bos>",
            eosToken: "<eos>"
        )

        XCTAssertTrue(prompt.contains("<|turn>system\n"))
        XCTAssertTrue(prompt.contains("System"))
        XCTAssertTrue(prompt.contains("<turn|>\n"))
        XCTAssertTrue(prompt.contains("<|turn>user\n"))
        XCTAssertTrue(prompt.contains("User"))
        XCTAssertTrue(prompt.contains("<|turn>model\n"))
        XCTAssertTrue(prompt.hasSuffix("<|turn>model\n<|channel>thought\n<channel|>"))
        XCTAssertFalse(prompt.contains("<start_of_turn>"))
        XCTAssertFalse(prompt.contains("<end_of_turn>"))
    }

    func testCommonChatGemma4ThinkingEnabledStartsImplicitThoughtChannel() throws {
        let template = try String(contentsOf: Self.gemma4TemplateURL, encoding: .utf8)
        let prompt = try LlamaChatTemplateRenderer.format(
            template: template,
            system: "System",
            user: "User",
            bosToken: "<bos>",
            eosToken: "<eos>",
            enableThinking: true
        )
        let profile = OutputSanitizationProfile.derived(fromChatTemplate: template)

        XCTAssertTrue(prompt.contains("<|turn>system\n<|think|>\n"))
        XCTAssertTrue(prompt.hasSuffix("<|turn>model\n"))
        XCTAssertEqual(
            LlamaEngine.continuingOpenThinkingPairs(in: prompt, profile: profile),
            [OutputDelimiterPair(open: "", close: "<channel|>")]
        )
    }

    func testCommonChatGemma4RendersNativeTools() throws {
        let template = try String(contentsOf: Self.gemma4TemplateURL, encoding: .utf8)
        let formatter = try LlamaChatTemplateRenderer(template: template)
        let tool = LLMToolDefinition(
            name: "bing_search",
            description: "Search the web.",
            parameters: [
                "type": "object",
                "properties": [
                    "queries": [
                        "type": "array",
                        "items": ["type": "string"]
                    ]
                ],
                "required": ["queries"]
            ]
        )

        let prompt = try formatter.format(
            messages: [
                ChatTemplateMessage(role: "system", content: "System"),
                ChatTemplateMessage(role: "user", content: "Is Ted Turner still alive?")
            ],
            tools: [tool],
            bosToken: "<bos>",
            eosToken: "<eos>"
        )

        XCTAssertTrue(prompt.contains("<|tool>"))
        XCTAssertTrue(prompt.contains("declaration:bing_search"))
        XCTAssertTrue(prompt.contains("description:<|\"|>Search the web.<|\"|>"))
        XCTAssertTrue(prompt.contains("<tool|>"))
    }

    func testCommonChatGemma4RendersNativeToolHistory() throws {
        let template = try String(contentsOf: Self.gemma4TemplateURL, encoding: .utf8)
        let formatter = try LlamaChatTemplateRenderer(template: template)
        let call = LLMToolCall(
            executionID: "call_1",
            rawID: "call_1",
            name: "bing_search",
            arguments: ["queries": ["is Ted Turner still alive"]]
        )

        let prompt = try formatter.format(
            messages: [
                ChatTemplateMessage(role: "user", content: "Is Ted Turner still alive?"),
                ChatTemplateMessage(role: "assistant", content: "", toolCalls: [call]),
                ChatTemplateMessage(
                    role: "tool",
                    content: ["ok": true, "answer": "Ted Turner is alive."],
                    toolCallID: "call_1",
                    name: "bing_search"
                )
            ],
            bosToken: "<bos>",
            eosToken: "<eos>"
        )

        XCTAssertTrue(prompt.contains(#"<|tool_call>call:bing_search{queries:[<|"|>is Ted Turner still alive<|"|>]}<tool_call|>"#))
        XCTAssertTrue(prompt.contains(#"<|tool_response>response:bing_search{"#))
        XCTAssertTrue(prompt.contains("Ted Turner is alive."))
        XCTAssertTrue(prompt.contains("<tool_response|>"))
    }

    func testCommonChatQwenThinkingEnabledLeavesThinkingOpen() throws {
        let template = try String(contentsOf: Self.qwen35TemplateURL, encoding: .utf8)
        let prompt = try LlamaChatTemplateRenderer.format(
            template: template,
            system: "System",
            user: "User",
            bosToken: "<bos>",
            eosToken: "<eos>",
            enableThinking: true
        )
        let profile = OutputSanitizationProfile.derived(fromChatTemplate: template)

        XCTAssertTrue(prompt.hasSuffix("<|im_start|>assistant\n<think>\n"))
        XCTAssertEqual(
            LlamaEngine.continuingOpenThinkingPairs(in: prompt, profile: profile),
            [OutputDelimiterPair(open: "<think>", close: "</think>")]
        )
    }

    func testCommonChatQwenThinkingDisabledUsesClosedEmptyThinkingBlock() throws {
        let template = try String(contentsOf: Self.qwen35TemplateURL, encoding: .utf8)
        let prompt = try LlamaChatTemplateRenderer.format(
            template: template,
            system: "System",
            user: "User",
            bosToken: "<bos>",
            eosToken: "<eos>",
            enableThinking: false
        )
        let profile = OutputSanitizationProfile.derived(fromChatTemplate: template)

        XCTAssertTrue(prompt.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        XCTAssertEqual(LlamaEngine.continuingOpenThinkingPairs(in: prompt, profile: profile), [])
    }

    func testCommonChatForwardsReasoningControlsAndAssistantReasoningHistory() throws {
        let template = """
        {{ reasoning_effort|default('template-default') }}|{{ preserve_thinking|default(true) }}|
        {%- for message in messages -%}
        {{ message.role }}={{ message.content }}:{{ message.reasoning_content|default('none') }};
        {%- endfor -%}
        {{- add_generation_prompt -}}
        """
        let formatter = try LlamaChatTemplateRenderer(template: template)
        let prompt = try formatter.format(
            messages: [
                ChatTemplateMessage(role: "user", content: "Question"),
                ChatTemplateMessage(
                    role: "assistant",
                    content: "Visible answer",
                    reasoningContent: "Historical reasoning"
                )
            ],
            bosToken: "<bos>",
            eosToken: "<eos>",
            enableThinking: true,
            reasoningEffort: .medium,
            preserveThinking: false
        )

        XCTAssertTrue(prompt.lowercased().contains("medium|false|"))
        XCTAssertTrue(prompt.contains("assistant=Visible answer:Historical reasoning;"))
    }

    func testCommonChatLeavesReasoningControlsUndefinedWhenNotRequested() throws {
        let template = "{{ reasoning_effort|default('xhigh') }}|{{ preserve_thinking|default(true) }}|{% for message in messages %}{{ message.content }}{% endfor %}"
        let prompt = try LlamaChatTemplateRenderer.format(
            template: template,
            system: "",
            user: "Question",
            bosToken: "<bos>",
            eosToken: "<eos>",
            enableThinking: true
        )

        XCTAssertTrue(prompt.lowercased().hasPrefix("xhigh|true|"))
        XCTAssertTrue(prompt.contains("Question"))
    }

    func testReasoningBudgetPlanRequiresEnabledThinkingAndDelimiters() {
        let profile = OutputSanitizationProfile(thinkingPairs: [
            OutputDelimiterPair(open: "<think>", close: "</think>")
        ])

        XCTAssertNil(LlamaEngine.reasoningBudgetPlan(
            for: GenerationOptions(thinkingBudgetTokens: 8),
            profile: profile,
            continuingOpenThinkingPairs: []
        ))

        XCTAssertNil(LlamaEngine.reasoningBudgetPlan(
            for: GenerationOptions(enableThinking: true, thinkingBudgetTokens: 8),
            profile: .empty,
            continuingOpenThinkingPairs: []
        ))

        let plan = LlamaEngine.reasoningBudgetPlan(
            for: GenerationOptions(enableThinking: true, thinkingBudgetTokens: 8),
            profile: profile,
            continuingOpenThinkingPairs: []
        )
        XCTAssertEqual(plan?.pair, OutputDelimiterPair(open: "<think>", close: "</think>"))
        XCTAssertEqual(plan?.budgetTokens, 8)
        XCTAssertEqual(plan?.initialState, .idle)
    }

    func testReasoningBudgetPlanStartsCountingForContinuingThinkingPrompt() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")

        let plan = LlamaEngine.reasoningBudgetPlan(
            for: GenerationOptions(
                enableThinking: true,
                thinkingBudgetTokens: 0,
                thinkingBudgetMessage: "Budget reached."
            ),
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: [pair]
        )

        XCTAssertEqual(plan?.pair, pair)
        XCTAssertEqual(plan?.budgetTokens, 0)
        XCTAssertEqual(plan?.message, "Budget reached.")
        XCTAssertEqual(plan?.initialState, .counting)
    }

    func testReasoningBudgetPlanStartsCountingForExplicitPromptThinkingPhase() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")

        let plan = LlamaEngine.reasoningBudgetPlan(
            for: GenerationOptions(enableThinking: true, thinkingBudgetTokens: 4),
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: [],
            startsInThinking: true
        )

        XCTAssertEqual(plan?.pair, pair)
        XCTAssertEqual(plan?.initialState, .counting)
    }

    func testReasoningBudgetPlanStartsCountingForImplicitPromptThinkingPhase() {
        let pair = OutputDelimiterPair(open: "", close: "<channel|>")

        let plan = LlamaEngine.reasoningBudgetPlan(
            for: GenerationOptions(enableThinking: true),
            profile: OutputSanitizationProfile(thinkingPairs: [
                OutputDelimiterPair(open: "<|channel>thought", close: "<channel|>")
            ]),
            continuingOpenThinkingPairs: [pair],
            requiresSampler: true
        )

        XCTAssertEqual(plan?.pair, pair)
        XCTAssertEqual(plan?.budgetTokens, Int(Int32.max))
        XCTAssertEqual(plan?.initialState, .counting)
    }

    func testReasoningBudgetPlanCreatesControlSamplerWithoutFixedBudget() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")

        XCTAssertNil(LlamaEngine.reasoningBudgetPlan(
            for: GenerationOptions(enableThinking: true),
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: []
        ))

        let plan = LlamaEngine.reasoningBudgetPlan(
            for: GenerationOptions(enableThinking: true),
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: [],
            requiresSampler: true
        )

        XCTAssertEqual(plan?.pair, pair)
        XCTAssertEqual(plan?.budgetTokens, Int(Int32.max))
        XCTAssertEqual(plan?.initialState, .idle)
    }

    func testReasoningBudgetPlanUsesFinalMarkerForControlSampler() {
        let plan = LlamaEngine.reasoningBudgetPlan(
            for: GenerationOptions(enableThinking: true),
            profile: OutputSanitizationProfile(finalMarkers: ["<final>"]),
            continuingOpenThinkingPairs: [],
            requiresSampler: true
        )

        XCTAssertEqual(plan?.pair, OutputDelimiterPair(open: "", close: "<final>"))
        XCTAssertEqual(plan?.budgetTokens, Int(Int32.max))
        XCTAssertEqual(plan?.initialState, .counting)
    }

    func testContinuingOpenThinkingPairsIgnoresLiteralUserThinkingTagBeforeAssistantPrompt() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let profile = OutputSanitizationProfile(thinkingPairs: [pair])
        let renderedPrompt = """
        <|im_start|>user
        What does the literal <think> tag mean?<|im_end|>
        <|im_start|>assistant
        """

        XCTAssertEqual(LlamaEngine.continuingOpenThinkingPairs(in: renderedPrompt, profile: profile), [])
    }

    func testGenerationGrammarModeUsesLazyForThinkingStructuredOutput() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let profile = OutputSanitizationProfile(thinkingPairs: [pair])
        let options = GenerationOptions(grammar: "root ::= object", enableThinking: true)

        let mode = LlamaEngine.generationGrammarMode(
            for: options,
            profile: profile,
            continuingOpenThinkingPairs: [pair]
        )

        guard case .lazy(let grammar, let triggerPatterns) = mode else {
            return XCTFail("Expected lazy grammar mode.")
        }
        XCTAssertEqual(grammar, "root ::= object")
        XCTAssertTrue(triggerPatterns.contains(#"</think>\s*(\{|\[)"#))
        XCTAssertTrue(triggerPatterns.contains(#"^\s*(\{|\[)"#))
    }

    func testGenerationGrammarModeKeepsEagerGrammarWhenThinkingDisabled() {
        let profile = OutputSanitizationProfile(
            thinkingPairs: [OutputDelimiterPair(open: "<think>", close: "</think>")]
        )
        let options = GenerationOptions(grammar: "root ::= object", enableThinking: false)

        let mode = LlamaEngine.generationGrammarMode(
            for: options,
            profile: profile,
            continuingOpenThinkingPairs: []
        )

        XCTAssertEqual(mode, .eager(grammar: "root ::= object"))
    }

    func testStreamContentPhaseDetectsGeneratedThinkingBlock() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )

        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "", plan: plan), .unknown)
        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "<thi", plan: plan), .unknown)
        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "<think>draft", plan: plan), .thinking)
        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "<think>draft</think>", plan: plan), .final)
        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "final answer", plan: plan), .final)
    }

    func testStreamContentPhaseStartsThinkingForPromptPrefilledThinking() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: [pair],
            startsInThinking: nil
        )

        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "", plan: plan), .thinking)
        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "draft notes", plan: plan), .thinking)
        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "draft notes</think>", plan: plan), .final)
    }

    func testStreamContentPhaseSupportsExplicitPromptDrivenThinking() {
        let pair = OutputDelimiterPair(open: "<|channel>thought", close: "<channel|>")
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: [],
            startsInThinking: true
        )

        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "", plan: plan), .thinking)
        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "private notes", plan: plan), .thinking)
        XCTAssertEqual(
            LlamaEngine.streamContentPhase(in: "private notes<channel|>final answer", plan: plan),
            .final
        )
    }

    func testStreamContentPhaseSupportsImplicitPromptDrivenThinking() {
        let profilePair = OutputDelimiterPair(open: "<|channel>thought", close: "<channel|>")
        let implicitPair = OutputDelimiterPair(open: "", close: "<channel|>")
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(thinkingPairs: [profilePair]),
            continuingOpenThinkingPairs: [implicitPair],
            startsInThinking: nil
        )

        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "", plan: plan), .thinking)
        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "private notes", plan: plan), .thinking)
        XCTAssertEqual(
            LlamaEngine.streamContentPhase(in: "private notes<channel|>final answer", plan: plan),
            .final
        )
    }

    func testStreamContentPhaseDetectsAdditionalThinkingDialects() {
        let pairs = [
            OutputDelimiterPair(open: "[THINK]", close: "[/THINK]"),
            OutputDelimiterPair(open: "<seed:think>", close: "</seed:think>"),
            OutputDelimiterPair(open: "<|inner_prefix|>", close: "<|inner_suffix|>"),
            OutputDelimiterPair(open: "<|think|>", close: "<|end|>")
        ]

        for pair in pairs {
            let plan = LlamaEngine.StreamPhasePlan(
                profile: OutputSanitizationProfile(thinkingPairs: [pair]),
                continuingOpenThinkingPairs: [],
                startsInThinking: nil
            )

            XCTAssertEqual(LlamaEngine.streamContentPhase(in: pair.open + "private", plan: plan), .thinking)
            XCTAssertEqual(
                LlamaEngine.streamContentPhase(
                    in: pair.open + "private" + pair.close + "final answer",
                    plan: plan
                ),
                .final
            )
        }
    }

    func testStreamContentPhaseUsesFinalResponseMarkerForUndelimitedReasoning() {
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(finalMarkers: ["[BEGIN FINAL RESPONSE]"]),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )

        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "private notes", plan: plan), .thinking)
        XCTAssertEqual(
            LlamaEngine.streamContentPhase(
                in: "private notes[BEGIN FINAL RESPONSE]visible answer",
                plan: plan
            ),
            .final
        )
    }

    func testStreamContentPhaseKeepsSolarContentMarkerAsFinalBoundary() {
        let pair = OutputDelimiterPair(open: "<|think|>", close: "<|end|>")
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(
                thinkingPairs: [pair],
                scrubTokens: ["<|end|>", "<|content|>"],
                finalMarkers: ["<|content|>"]
            ),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )

        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "<|think|>private", plan: plan), .thinking)
        XCTAssertEqual(
            LlamaEngine.streamContentPhase(in: "<|think|>private<|end|>", plan: plan),
            .unknown
        )
        XCTAssertEqual(
            LlamaEngine.streamContentPhase(
                in: "<|think|>private<|end|><|content|>visible answer",
                plan: plan
            ),
            .final
        )
    }

    func testStreamContentPhaseUsesFinalMarkersForAnalysisChannels() {
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(finalMarkers: ["<|channel|>final<|message|>"]),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )

        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "", plan: plan), .unknown)
        XCTAssertEqual(LlamaEngine.streamContentPhase(in: "analysis", plan: plan), .thinking)
        XCTAssertEqual(
            LlamaEngine.streamContentPhase(
                in: "analysis<|channel|>final<|message|>answer",
                plan: plan
            ),
            .final
        )
    }

    func testStreamContentPhaseDoesNotTreatLiteralFinalTextAsThinking() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )

        XCTAssertEqual(
            LlamaEngine.streamContentPhase(
                in: "The literal <think> tag can be discussed safely.",
                plan: plan
            ),
            .final
        )
    }

    func testSanitizedGeneratedTextCanDriveFinalAnswerStreamingAfterThinking() throws {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let profile = OutputSanitizationProfile(
            thinkingPairs: [pair],
            finalMarkers: ["<final>"]
        )

        XCTAssertEqual(
            try LlamaEngine.sanitizedGeneratedText(
                "<think>private</think><final>visible",
                profile: profile,
                continuingOpenThinkingPairs: [],
                requiresNonEmptyStructuredOutput: false
            ),
            "visible"
        )

        XCTAssertEqual(
            try LlamaEngine.sanitizedGeneratedText(
                "private</think> visible",
                profile: profile,
                continuingOpenThinkingPairs: [pair],
                requiresNonEmptyStructuredOutput: false
            ),
            "visible"
        )
    }

    func testPhasedGeneratedTextStripsThinkingDelimiters() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )

        let phased = LlamaEngine.phasedGeneratedText(
            "<think>private notes</think>visible answer",
            plan: plan
        )

        XCTAssertEqual(phased.thinkingText, "private notes")
        XCTAssertEqual(phased.finalText, "visible answer")
        XCTAssertEqual(phased.phaseSegments, [
            LLMGenerationPhaseSegment(phase: .thinking, text: "private notes"),
            LLMGenerationPhaseSegment(phase: .final, text: "visible answer")
        ])
    }

    func testPhasedGeneratedTextHandlesPromptOpenThinking() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: [pair],
            startsInThinking: nil
        )

        let phased = LlamaEngine.phasedGeneratedText(
            "continued notes</think>visible answer",
            plan: plan
        )

        XCTAssertEqual(phased.thinkingText, "continued notes")
        XCTAssertEqual(phased.finalText, "visible answer")
    }

    func testPhasedGeneratedTextHandlesFinalMarkerOnlyReasoning() {
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(
                scrubTokens: ["[END FINAL RESPONSE]"],
                finalMarkers: ["[BEGIN FINAL RESPONSE]"]
            ),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )

        let phased = LlamaEngine.phasedGeneratedText(
            "private notes[BEGIN FINAL RESPONSE]visible answer[END FINAL RESPONSE]",
            plan: plan
        )

        XCTAssertEqual(phased.thinkingText, "private notes")
        XCTAssertEqual(phased.finalText, "visible answer")
    }

    func testPhasedGeneratedTextHoldsBackPartialFinalMarkerPrefix() {
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(
                finalMarkers: ["<|channel|>final<|message|>"]
            ),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )

        let phased = LlamaEngine.phasedGeneratedText(
            "private notes<|channel|>fin",
            plan: plan
        )

        XCTAssertEqual(phased.thinkingText, "private notes")
        XCTAssertEqual(phased.finalText, "")
        XCTAssertEqual(phased.phaseSegments, [
            LLMGenerationPhaseSegment(phase: .thinking, text: "private notes")
        ])

        let markerOnly = LlamaEngine.phasedGeneratedText(
            "<|channel|>fin",
            plan: plan
        )
        XCTAssertEqual(markerOnly.thinkingText, "")
        XCTAssertEqual(markerOnly.finalText, "")
        XCTAssertEqual(markerOnly.phaseSegments, [])
    }

    func testPhasedGeneratedTextHandlesMultipleThinkingBlocks() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )

        let phased = LlamaEngine.phasedGeneratedText(
            "<think>first</think><think>second</think>final",
            plan: plan
        )

        XCTAssertEqual(phased.thinkingText, "first\nsecond")
        XCTAssertEqual(phased.finalText, "final")
        XCTAssertEqual(phased.phaseSegments, [
            LLMGenerationPhaseSegment(phase: .thinking, text: "first"),
            LLMGenerationPhaseSegment(phase: .thinking, text: "second"),
            LLMGenerationPhaseSegment(phase: .final, text: "final")
        ])
    }

    func testPhasedGeneratedTextKeepsLiteralThinkingTagInFinalText() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )

        let phased = LlamaEngine.phasedGeneratedText(
            "The literal <think> tag can be discussed safely.",
            plan: plan
        )

        XCTAssertEqual(phased.thinkingText, "")
        XCTAssertEqual(phased.finalText, "The literal <think> tag can be discussed safely.")
    }

    func testPhasedGeneratedTextReturnsUnclosedThinking() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )

        let phased = LlamaEngine.phasedGeneratedText(
            "<think>unfinished private notes",
            plan: plan
        )

        XCTAssertEqual(phased.thinkingText, "unfinished private notes")
        XCTAssertEqual(phased.finalText, "")
        XCTAssertEqual(phased.rawGeneratedText, "<think>unfinished private notes")
    }

    func testPhasedGenerationTextReplacementPreservesMatchingSegmentBoundaries() {
        let phased = LlamaEngine.PhasedGenerationText(
            thinkingText: "private",
            finalText: "first\nsecond",
            phaseSegments: [
                LLMGenerationPhaseSegment(phase: .thinking, text: "private"),
                LLMGenerationPhaseSegment(phase: .final, text: "first"),
                LLMGenerationPhaseSegment(phase: .thinking, text: "more private"),
                LLMGenerationPhaseSegment(phase: .final, text: "second")
            ],
            rawGeneratedText: "raw"
        )

        let unchanged = phased.replacingPhaseText(.final, with: "first\nsecond")

        XCTAssertEqual(unchanged.phaseSegments, phased.phaseSegments)

        let replaced = phased.replacingPhaseText(.final, with: "visible first\nvisible second")

        XCTAssertEqual(replaced.finalText, "visible first\nvisible second")
        XCTAssertEqual(replaced.phaseSegments, [
            LLMGenerationPhaseSegment(phase: .thinking, text: "private"),
            LLMGenerationPhaseSegment(phase: .final, text: "visible first"),
            LLMGenerationPhaseSegment(phase: .thinking, text: "more private"),
            LLMGenerationPhaseSegment(phase: .final, text: "visible second")
        ])
    }

    func testToolPhasedStreamSuppressesFinalWhenVisibleTextIsNilForPartialFinalMarkerPrefix() {
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(
                finalMarkers: ["<|channel|>final<|message|>"]
            ),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )
        let recorder = ToolPhasedEventRecorder()
        let state = LlamaToolPhasedGenerationStreamState { event in
            recorder.append(event)
        }

        state.emit(
            raw: "private notes<|channel|>fin",
            plan: plan,
            finalText: nil,
            allowsParserDerivedFinalText: false,
            snapshotReason: .streamCorrection
        )

        let generationEvents = recorder.generationEvents
        XCTAssertEqual(generationEvents.contentText(for: .thinking), "private notes")
        XCTAssertEqual(generationEvents.contentText(for: .final), "")

        let result = state.result(
            stopReason: "tool-call-complete",
            promptTokens: 1,
            generatedTokens: 1,
            templateMode: .unavailable,
            accelerationStats: nil
        )
        XCTAssertEqual(result.thinkingText, "private notes")
        XCTAssertEqual(result.finalText, "")
    }

    func testToolPhasedStreamSuppressesRawToolProtocolAsFinalWhenVisibleTextIsNil() {
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )
        let recorder = ToolPhasedEventRecorder()
        let state = LlamaToolPhasedGenerationStreamState { event in
            recorder.append(event)
        }
        let rawToolProtocol = #"<|tool_call>call:lookup {"query":"private"}<tool_call|>"#

        state.emit(
            raw: rawToolProtocol,
            plan: plan,
            finalText: nil,
            allowsParserDerivedFinalText: false,
            snapshotReason: .streamCorrection
        )

        XCTAssertEqual(recorder.generationEvents.contentText(for: .thinking), "")
        XCTAssertEqual(recorder.generationEvents.contentText(for: .final), "")

        let result = state.result(
            stopReason: "tool-call-complete",
            promptTokens: 1,
            generatedTokens: 1,
            templateMode: .unavailable,
            accelerationStats: nil
        )
        XCTAssertEqual(result.thinkingText, "")
        XCTAssertEqual(result.finalText, "")
        XCTAssertEqual(result.rawGeneratedText, rawToolProtocol)
    }

    func testSanitizedGeneratedTextStripsImplicitGemma4ThinkingPrefix() throws {
        let profilePair = OutputDelimiterPair(open: "<|channel>thought", close: "<channel|>")
        let implicitPair = OutputDelimiterPair(open: "", close: "<channel|>")

        XCTAssertEqual(
            try LlamaEngine.sanitizedGeneratedText(
                "private notes<channel|>visible answer",
                profile: OutputSanitizationProfile(thinkingPairs: [profilePair]),
                continuingOpenThinkingPairs: [implicitPair],
                requiresNonEmptyStructuredOutput: false
            ),
            "visible answer"
        )
    }

    func testSanitizedGeneratedTextHandlesFinalResponseMarkerThinking() throws {
        XCTAssertEqual(
            try LlamaEngine.sanitizedGeneratedText(
                "private notes[BEGIN FINAL RESPONSE]visible answer[END FINAL RESPONSE]",
                profile: OutputSanitizationProfile(
                    scrubTokens: ["[END FINAL RESPONSE]"],
                    finalMarkers: ["[BEGIN FINAL RESPONSE]"]
                ),
                continuingOpenThinkingPairs: [],
                requiresNonEmptyStructuredOutput: false
            ),
            "visible answer"
        )
    }

    func testSanitizedGeneratedTextHandlesSolarThinkingAndContentMarkers() throws {
        let pair = OutputDelimiterPair(open: "<|think|>", close: "<|end|>")

        XCTAssertEqual(
            try LlamaEngine.sanitizedGeneratedText(
                "<|think|>private notes<|end|><|content|>visible answer<|end|>",
                profile: OutputSanitizationProfile(
                    thinkingPairs: [pair],
                    scrubTokens: ["<|end|>", "<|content|>"],
                    finalMarkers: ["<|content|>"]
                ),
                continuingOpenThinkingPairs: [],
                requiresNonEmptyStructuredOutput: false
            ),
            "visible answer"
        )
    }

    func testStructuredBoundaryWaitsForPromptOpenThinkingCloseBeforeJSON() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let profile = OutputSanitizationProfile(thinkingPairs: [pair])
        let plan = LlamaEngine.StructuredOutputPlan(
            profile: profile,
            continuingOpenThinkingPairs: [pair],
            grammarMode: .lazy(
                grammar: "root ::= object",
                triggerPatterns: LlamaEngine.lazyGrammarTriggerPatterns(
                    profile: profile,
                    continuingOpenThinkingPairs: [pair]
                )
            )
        )

        XCTAssertNil(LlamaEngine.firstStructuredGenerationBoundary(
            in: #"reasoning {"draft":true}"#,
            stopSequences: [],
            stopAtBalancedJSON: true,
            plan: plan
        ))
        XCTAssertEqual(
            LlamaEngine.structuredOutputPhase(in: #"reasoning {"draft":true}"#, plan: plan),
            .thinking
        )

        let text = #"reasoning {"draft":true}</think>{"title":"x"} trailing"#
        let boundary = LlamaEngine.firstStructuredGenerationBoundary(
            in: text,
            stopSequences: [],
            stopAtBalancedJSON: true,
            plan: plan
        )

        XCTAssertEqual(boundary?.text, #"reasoning {"draft":true}</think>{"title":"x"}"#)
        XCTAssertEqual(boundary?.reason, "json-complete")
    }

    func testStructuredBoundarySkipsGeneratedThinkingBlockBeforeJSON() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let profile = OutputSanitizationProfile(thinkingPairs: [pair])
        let plan = LlamaEngine.StructuredOutputPlan(
            profile: profile,
            continuingOpenThinkingPairs: [],
            grammarMode: .lazy(
                grammar: "root ::= object",
                triggerPatterns: LlamaEngine.lazyGrammarTriggerPatterns(
                    profile: profile,
                    continuingOpenThinkingPairs: []
                )
            )
        )
        let text = #"<think>{"draft":true}</think>{"title":"x"} trailing"#

        let boundary = LlamaEngine.firstStructuredGenerationBoundary(
            in: text,
            stopSequences: [],
            stopAtBalancedJSON: true,
            plan: plan
        )

        XCTAssertEqual(boundary?.text, #"<think>{"draft":true}</think>{"title":"x"}"#)
        XCTAssertEqual(
            try LlamaEngine.sanitizedGeneratedText(
                boundary?.text ?? "",
                profile: profile,
                continuingOpenThinkingPairs: [],
                requiresNonEmptyStructuredOutput: true
            ),
            #"{"title":"x"}"#
        )
    }

    func testStructuredBoundaryWaitsForFinalMarkerBeforeJSON() {
        let profile = OutputSanitizationProfile(
            sliceAfterMarker: "<|channel|>final<|message|>",
            scrubTokens: ["<|return|>"]
        )
        let plan = LlamaEngine.StructuredOutputPlan(
            profile: profile,
            continuingOpenThinkingPairs: [],
            grammarMode: .lazy(
                grammar: "root ::= object",
                triggerPatterns: LlamaEngine.lazyGrammarTriggerPatterns(
                    profile: profile,
                    continuingOpenThinkingPairs: []
                )
            )
        )
        let text = #"analysis {"draft":true}<|channel|>final<|message|>{"title":"x"} trailing"#

        let boundary = LlamaEngine.firstStructuredGenerationBoundary(
            in: text,
            stopSequences: [],
            stopAtBalancedJSON: true,
            plan: plan
        )

        XCTAssertEqual(
            boundary?.text,
            #"analysis {"draft":true}<|channel|>final<|message|>{"title":"x"}"#
        )
        XCTAssertEqual(boundary?.reason, "json-complete")
    }

    func testFinalAnswerProgressGateBlocksLazyStructuredPreambleAfterThinkingClose() throws {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let profile = OutputSanitizationProfile(thinkingPairs: [pair])
        let structuredPlan = LlamaEngine.StructuredOutputPlan(
            profile: profile,
            continuingOpenThinkingPairs: [],
            grammarMode: .lazy(
                grammar: "root ::= object",
                triggerPatterns: LlamaEngine.lazyGrammarTriggerPatterns(
                    profile: profile,
                    continuingOpenThinkingPairs: []
                )
            )
        )
        let streamPlan = LlamaEngine.StreamPhasePlan(
            profile: profile,
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )
        let text = "<think>draft</think>I will now return JSON."

        let structuredPhase = LlamaEngine.structuredOutputPhase(in: text, plan: structuredPlan)
        let streamPhase = LlamaEngine.streamContentPhase(in: text, plan: streamPlan)

        XCTAssertEqual(structuredPhase, .awaitingFinal)
        XCTAssertEqual(streamPhase, .final)
        XCTAssertFalse(LlamaEngine.shouldEmitFinalAnswerProgress(
            currentPhase: streamPhase,
            structuredPhase: structuredPhase
        ))
        XCTAssertEqual(
            try LlamaEngine.sanitizedGeneratedText(
                text,
                profile: profile,
                continuingOpenThinkingPairs: [],
                requiresNonEmptyStructuredOutput: false
            ),
            "I will now return JSON."
        )
    }

    func testFinalAnswerProgressGateAllowsLazyStructuredJSONAfterTrigger() throws {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let profile = OutputSanitizationProfile(thinkingPairs: [pair])
        let structuredPlan = LlamaEngine.StructuredOutputPlan(
            profile: profile,
            continuingOpenThinkingPairs: [],
            grammarMode: .lazy(
                grammar: "root ::= object",
                triggerPatterns: LlamaEngine.lazyGrammarTriggerPatterns(
                    profile: profile,
                    continuingOpenThinkingPairs: []
                )
            )
        )
        let streamPlan = LlamaEngine.StreamPhasePlan(
            profile: profile,
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )
        let text = #"<think>draft</think>{"title":"x"}"#

        let structuredPhase = LlamaEngine.structuredOutputPhase(in: text, plan: structuredPlan)
        let streamPhase = LlamaEngine.streamContentPhase(in: text, plan: streamPlan)

        XCTAssertEqual(structuredPhase, .final)
        XCTAssertEqual(streamPhase, .final)
        XCTAssertTrue(LlamaEngine.shouldEmitFinalAnswerProgress(
            currentPhase: streamPhase,
            structuredPhase: structuredPhase
        ))
        XCTAssertEqual(
            try LlamaEngine.sanitizedGeneratedText(
                text,
                profile: profile,
                continuingOpenThinkingPairs: [],
                requiresNonEmptyStructuredOutput: false
            ),
            #"{"title":"x"}"#
        )
    }

    func testGenerationBoundaryPreservesGemmaNativeToolCallEnvelope() {
        let text = #"<|tool_call>call:bing_search{queries:["is Ted Turner still alive"]}<tool_call|> trailing"#

        let boundary = LlamaEngine.firstGenerationBoundary(
            in: text,
            stopSequences: [],
            stopAtBalancedJSON: true
        )

        XCTAssertEqual(
            boundary?.text,
            #"<|tool_call>call:bing_search{queries:["is Ted Turner still alive"]}<tool_call|>"#
        )
        XCTAssertEqual(boundary?.reason, "tool-call-complete")
    }

    func testToolStreamInterpreterHidesSplitNativeMarker() {
        XCTAssertEqual(
            LlamaToolStreamInterpreter.visibleRawPrefix(in: "I will check <|tool_"),
            "I will check "
        )
    }

    func testToolStreamInterpreterCapturesNativeCallAfterProse() {
        let text = #"I will check. <|tool_call>call:lookup{query:"swift"}<tool_call|>"#
        let interception = LlamaToolStreamInterpreter.completedToolCall(in: text)

        XCTAssertEqual(interception?.calls.count, 1)
        XCTAssertEqual(interception?.calls[0].name, "lookup")
        XCTAssertEqual(interception?.calls[0].arguments.string(forKey: "query"), "swift")
        XCTAssertEqual(
            LlamaToolStreamInterpreter.visibleRawPrefix(in: text),
            "I will check. "
        )
    }

    func testToolStreamInterpreterDefersCompleteNativeCallAtStreamingTail() {
        let text = #"I will check. <|tool_call>call:lookup{query:"swift"}<tool_call|>"#

        XCTAssertNil(LlamaToolStreamInterpreter.completedToolCallForStreaming(in: text))
        XCTAssertEqual(LlamaToolStreamInterpreter.pendingNativeToolCallBatch(in: text)?.calls.count, 1)
        XCTAssertEqual(
            LlamaToolStreamInterpreter.visibleRawPrefix(in: text),
            "I will check. "
        )
    }

    func testToolStreamInterpreterDefersCompleteNativeCallBeforeSplitSibling() {
        let text = #"I will check. <|tool_call>call:lookup{query:"swift"}<tool_call|> <|tool_"#

        XCTAssertNil(LlamaToolStreamInterpreter.completedToolCallForStreaming(in: text))
        XCTAssertEqual(LlamaToolStreamInterpreter.pendingNativeToolCallBatch(in: text)?.calls.count, 1)
        XCTAssertEqual(
            LlamaToolStreamInterpreter.visibleRawPrefix(in: text),
            "I will check. "
        )
    }

    func testToolStreamInterpreterCapturesAdjacentNativeCallsWhenBoundaryIsProven() {
        let text = #"I will check. <|tool_call>call:lookup{query:"swift"}<tool_call|> <|tool_call>call:lookup{query:"swift forums"}<tool_call|> done"#
        let interception = LlamaToolStreamInterpreter.completedToolCallForStreaming(in: text)

        XCTAssertEqual(interception?.calls.count, 2)
        XCTAssertEqual(interception?.calls.map(\.name), ["lookup", "lookup"])
        XCTAssertEqual(interception?.calls[0].arguments.string(forKey: "query"), "swift")
        XCTAssertEqual(interception?.calls[1].arguments.string(forKey: "query"), "swift forums")
        XCTAssertEqual(
            LlamaToolStreamInterpreter.visibleRawPrefix(in: text),
            "I will check. "
        )
    }

    func testToolStreamInterpreterKeepsAdjacentNativeCallsPendingAtStreamingTail() {
        let text = #"I will check. <|tool_call>call:lookup{query:"swift"}<tool_call|> <|tool_call>call:lookup{query:"swift forums"}<tool_call|>"#

        XCTAssertNil(LlamaToolStreamInterpreter.completedToolCallForStreaming(in: text))
        XCTAssertEqual(LlamaToolStreamInterpreter.pendingNativeToolCallBatch(in: text)?.calls.count, 2)
    }

    func testToolStreamInterpreterCapturesThinkingPhaseForNativeCall() {
        let text = #"<think>I need current data. <|tool_call>call:lookup{query:"swift"}<tool_call|>"#
        let interception = LlamaToolStreamInterpreter.completedToolCall(
            in: text,
            phasePlan: thinkingTagPhasePlan()
        )

        XCTAssertEqual(interception?.calls.count, 1)
        XCTAssertEqual(interception?.calls[0].triggerPhase, .thinking)
    }

    func testToolStreamInterpreterCapturesFinalPhaseForNativeCallAfterThinkingClose() {
        let text = #"<think>I need current data.</think><|tool_call>call:lookup{query:"swift"}<tool_call|>"#
        let interception = LlamaToolStreamInterpreter.completedToolCall(
            in: text,
            phasePlan: thinkingTagPhasePlan()
        )

        XCTAssertEqual(interception?.calls.count, 1)
        XCTAssertEqual(interception?.calls[0].triggerPhase, .final)
    }

    func testToolStreamInterpreterCapturesSplitNativeMarkerAtOpenerStart() {
        let text = #"<think>I need current data. <|tool_"#
        let capture = LlamaToolStreamInterpreter.startedToolCall(in: text)
        let phase = capture.map {
            LlamaEngine.streamContentPhase(
                in: String(text[..<$0.range.lowerBound]),
                plan: thinkingTagPhasePlan()
            )
        }

        XCTAssertEqual(capture.map { String(text[$0.range]) }, "<|tool_")
        XCTAssertEqual(phase, .thinking)
        XCTAssertEqual(LlamaToolStreamInterpreter.visibleRawPrefix(in: text), "<think>I need current data. ")
    }

    func testToolStreamInterpreterHidesSplitJSONToolObject() {
        XCTAssertEqual(
            LlamaToolStreamInterpreter.visibleRawPrefix(in: #"I will check. {"tool_"#),
            "I will check. "
        )
    }

    func testToolStreamInterpreterCapturesJSONToolObjectAfterProse() {
        let text = #"I will check. {"tool_calls":[{"name":"lookup","arguments":{"query":"swift"}}]}"#
        let interception = LlamaToolStreamInterpreter.completedToolCall(in: text)

        XCTAssertEqual(interception?.calls.count, 1)
        XCTAssertEqual(interception?.calls[0].name, "lookup")
        XCTAssertEqual(interception?.calls[0].arguments.string(forKey: "query"), "swift")
        XCTAssertEqual(
            LlamaToolStreamInterpreter.visibleRawPrefix(in: text),
            "I will check. "
        )
    }

    func testToolStreamInterpreterCapturesSplitJSONToolObjectAtOpenerStart() {
        let text = #"<think>I need current data. {"tool_"#
        let capture = LlamaToolStreamInterpreter.startedToolCall(in: text)
        let phase = capture.map {
            LlamaEngine.streamContentPhase(
                in: String(text[..<$0.range.lowerBound]),
                plan: thinkingTagPhasePlan()
            )
        }

        XCTAssertEqual(capture.map { String(text[$0.range]) }, #"{"tool_"#)
        XCTAssertEqual(phase, .thinking)
        XCTAssertEqual(LlamaToolStreamInterpreter.visibleRawPrefix(in: text), "<think>I need current data. ")
    }

    func testToolStreamInterpreterLeavesNormalProseVisible() {
        XCTAssertEqual(
            LlamaToolStreamInterpreter.visibleRawPrefix(in: "No tool call is needed."),
            "No tool call is needed."
        )
        XCTAssertNil(LlamaToolStreamInterpreter.completedToolCall(in: "No tool call is needed."))
    }

    func testToolStreamInterpreterClearsFalseToolPrefixes() {
        XCTAssertTrue(LlamaToolStreamInterpreter.isPotentialStartedToolCall(in: "<"))
        XCTAssertFalse(LlamaToolStreamInterpreter.isPotentialStartedToolCall(in: "<not a tool"))
        XCTAssertTrue(LlamaToolStreamInterpreter.isPotentialStartedToolCall(in: "{"))
        XCTAssertFalse(LlamaToolStreamInterpreter.isPotentialStartedToolCall(in: "{not a tool"))
    }

    func testGenerationBoundaryDoesNotTreatIncompleteGemmaNativeArgumentsAsJSON() {
        let text = #"<|tool_call>call:bing_search{queries:["is Ted Turner still alive"]}"#

        let boundary = LlamaEngine.firstGenerationBoundary(
            in: text,
            stopSequences: [],
            stopAtBalancedJSON: true
        )

        XCTAssertNil(boundary)
    }

    func testStructuredSanitizerThrowsWhenPromptOpenThinkingNeverCloses() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let profile = OutputSanitizationProfile(thinkingPairs: [pair])

        XCTAssertThrowsError(try LlamaEngine.sanitizedGeneratedText(
            #"reasoning {"draft":true}"#,
            profile: profile,
            continuingOpenThinkingPairs: [pair],
            requiresNonEmptyStructuredOutput: true
        )) { error in
            guard case LLMEngineError.structuredOutputPhaseFailed = error else {
                return XCTFail("Expected structured output phase failure, got \(error).")
            }
        }
    }

    func testStructuredSanitizerReturnsJSONAfterPromptOpenThinkingCloses() throws {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let profile = OutputSanitizationProfile(thinkingPairs: [pair])

        let returned = try LlamaEngine.sanitizedGeneratedText(
            #"reasoning</think>{"title":"x"}"#,
            profile: profile,
            continuingOpenThinkingPairs: [pair],
            requiresNonEmptyStructuredOutput: true
        )

        XCTAssertEqual(returned, #"{"title":"x"}"#)
    }

    func testCommonChatGemma4DefaultThinkingPromptIsClosed() throws {
        let template = try String(contentsOf: Self.gemma4TemplateURL, encoding: .utf8)
        let prompt = try LlamaChatTemplateRenderer.format(
            template: template,
            system: "System",
            user: "User",
            bosToken: "<bos>",
            eosToken: "<eos>"
        )
        let profile = OutputSanitizationProfile.derived(fromChatTemplate: template)

        XCTAssertTrue(prompt.hasSuffix("<|turn>model\n<|channel>thought\n<channel|>"))
        XCTAssertEqual(LlamaEngine.continuingOpenThinkingPairs(in: prompt, profile: profile), [])
    }

    func testPreparedCommonChatRendererCanBeReused() throws {
        let template = try String(contentsOf: Self.gemma4TemplateURL, encoding: .utf8)
        let formatter = try LlamaChatTemplateRenderer(template: template)

        let first = try formatter.format(
            system: "System",
            user: "First",
            bosToken: "<bos>",
            eosToken: "<eos>"
        )
        let second = try formatter.format(
            system: "System",
            user: "Second",
            bosToken: "<bos>",
            eosToken: "<eos>"
        )

        XCTAssertTrue(first.contains("First"))
        XCTAssertTrue(second.contains("Second"))
        XCTAssertFalse(second.contains("First"))
    }

    func testCommonChatRejectsNonTemplateAliasBeforeLegacyFallback() {
        XCTAssertThrowsError(try LlamaChatTemplateRenderer.format(
            template: "chatml",
            system: "System",
            user: "User",
            bosToken: "",
            eosToken: ""
        ))
    }

    func testLegacyCAPIStillSupportsChatMLAlias() {
        let prompt = LlamaEngine.formatMessagesWithLegacyTemplate(
            template: "chatml",
            system: "System",
            user: "User"
        )

        XCTAssertNotNil(prompt)
        XCTAssertTrue(prompt?.contains("<|im_start|>system") == true)
        XCTAssertTrue(prompt?.contains("<|im_start|>assistant") == true)
    }

    func testCalgacusRankUsesDescendingLogitsAndTokenIDTieBreak() throws {
        let logits: [Float] = [0.5, 2.0, 2.0, -1.0]

        XCTAssertEqual(try LlamaEngine.calgacusRank(of: 1, in: logits), 1)
        XCTAssertEqual(try LlamaEngine.calgacusRank(of: 2, in: logits), 2)
        XCTAssertEqual(try LlamaEngine.calgacusRank(of: 0, in: logits), 3)
        XCTAssertEqual(try LlamaEngine.calgacusRank(of: 3, in: logits), 4)
    }

    func testCalgacusTokenAtRankUsesSameOrdering() throws {
        let logits: [Float] = [0.5, 2.0, 2.0, -1.0]

        XCTAssertEqual(try LlamaEngine.calgacusToken(atRank: 1, in: logits), 1)
        XCTAssertEqual(try LlamaEngine.calgacusToken(atRank: 2, in: logits), 2)
        XCTAssertEqual(try LlamaEngine.calgacusToken(atRank: 3, in: logits), 0)
        XCTAssertEqual(try LlamaEngine.calgacusToken(atRank: 4, in: logits), 3)
    }

    func testCalgacusRankTreatsNonFiniteLogitsAsLeastLikely() throws {
        let logits: [Float] = [.nan, 1.0, -.infinity, .infinity]

        XCTAssertEqual(try LlamaEngine.calgacusToken(atRank: 1, in: logits), 1)
        XCTAssertEqual(try LlamaEngine.calgacusToken(atRank: 2, in: logits), 0)
        XCTAssertEqual(try LlamaEngine.calgacusToken(atRank: 3, in: logits), 2)
    }

    func testCalgacusNegativeLogProbabilityUsesSoftmax() throws {
        let logits: [Float] = [0, 0]

        XCTAssertEqual(
            try LlamaEngine.calgacusNegativeLogProbability(of: 0, in: logits),
            log(2.0),
            accuracy: 0.000_001
        )
    }

    func testCalgacusTokenMetricsMatchSeparateCalculations() throws {
        let logits: [Float] = [0.5, 2.0, 2.0, -1.0]

        let metrics = try logits.withUnsafeBufferPointer {
            try LlamaEngine.calgacusTokenMetrics(of: 0, in: $0)
        }

        XCTAssertEqual(metrics.rank, try LlamaEngine.calgacusRank(of: 0, in: logits))
        XCTAssertEqual(
            metrics.negativeLogProbability,
            try LlamaEngine.calgacusNegativeLogProbability(of: 0, in: logits),
            accuracy: 0.000_001
        )
    }

    func testCalgacusTokenMetricsSkipVocabularyStatisticsByDefault() throws {
        let likelihood = LlamaTokenLikelihood(
            index: 0,
            tokenID: 1,
            tokenBytes: Data("token".utf8),
            byteRange: 0..<5,
            logProbability: -0.25,
            rank: 2
        )
        let options = LlamaTokenLikelihoodOptions()

        XCTAssertNil(options.vocabularyStatistics)
        XCTAssertNil(likelihood.vocabularyStatistics)
    }

    func testCalgacusAdvancedMetricsPreserveLikelihoodAndSummarizeVocabulary() throws {
        let logits: [Float] = [0.5, 2.0, 2.0, -1.0]
        let basic = try logits.withUnsafeBufferPointer {
            try LlamaEngine.calgacusTokenMetrics(of: 0, in: $0)
        }
        let advanced = try logits.withUnsafeBufferPointer {
            try LlamaEngine.calgacusTokenMetrics(
                of: 0,
                in: $0,
                vocabularyTemperatures: [0.5, 1, 2]
            )
        }

        XCTAssertEqual(advanced.rank, basic.rank)
        XCTAssertEqual(
            advanced.negativeLogProbability,
            basic.negativeLogProbability,
            accuracy: 0.000_000_000_001
        )
        let statistics = try XCTUnwrap(advanced.vocabularyStatistics)

        let maxLogit = Double(logits.max() ?? 0)
        let weights = logits.map { exp(Double($0) - maxLogit) }
        let normalizer = weights.reduce(0, +)
        let probabilities = weights.map { $0 / normalizer }
        let logProbabilities = probabilities.map(log)
        let expectedLogProbability = zip(probabilities, logProbabilities)
            .reduce(0) { $0 + $1.0 * $1.1 }
        let variance = zip(probabilities, logProbabilities).reduce(0) {
            $0 + $1.0 * pow($1.1 - expectedLogProbability, 2)
        }

        XCTAssertEqual(
            statistics.expectedLogProbability,
            expectedLogProbability,
            accuracy: 0.000_000_000_001
        )
        XCTAssertEqual(statistics.entropy, -expectedLogProbability, accuracy: 0.000_000_000_001)
        XCTAssertEqual(statistics.logProbabilityVariance, variance, accuracy: 0.000_000_000_001)
        XCTAssertEqual(
            statistics.logProbabilityStandardDeviation,
            sqrt(variance),
            accuracy: 0.000_000_000_001
        )
        XCTAssertEqual(statistics.temperatureNormalizations.map(\.temperature), [0.5, 1, 2])
        for temperature in [0.5, 1.0, 2.0] {
            let normalization = try XCTUnwrap(
                statistics.temperatureNormalization(at: temperature)
            )
            let expectedLogNormalizer = log(
                probabilities.reduce(0) { $0 + pow($1, 1 / temperature) }
            )
            XCTAssertEqual(
                normalization.logNormalizer,
                expectedLogNormalizer,
                accuracy: 0.000_000_000_001
            )
        }
    }

    func testCalgacusTemperatureValidationRejectsInvalidAndNormalizesOrder() throws {
        XCTAssertEqual(
            try LlamaEngine.calgacusValidatedTemperatures([1, 0.5, 1, 2]),
            [0.5, 1, 2]
        )
        for temperature in [0, -1, .infinity, .nan] {
            XCTAssertThrowsError(
                try LlamaEngine.calgacusValidatedTemperatures([temperature])
            ) { error in
                guard case LlamaVocabularyStatisticsError.invalidTemperature = error else {
                    return XCTFail("Expected invalidTemperature, got \(error)")
                }
            }
        }
    }

    func testLikelihoodBatchSizeBoundsLogitsMemoryAndHonorsOverrides() {
        XCTAssertEqual(
            LlamaEngine.calgacusLikelihoodBatchSize(
                vocabularySize: 100,
                contextBatchSize: 32,
                logitsMemoryBudget: 4_000
            ),
            10
        )
        XCTAssertEqual(
            LlamaEngine.calgacusLikelihoodBatchSize(
                vocabularySize: 100,
                contextBatchSize: 32,
                maximumBatchTokenCount: 3,
                logitsMemoryBudget: 4_000
            ),
            3
        )
        XCTAssertEqual(
            LlamaEngine.calgacusLikelihoodBatchSize(
                vocabularySize: 100,
                contextBatchSize: 0,
                maximumBatchTokenCount: 0,
                logitsMemoryBudget: 0
            ),
            1
        )
    }

    func testLlamaTokenLikelihoodProvidesDerivedValues() {
        let likelihood = LlamaTokenLikelihood(
            index: 2,
            tokenID: 42,
            tokenBytes: Data("word".utf8),
            byteRange: 5..<9,
            logProbability: log(0.25),
            rank: 3
        )

        XCTAssertEqual(likelihood.id, 2)
        XCTAssertEqual(likelihood.tokenText, "word")
        XCTAssertEqual(likelihood.negativeLogProbability, -log(0.25), accuracy: 0.000_001)
        XCTAssertEqual(likelihood.probability, 0.25, accuracy: 0.000_001)
    }

    func testLlamaTokenLikelihoodProgressProvidesFraction() {
        let progress = LlamaTokenLikelihoodProgress(
            completedTokenCount: 3,
            totalTokenCount: 8
        )

        XCTAssertEqual(progress.fractionCompleted, 0.375, accuracy: 0.000_001)
        XCTAssertEqual(
            LlamaTokenLikelihoodProgress(
                completedTokenCount: 0,
                totalTokenCount: 0
            ).fractionCompleted,
            1
        )
        XCTAssertEqual(
            LlamaTokenLikelihoodProgress(
                completedTokenCount: 9,
                totalTokenCount: 8
            ).fractionCompleted,
            1
        )
    }

    func testCalgacusStatsSummarizeTrace() {
        let trace = [
            CalgacusTraceEntry(index: 0, tokenID: 10, tokenText: "a", rank: 1, negativeLogProbability: 0.1),
            CalgacusTraceEntry(index: 1, tokenID: 11, tokenText: "b", rank: 5, negativeLogProbability: 0.2),
            CalgacusTraceEntry(index: 2, tokenID: 12, tokenText: "c", rank: 9, negativeLogProbability: 0.3)
        ]

        let stats = LlamaEngine.calgacusStats(for: trace)

        XCTAssertEqual(stats.tokenCount, 3)
        XCTAssertEqual(stats.maxRank, 9)
        XCTAssertEqual(stats.meanRank, 5)
        XCTAssertEqual(stats.medianRank, 5)
        XCTAssertEqual(stats.cumulativeNegativeLogProbability, 0.6, accuracy: 0.000_001)
        XCTAssertEqual(stats.averageNegativeLogProbability, 0.2, accuracy: 0.000_001)
    }

    func testCalgacusContextBudgetRejectsOverflow() {
        XCTAssertNoThrow(try LlamaEngine.calgacusValidateBudget(
            operation: "test",
            contextSize: 8,
            contextTokenCount: 3,
            payloadTokenCount: 5
        ))

        XCTAssertThrowsError(try LlamaEngine.calgacusValidateBudget(
            operation: "test",
            contextSize: 8,
            contextTokenCount: 4,
            payloadTokenCount: 5
        )) { error in
            guard case CalgacusError.contextBudgetExceeded = error else {
                return XCTFail("Expected contextBudgetExceeded, got \(error)")
            }
        }
    }

    func testPreflightLiveReportsExactCountsAndRejectsInvalidGrammarWhenModelPathProvided() async throws {
        let path = try LiveModelTestPolicy.requiredReadablePath(
            "LLAMA_TEST_MODEL_PATH",
            for: .textPreflight
        )

        let engine = LlamaEngine(configuration: LlamaEngineConfiguration(
            gpuLayerCount: 0,
            useMemoryMap: true,
            batchSizeLimit: 512,
            threadCount: 2,
            promptReserveTokens: 128
        ))
        let loaded = try await engine.load(
            modelAt: URL(fileURLWithPath: path),
            displayName: "Live Test Model",
            requestedContext: 1_024
        )

        let preflight = try await engine.preflight(
            system: "You are concise.",
            prompt: "Say hello.",
            options: GenerationOptions(maxOutputTokens: 64)
        )

        XCTAssertEqual(preflight.loadedContextSize, loaded.contextSize)
        XCTAssertEqual(preflight.modelTrainingContextSize, loaded.trainingContextSize)
        XCTAssertEqual(preflight.reservedOutputTokens, 128)
        XCTAssertEqual(preflight.requestedMaxOutputTokens, 64)
        XCTAssertTrue(preflight.usesExactTokenCounts)
        XCTAssertGreaterThan(preflight.promptTokens, 0)
        XCTAssertLessThanOrEqual(preflight.effectiveMaxOutputTokens, 64)

        do {
            _ = try await engine.preflight(
                system: "Return JSON.",
                prompt: #"Return {"ok": true}."#,
                options: GenerationOptions(grammar: #"root ::= "unterminated"#)
            )
            XCTFail("Expected invalid grammar to fail during preflight.")
        } catch let error as LLMEngineError {
            guard case .grammarParseFailed = error else {
                return XCTFail("Expected grammarParseFailed, got \(error)")
            }
        }

        await engine.unload()
    }

    func testVisionLiveClearsKVCacheAfterPriorTextGenerationWhenModelPathsProvided() async throws {
        let environment = ProcessInfo.processInfo.environment
        let modelPath = try LiveModelTestPolicy.requiredReadablePath(
            "CLLM_VISION_MODEL_PATH",
            for: .visionKVReset,
            environment: environment
        )
        let mmprojPath = try LiveModelTestPolicy.requiredReadablePath(
            "CLLM_VISION_MMPROJ_PATH",
            for: .visionKVReset,
            environment: environment
        )

        let requestedContext = environment["CLLM_VISION_CONTEXT"].flatMap(Int.init) ?? 4_096
        let engine = LlamaEngine(configuration: LlamaEngineConfiguration(
            gpuLayerCount: 0,
            useMemoryMap: true,
            batchSizeLimit: 128,
            threadCount: 2,
            promptReserveTokens: 16,
            accelerationPolicy: .disabled
        ))
        let loaded = try await engine.load(
            descriptor: LlamaModelDescriptor(
                url: URL(fileURLWithPath: modelPath),
                mmprojURL: URL(fileURLWithPath: mmprojPath),
                displayName: "Live Vision Test Model"
            ),
            requestedContext: requestedContext
        )
        XCTAssertTrue(loaded.supportsVision)

        _ = try await engine.generate(
            system: "You are concise.",
            prompt: "Reply with one word: ok.",
            options: GenerationOptions(temperature: 0, maxOutputTokens: 2),
            onPhaseAwareEvent: { _ in }
        )

        let image = LLMImageInput.rgb8(
            width: 2,
            height: 2,
            data: Data([
                255, 0, 0, 0, 255, 0,
                0, 0, 255, 255, 255, 255
            ])
        )
        let result = try await engine.generatePhased(
            messages: [
                LLMChatMessage(role: .user, content: [
                    .text("Describe the attached test image in a short phrase."),
                    .image(image)
                ])
            ],
            options: GenerationOptions(temperature: 0, maxOutputTokens: 8),
            onEvent: { _ in }
        )

        XCTAssertGreaterThan(result.promptTokens, 0)
        await engine.unload()
    }

    func testAudioLiveTranscribesWhenModelPathsProvided() async throws {
        let environment = ProcessInfo.processInfo.environment
        let modelPath = try LiveModelTestPolicy.requiredReadablePath(
            "CLLM_AUDIO_MODEL_PATH",
            for: .audioTranscription,
            environment: environment
        )
        let mmprojPath = try LiveModelTestPolicy.requiredReadablePath(
            "CLLM_AUDIO_MMPROJ_PATH",
            for: .audioTranscription,
            environment: environment
        )
        let samplePath = try LiveModelTestPolicy.requiredReadablePath(
            "CLLM_AUDIO_SAMPLE_PATH",
            for: .audioTranscription,
            environment: environment
        )

        let requestedContext = environment["CLLM_AUDIO_CONTEXT"].flatMap(Int.init) ?? 4_096
        let engine = LlamaEngine(configuration: LlamaEngineConfiguration(
            gpuLayerCount: 0,
            useMemoryMap: true,
            batchSizeLimit: 128,
            threadCount: 2,
            promptReserveTokens: 16,
            accelerationPolicy: .disabled
        ))
        let loaded = try await engine.load(
            descriptor: LlamaModelDescriptor(
                url: URL(fileURLWithPath: modelPath),
                mmprojURL: URL(fileURLWithPath: mmprojPath),
                displayName: "Live Audio Test Model"
            ),
            requestedContext: requestedContext
        )
        XCTAssertTrue(loaded.supportsAudio)

        let sampleData = try Data(contentsOf: URL(fileURLWithPath: samplePath))
        let result = try await engine.generatePhased(
            messages: [
                LLMChatMessage(role: .user, content: [
                    .text("Transcribe the following speech segment in its original language. Only output the transcription."),
                    .audio(.encoded(data: sampleData))
                ])
            ],
            options: GenerationOptions(temperature: 0, maxOutputTokens: 64),
            onEvent: { _ in }
        )

        XCTAssertGreaterThan(result.promptTokens, 0)
        XCTAssertFalse(result.finalText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        await engine.unload()
    }

    func testCalgacusLiveRoundTripWhenModelPathProvided() async throws {
        let path = try LiveModelTestPolicy.requiredReadablePath(
            "CALGACUS_TEST_MODEL_PATH",
            for: .calgacusRoundTrip
        )

        let engine = LlamaEngine(configuration: LlamaEngineConfiguration(
            gpuLayerCount: 0,
            useMemoryMap: true,
            batchSizeLimit: 512,
            threadCount: 2,
            promptReserveTokens: 0
        ))
        let loaded = try await engine.load(
            modelAt: URL(fileURLWithPath: path),
            requestedContext: 1_024
        )

        let secret = "The recipe is simple."
        let likelihoodText = Array(repeating: secret, count: 8).joined(separator: " ")
        let singleTokenStartedAt = Date()
        let singleTokenLikelihoods = try await engine.tokenLogLikelihoods(
            for: likelihoodText,
            maximumBatchTokenCount: 1
        )
        let singleTokenDuration = Date().timeIntervalSince(singleTokenStartedAt)
        let batchedStartedAt = Date()
        let likelihoods = try await engine.tokenLogLikelihoods(for: likelihoodText)
        let batchedDuration = Date().timeIntervalSince(batchedStartedAt)
        print(
            "Token likelihood live check: single=\(singleTokenDuration)s "
                + "batched=\(batchedDuration)s tokens=\(likelihoods.count)"
        )
        XCTAssertFalse(likelihoods.isEmpty)
        XCTAssertEqual(likelihoods.count, singleTokenLikelihoods.count)
        for (batched, singleToken) in zip(likelihoods, singleTokenLikelihoods) {
            XCTAssertEqual(batched.tokenID, singleToken.tokenID)
            XCTAssertEqual(batched.tokenBytes, singleToken.tokenBytes)
            XCTAssertEqual(batched.byteRange, singleToken.byteRange)
            XCTAssertGreaterThan(batched.rank, 0)
            XCTAssertTrue(batched.logProbability.isFinite)
        }
        XCTAssertEqual(
            likelihoods.reduce(into: Data()) { $0.append($1.tokenBytes) },
            Data(likelihoodText.utf8)
        )
        XCTAssertTrue(likelihoods.allSatisfy { $0.logProbability.isFinite })
        XCTAssertTrue(likelihoods.allSatisfy { $0.rank > 0 })

        let encoded = try await engine.encodeCalgacus(
            CalgacusEncodeRequest(
                secretText: secret,
                coverPrompt: "Write a friendly note about soup:",
                requestedContext: loaded.contextSize
            ),
            onEvent: { _ in }
        )
        let decoded = try await engine.decodeCalgacus(
            CalgacusDecodeRequest(
                coverText: encoded.coverText,
                coverPrompt: "Write a friendly note about soup:",
                requestedContext: loaded.contextSize
            ),
            onEvent: { _ in }
        )

        XCTAssertEqual(decoded.secretText, secret)
        await engine.unload()
    }

    func testLikelihoodStatisticsLiveParityAndOverheadWhenModelPathProvided() async throws {
        let path = try LiveModelTestPolicy.requiredReadablePath(
            "CLLM_LIKELIHOOD_TEST_MODEL_PATH",
            for: .likelihoodStatistics
        )
        let engine = LlamaEngine(configuration: LlamaEngineConfiguration(
            batchSizeLimit: 512,
            promptReserveTokens: 0
        ))
        _ = try await engine.load(
            modelAt: URL(fileURLWithPath: path),
            displayName: "Likelihood Statistics Benchmark",
            requestedContext: 2_048
        )

        let paragraph = "At dusk, the garden settled into shadow while a late bus turned the corner. "
            + "A causal language model assigns every next token a distribution conditioned on its context."
        let text = Array(repeating: paragraph, count: 8).joined(separator: " ")
        let curvatureOptions = LlamaTokenLikelihoodOptions(
            vocabularyStatistics: LlamaVocabularyStatisticsOptions()
        )
        let temperatureOptions = LlamaTokenLikelihoodOptions(
            vocabularyStatistics: LlamaVocabularyStatisticsOptions(
                temperatureNormalizationTemperatures: [0.6, 0.8, 1.2]
            )
        )

        _ = try await engine.tokenLogLikelihoods(for: text)

        var disabledDurations: [TimeInterval] = []
        var curvatureDurations: [TimeInterval] = []
        var temperatureDurations: [TimeInterval] = []
        var disabledResults: [LlamaTokenLikelihood] = []
        var curvatureResults: [LlamaTokenLikelihood] = []
        var temperatureResults: [LlamaTokenLikelihood] = []

        for iteration in 0..<3 {
            let modes = iteration.isMultiple(of: 2) ? [0, 1, 2] : [2, 1, 0]
            for mode in modes {
                let startedAt = Date()
                switch mode {
                case 0:
                    disabledResults = try await engine.tokenLogLikelihoods(for: text)
                    disabledDurations.append(Date().timeIntervalSince(startedAt))
                case 1:
                    curvatureResults = try await engine.tokenLogLikelihoods(
                        for: text,
                        options: curvatureOptions
                    )
                    curvatureDurations.append(Date().timeIntervalSince(startedAt))
                default:
                    temperatureResults = try await engine.tokenLogLikelihoods(
                        for: text,
                        options: temperatureOptions
                    )
                    temperatureDurations.append(Date().timeIntervalSince(startedAt))
                }
            }
        }

        XCTAssertEqual(disabledResults.count, curvatureResults.count)
        XCTAssertEqual(disabledResults.count, temperatureResults.count)
        for index in disabledResults.indices {
            let disabled = disabledResults[index]
            let curvature = curvatureResults[index]
            let temperature = temperatureResults[index]
            XCTAssertNil(disabled.vocabularyStatistics)
            XCTAssertEqual(curvature.tokenID, disabled.tokenID)
            XCTAssertEqual(curvature.rank, disabled.rank)
            XCTAssertEqual(curvature.logProbability, disabled.logProbability, accuracy: 0.000_000_000_001)
            XCTAssertEqual(temperature.tokenID, disabled.tokenID)
            XCTAssertEqual(temperature.rank, disabled.rank)
            XCTAssertEqual(temperature.logProbability, disabled.logProbability, accuracy: 0.000_000_000_001)
            let curvatureStatistics = try XCTUnwrap(curvature.vocabularyStatistics)
            XCTAssertTrue(curvatureStatistics.expectedLogProbability.isFinite)
            XCTAssertGreaterThanOrEqual(curvatureStatistics.logProbabilityVariance, 0)
            XCTAssertTrue(curvatureStatistics.temperatureNormalizations.isEmpty)
            let temperatureStatistics = try XCTUnwrap(temperature.vocabularyStatistics)
            XCTAssertEqual(temperatureStatistics.temperatureNormalizations.count, 3)
        }

        func median(_ values: [TimeInterval]) -> TimeInterval {
            let sorted = values.sorted()
            return sorted[sorted.count / 2]
        }
        let disabledMedian = median(disabledDurations)
        let curvatureMedian = median(curvatureDurations)
        let temperatureMedian = median(temperatureDurations)
        print(
            String(
                format: "Likelihood statistics live timing: tokens=%d disabled=%.4fs "
                    + "curvature=%.4fs (%+.1f%%) curvature+temperatures=%.4fs (%+.1f%%)",
                disabledResults.count,
                disabledMedian,
                curvatureMedian,
                (curvatureMedian / disabledMedian - 1) * 100,
                temperatureMedian,
                (temperatureMedian / disabledMedian - 1) * 100
            )
        )

        await engine.unload()
    }

    func testFallbackPromptDetectsGemmaWithoutAppConcepts() throws {
        let descriptor = LlamaModelDescriptor(
            url: URL(fileURLWithPath: "/tmp/gemma-3.gguf"),
            displayName: "Gemma 3",
            filename: "gemma-3.gguf"
        )

        let rendered = try LlamaEngine.fallbackPrompt(
            system: "System",
            user: "User",
            embeddedTemplate: nil,
            descriptor: descriptor
        )

        XCTAssertEqual(rendered.mode, .gemmaFallback)
        XCTAssertTrue(rendered.text.contains("<start_of_turn>user"))
        XCTAssertTrue(rendered.text.contains("<start_of_turn>model"))
        XCTAssertEqual(rendered.outputProfile.extraStopStrings, ["<end_of_turn>", "<start_of_turn>"])
    }

    func testFallbackPromptDetectsGemma4EmbeddedTemplate() throws {
        let template = try String(contentsOf: Self.gemma4TemplateURL, encoding: .utf8)

        let rendered = try LlamaEngine.fallbackPrompt(
            system: "System",
            user: "User",
            embeddedTemplate: template,
            descriptor: nil
        )

        XCTAssertEqual(rendered.mode, .gemmaFallback)
        XCTAssertTrue(rendered.text.contains("<|turn>system\nSystem<turn|>"))
        XCTAssertTrue(rendered.text.contains("<|turn>user\nUser<turn|>"))
        XCTAssertTrue(rendered.text.hasSuffix("<|turn>model\n<|channel>thought\n<channel|>"))
        XCTAssertFalse(rendered.text.contains("<start_of_turn>"))
        XCTAssertEqual(rendered.outputProfile.extraStopStrings, ["<turn|>", "<|turn>"])
    }

    func testFallbackPromptDetectsGemma4DescriptorWithoutEmbeddedTemplate() throws {
        let descriptor = LlamaModelDescriptor(
            url: URL(fileURLWithPath: "/tmp/google_gemma-4-26B-A4B-it-Q4_K_M.gguf"),
            displayName: "Gemma 4 26B A4B Instruct",
            filename: "google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
        )

        let rendered = try LlamaEngine.fallbackPrompt(
            system: "System",
            user: "User",
            embeddedTemplate: nil,
            descriptor: descriptor
        )

        XCTAssertEqual(rendered.mode, .gemmaFallback)
        XCTAssertTrue(rendered.text.contains("<|turn>system"))
        XCTAssertTrue(rendered.text.contains("<|turn>model"))
        XCTAssertFalse(rendered.text.contains("<start_of_turn>"))
    }

    func testFallbackPromptDetectsChatMLWithoutAppConcepts() throws {
        let descriptor = LlamaModelDescriptor(
            url: URL(fileURLWithPath: "/tmp/qwen.gguf"),
            displayName: "Qwen",
            filename: "qwen.gguf"
        )

        let rendered = try LlamaEngine.fallbackPrompt(
            system: "System",
            user: "User",
            embeddedTemplate: nil,
            descriptor: descriptor
        )

        XCTAssertEqual(rendered.mode, .chatMLFallback)
        XCTAssertTrue(rendered.text.contains("<|im_start|>system"))
        XCTAssertTrue(rendered.text.contains("<|im_start|>assistant"))
        XCTAssertEqual(rendered.outputProfile.extraStopStrings, ["<|im_end|>", "<|im_start|>"])
    }

    func testGenerationBoundaryStopsAtBalancedObjectOrArrayValue() {
        let object = LlamaEngine.firstGenerationBoundary(
            in: #"preamble {"items":[{"ok":true}]} trailing"#,
            stopSequences: [],
            stopAtBalancedJSON: true
        )
        XCTAssertEqual(object?.text, #"{"items":[{"ok":true}]}"#)
        XCTAssertEqual(object?.reason, "json-complete")

        let array = LlamaEngine.firstGenerationBoundary(
            in: #"lead [{"ok":true},{"ok":false}] trailing"#,
            stopSequences: [],
            stopAtBalancedJSON: true
        )
        XCTAssertEqual(array?.text, #"[{"ok":true},{"ok":false}]"#)
        XCTAssertEqual(array?.reason, "json-complete")
    }

    func testGenerationBoundaryUsesEarlierStopSequenceBeforeJSON() {
        let boundary = LlamaEngine.firstGenerationBoundary(
            in: #"prefix STOP {"ok":true}"#,
            stopSequences: ["STOP"],
            stopAtBalancedJSON: true
        )

        XCTAssertEqual(boundary?.text, "prefix ")
        XCTAssertEqual(boundary?.reason, "stop-sequence")
    }

    func testIncrementalUTF8AccumulatorMatchesWholeBufferDecodingAcrossSplitScalar() {
        let expected = "A🧪B"
        var raw = Data()
        var incremental = IncrementalUTF8Accumulator()
        var incrementalText = ""
        var legacyText = ""

        for byte in expected.utf8 {
            let piece = Data([byte])
            raw.append(piece)
            incrementalText.append(contentsOf: incremental.append(piece))
            if let decoded = String(data: raw, encoding: .utf8) {
                legacyText = decoded
            }

            XCTAssertEqual(incrementalText, legacyText)
            XCTAssertEqual(
                incremental.isAtUTF8Boundary,
                String(data: raw, encoding: .utf8) != nil
            )
        }

        XCTAssertEqual(incrementalText, expected)
    }

    func testIncrementalUTF8AccumulatorFreezesAfterIrrecoverablyInvalidPrefix() {
        var incremental = IncrementalUTF8Accumulator()

        XCTAssertEqual(incremental.append(Data([0xFF])), "")
        XCTAssertFalse(incremental.isAtUTF8Boundary)
        XCTAssertEqual(incremental.append(Data("later".utf8)), "")
        XCTAssertFalse(incremental.isAtUTF8Boundary)
    }

    func testIncrementalGenerationParserFindsStopSequenceAcrossPieces() {
        assertIncrementalGenerationParserMatchesLegacy(
            chunks: ["prefix S", "T", "OP", " trailing"],
            stopSequences: ["STOP"],
            stopAtBalancedJSON: false
        )
    }

    func testIncrementalGenerationParserFindsUnicodeStopAcrossGraphemeBoundary() {
        assertIncrementalGenerationParserMatchesLegacy(
            chunks: ["prefix e", "\u{301}", " trailing"],
            stopSequences: ["e\u{301}"],
            stopAtBalancedJSON: false
        )
    }

    func testIncrementalGenerationParserMatchesCanonicallyEquivalentStop() {
        assertIncrementalGenerationParserMatchesLegacy(
            chunks: ["prefix e", "\u{301}", " trailing"],
            stopSequences: ["é"],
            stopAtBalancedJSON: false
        )
    }

    func testIncrementalGenerationParserMatchesCanonicallyEquivalentPhaseDelimiters() {
        let pair = OutputDelimiterPair(open: "<think>", close: "é")
        let profile = OutputSanitizationProfile(thinkingPairs: [pair])
        let streamPlan = LlamaEngine.StreamPhasePlan(
            profile: profile,
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )
        let structuredPlan = LlamaEngine.StructuredOutputPlan(
            profile: profile,
            continuingOpenThinkingPairs: [],
            grammarMode: .lazy(grammar: "root ::= object", triggerPatterns: [])
        )

        assertIncrementalGenerationParserMatchesLegacy(
            chunks: ["<think>draft e", "\u{301}", #"{"answer":true}"#],
            streamPlan: streamPlan,
            structuredPlan: structuredPlan,
            stopSequences: [],
            stopAtBalancedJSON: true
        )
    }

    func testIncrementalGenerationParserTracksNestedBalancedJSONAndEscapes() {
        let text = #"preamble {"items":[{"text":"brace } and quote \""},[1,2]]}"#
        assertIncrementalGenerationParserMatchesLegacy(
            chunks: text.map(String.init),
            stopSequences: [],
            stopAtBalancedJSON: true
        )
    }

    func testIncrementalGenerationParserMatchesStructuredThinkingAndFinalPhases() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let profile = OutputSanitizationProfile(
            thinkingPairs: [pair],
            finalMarkers: ["<final>"]
        )
        let streamPlan = LlamaEngine.StreamPhasePlan(
            profile: profile,
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )
        let structuredPlan = LlamaEngine.StructuredOutputPlan(
            profile: profile,
            continuingOpenThinkingPairs: [],
            grammarMode: .lazy(
                grammar: "root ::= object",
                triggerPatterns: LlamaEngine.lazyGrammarTriggerPatterns(
                    profile: profile,
                    continuingOpenThinkingPairs: []
                )
            )
        )

        assertIncrementalGenerationParserMatchesLegacy(
            chunks: [
                "<thi", "nk>draft {\"ignored\":true}", "</thi", "nk>",
                "<fi", "nal>", " ", #"{"answer":[1,{"text":"}"}]}"#
            ],
            streamPlan: streamPlan,
            structuredPlan: structuredPlan,
            stopSequences: ["</think>"],
            stopAtBalancedJSON: true
        )
    }

    func testTerminalBoundaryPhaseUsesTrimmedText() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let plan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )
        var parser = IncrementalGenerationParser(
            streamPhasePlan: plan,
            structuredOutputPlan: nil,
            stopSequences: [pair.close],
            stopAtBalancedJSON: false
        )
        let text = "<think>draft</think>"
        let snapshot = parser.append(text, fullText: text)

        XCTAssertEqual(snapshot.phase, .final)
        XCTAssertEqual(snapshot.boundary?.text, "<think>draft")
        XCTAssertEqual(
            snapshot.boundary.map { LlamaEngine.streamContentPhase(in: $0.text, plan: plan) },
            .thinking
        )
    }

    func testIncrementalGenerationParserDoesNotCompleteIncompleteToolJSON() {
        assertIncrementalGenerationParserMatchesLegacy(
            chunks: ["prose {", "\"tool_calls\":[{", "\"name\":\"lookup\"", "}"],
            stopSequences: [],
            stopAtBalancedJSON: true
        )
    }

    func testIncrementalGenerationParserMatchesLegacyAtEverySplitPoint() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let profile = OutputSanitizationProfile(
            thinkingPairs: [pair],
            finalMarkers: ["<final>"]
        )
        let streamPlan = LlamaEngine.StreamPhasePlan(
            profile: profile,
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )
        let structuredPlan = LlamaEngine.StructuredOutputPlan(
            profile: profile,
            continuingOpenThinkingPairs: [],
            grammarMode: .lazy(grammar: "root ::= object", triggerPatterns: [])
        )
        let cases: [(String, LlamaEngine.StructuredOutputPlan?, [String], Bool)] = [
            ("ordinary output STOP trailing", nil, ["STOP"], false),
            (#"prefix {"items":[{"value":"}"}]} trailing"#, nil, [], true),
            (#"<|tool_call>call:lookup{"query":"swift"}<tool_call|>"#, nil, [], true),
            (#"<think>draft</think><final>{"answer":[1,2]}"#, structuredPlan, ["</think>"], true)
        ]

        for (text, plan, stops, stopsAtJSON) in cases {
            for split in text.indicesIncludingEnd {
                assertIncrementalGenerationParserMatchesLegacy(
                    chunks: [String(text[..<split]), String(text[split...])],
                    streamPlan: streamPlan,
                    structuredPlan: plan,
                    stopSequences: stops,
                    stopAtBalancedJSON: stopsAtJSON
                )
            }
        }

        let explicitlyFinalPlan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: [],
            startsInThinking: false
        )
        for text in ["<think>draft</think>", "<think>one</think><think>two</think>"] {
            for split in text.indicesIncludingEnd {
                assertIncrementalGenerationParserMatchesLegacy(
                    chunks: [String(text[..<split]), String(text[split...])],
                    streamPlan: explicitlyFinalPlan,
                    stopSequences: [],
                    stopAtBalancedJSON: false
                )
            }
        }
    }

    func testIncrementalToolStreamParserMatchesLegacyAtEverySplitPoint() {
        let cases = [
            #"I will check <|tool_call>call:lookup{"query":"swift"}<tool_call|> done"#,
            #"I will check <|tool_call>call:lookup{"query":"swift"}<tool_call|> <|tool_call>call:lookup{"query":"forums"}<tool_call|> done"#,
            #"Before {   "tool_calls":[{"function":{"name":"lookup","arguments":{"query":"swift"}}}]}"#,
            #"Before {"tool_calls":[{"function":{"name":"lookup","arguments":{"query":"swift"}}}"#,
            "Before {\"id\":\"\(String(repeating: "x", count: 100))\",\"name\":\"lookup\",\"arguments\":{}}",
            "Before [{\"id\":\"\(String(repeating: "x", count: 100))\",\"name\":\"lookup\",\"arguments\":{}}]",
            "Before <|tool_call>call:<tool_call|> after",
            #"{"x":{"name":"lookup","arguments":{}}}"#,
            #"{] noise {"name":"lookup","arguments":{}}"#,
            #"{"name":"outer","arguments":{"payload":"<|tool_call>call:inner{}<tool_call|>"}}"#,
            #"{"name":"outer","arguments":{"payload":"<|tool_call>call:<tool_call|> then <|tool_call>call:inner{}<tool_call|>"}}"#,
            "{\"name\":123,\"payload\":\"<|tool_call>call:oops\(String(repeating: "x", count: 100))\"}",
            #"<|tool_call>call:<tool_call|> after <|tool_call>call:lookup{}<tool_call|> done"#,
            #"<|tool_call>call:<tool_call|> <|tool_call>call:lookup{}<tool_call|> done"#,
            #"<|tool_call>call:inner{}<tool_call|> {"name":"outer","arguments":{}} done"#,
            #"<|tool_call>call:outer{"name":"inner","arguments":{}}<tool_call|>"#,
            #"<|tool_call>call:lookup{}<tool_call|>"# + String(repeating: " ", count: 200) + "done",
            "No tool call; marker-like <|tool_no and object {  not-a-tool"
        ]

        for text in cases {
            for split in text.indicesIncludingEnd {
                assertIncrementalToolStreamParserMatchesLegacy(
                    chunks: [String(text[..<split]), String(text[split...])]
                )
            }
            assertIncrementalToolStreamParserMatchesLegacy(chunks: text.map(String.init))
        }
    }

    func testToolCaptureBudgetPreservesOriginalPotentialNativeEnvelope() {
        let prefix = "🙂 preface "
        let malformedThenValid = prefix
            + #"<|tool_call>call:<tool_call|> then <|tool_call>call:lookup{}<tool_call|>"#
        XCTAssertTrue(IncrementalToolStreamParser.shouldPreserveCaptureBudget(
            in: malformedThenValid,
            captureStartUTF8Offset: prefix.utf8.count
        ))

        let releasedJSON = prefix + #"{"not_a_tool":true}"#
        XCTAssertFalse(IncrementalToolStreamParser.shouldPreserveCaptureBudget(
            in: releasedJSON,
            captureStartUTF8Offset: prefix.utf8.count
        ))
    }

    func testFallbackPromptRejectsUnknownTemplateFamily() {
        let descriptor = LlamaModelDescriptor(
            url: URL(fileURLWithPath: "/tmp/unknown.gguf"),
            displayName: "Unknown",
            filename: "unknown.gguf"
        )

        XCTAssertThrowsError(try LlamaEngine.fallbackPrompt(
            system: "System",
            user: "User",
            embeddedTemplate: nil,
            descriptor: descriptor
        )) { error in
            guard case LLMEngineError.chatTemplateUnavailable = error else {
                return XCTFail("Expected chatTemplateUnavailable, got \(error)")
            }
        }
    }

    func testEmbeddedTemplateFailureDoesNotInferFallbackFromDescriptor() {
        let descriptor = LlamaModelDescriptor(
            url: URL(fileURLWithPath: "/tmp/gemma-4.gguf"),
            displayName: "Gemma 4",
            filename: "gemma-4.gguf"
        )

        XCTAssertThrowsError(try LlamaEngine.fallbackPrompt(
            system: "System",
            user: "User",
            embeddedTemplate: "{{ unsupported_gemma4_marker }}",
            descriptor: descriptor
        )) { error in
            guard case LLMEngineError.chatTemplateUnavailable = error else {
                return XCTFail("Expected chatTemplateUnavailable, got \(error)")
            }
        }
    }
}

private extension CarbocationLlamaRuntimeTests {
    func assertIncrementalGenerationParserMatchesLegacy(
        chunks: [String],
        streamPlan: LlamaEngine.StreamPhasePlan = LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        ),
        structuredPlan: LlamaEngine.StructuredOutputPlan? = nil,
        stopSequences: [String],
        stopAtBalancedJSON: Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        var parser = IncrementalGenerationParser(
            streamPhasePlan: streamPlan,
            structuredOutputPlan: structuredPlan,
            stopSequences: stopSequences,
            stopAtBalancedJSON: stopAtBalancedJSON
        )
        var text = ""

        for chunk in chunks {
            text.append(contentsOf: chunk)
            let snapshot = parser.append(chunk, fullText: text)
            let expectedBoundary = if let structuredPlan {
                LlamaEngine.firstStructuredGenerationBoundary(
                    in: text,
                    stopSequences: stopSequences,
                    stopAtBalancedJSON: stopAtBalancedJSON,
                    plan: structuredPlan
                )
            } else {
                LlamaEngine.firstGenerationBoundary(
                    in: text,
                    stopSequences: stopSequences,
                    stopAtBalancedJSON: stopAtBalancedJSON
                )
            }

            XCTAssertEqual(
                snapshot.phase,
                LlamaEngine.streamContentPhase(in: text, plan: streamPlan),
                "Phase mismatch after appending \(String(reflecting: chunk))",
                file: file,
                line: line
            )
            XCTAssertEqual(
                snapshot.structuredPhase,
                structuredPlan.map { LlamaEngine.structuredOutputPhase(in: text, plan: $0) },
                "Structured phase mismatch after appending \(String(reflecting: chunk))",
                file: file,
                line: line
            )
            XCTAssertEqual(
                snapshot.boundary,
                expectedBoundary,
                "Boundary mismatch after appending \(String(reflecting: chunk))",
                file: file,
                line: line
            )

            if expectedBoundary != nil { break }
        }
    }

    func assertIncrementalToolStreamParserMatchesLegacy(
        chunks: [String],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        var parser = IncrementalToolStreamParser()
        var text = ""

        for chunk in chunks {
            text.append(contentsOf: chunk)
            let snapshot = parser.append(chunk, fullText: text)
            let expectedCapture = LlamaToolStreamInterpreter.startedToolCall(in: text)
            let expectedVisible = LlamaToolStreamInterpreter.visibleRawPrefix(in: text)
            let expectedCompleted = LlamaToolStreamInterpreter.completedToolCallForStreaming(in: text)
            let expectedPending = LlamaToolStreamInterpreter.pendingNativeToolCallBatch(in: text)

            XCTAssertEqual(
                snapshot.captureStartUTF8Offset,
                expectedCapture.map {
                    text.utf8.distance(
                        from: text.utf8.startIndex,
                        to: $0.range.lowerBound.samePosition(in: text.utf8)!
                    )
                },
                "Capture mismatch after appending \(String(reflecting: chunk)) to \(String(reflecting: text))",
                file: file,
                line: line
            )
            XCTAssertEqual(
                snapshot.visiblePrefixUTF8Count,
                expectedVisible.utf8.count,
                "Visible prefix mismatch after appending \(String(reflecting: chunk)) to \(String(reflecting: text))",
                file: file,
                line: line
            )
            XCTAssertEqual(
                snapshot.completed?.utf8Range,
                expectedCompleted.map { utf8Offsets(of: $0.range, in: text) },
                "Completed range mismatch after appending \(String(reflecting: chunk)) to \(String(reflecting: text))",
                file: file,
                line: line
            )
            XCTAssertEqual(
                snapshot.completed?.calls.map(\.name),
                expectedCompleted?.calls.map(\.name),
                file: file,
                line: line
            )
            XCTAssertEqual(
                snapshot.pendingNative?.utf8Range,
                expectedPending.map { utf8Offsets(of: $0.range, in: text) },
                "Pending range mismatch after appending \(String(reflecting: chunk)) to \(String(reflecting: text))",
                file: file,
                line: line
            )
            XCTAssertEqual(
                snapshot.pendingNative?.calls.map(\.name),
                expectedPending?.calls.map(\.name),
                file: file,
                line: line
            )

            if expectedCompleted != nil { break }
        }
    }

    func utf8Offsets(
        of range: Range<String.Index>,
        in text: String
    ) -> Range<Int> {
        let lower = text.utf8.distance(
            from: text.utf8.startIndex,
            to: range.lowerBound.samePosition(in: text.utf8)!
        )
        let upper = text.utf8.distance(
            from: text.utf8.startIndex,
            to: range.upperBound.samePosition(in: text.utf8)!
        )
        return lower..<upper
    }

    func makeAACEncodedAudioData(fileExtension: String, duration: TimeInterval) throws -> Data {
        let sampleRate = 44_100.0
        let frameCount = AVAudioFrameCount((duration * sampleRate).rounded(.up))
        guard let format = AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: 1
        ),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount),
              let channel = buffer.floatChannelData?[0]
        else {
            throw XCTSkip("AVFoundation could not allocate test audio buffers.")
        }

        buffer.frameLength = frameCount
        for index in 0..<Int(frameCount) {
            channel[index] = Float(0.2 * sin(2 * Double.pi * 440 * Double(index) / sampleRate))
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("CarbocationLocalLLMTest-\(UUID().uuidString)")
            .appendingPathExtension(fileExtension)
        defer {
            try? FileManager.default.removeItem(at: url)
        }

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 64_000
        ]

        do {
            do {
                let file = try AVAudioFile(
                    forWriting: url,
                    settings: settings,
                    commonFormat: .pcmFormatFloat32,
                    interleaved: false
                )
                try file.write(from: buffer)
            }
            let data = try Data(contentsOf: url)
            guard !data.isEmpty else {
                throw XCTSkip("AVFoundation created an empty \(fileExtension) test file.")
            }
            return data
        } catch let skip as XCTSkip {
            throw skip
        } catch {
            throw XCTSkip(
                "AVFoundation could not create \(fileExtension) test audio: \(error.localizedDescription)"
            )
        }
    }

    func thinkingTagPhasePlan() -> LlamaEngine.StreamPhasePlan {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        return LlamaEngine.StreamPhasePlan(
            profile: OutputSanitizationProfile(thinkingPairs: [pair]),
            continuingOpenThinkingPairs: [],
            startsInThinking: nil
        )
    }

    static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    static var gemma4TemplateURL: URL {
        packageRoot
            .appendingPathComponent("Vendor/llama.cpp/models/templates/google-gemma-4-31B-it.jinja")
    }

    static var gemma4VocabModelURL: URL {
        packageRoot
            .appendingPathComponent("Vendor/llama.cpp/models/ggml-vocab-gemma-4.gguf")
    }

    static var qwen35TemplateURL: URL {
        packageRoot
            .appendingPathComponent("Vendor/llama.cpp/models/templates/Qwen3.5-4B.jinja")
    }
}

private final class ToolPhasedEventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [LLMToolPhasedStreamEvent] = []

    func append(_ event: LLMToolPhasedStreamEvent) {
        lock.lock()
        storage.append(event)
        lock.unlock()
    }

    var generationEvents: [LLMGenerationStreamEvent] {
        lock.lock()
        defer { lock.unlock() }
        return storage.compactMap { event in
            guard case .generationEvent(let generationEvent) = event else { return nil }
            return generationEvent
        }
    }
}

private extension Array where Element == LLMGenerationStreamEvent {
    func contentText(for phase: LLMStreamContentPhase) -> String {
        reduce(into: "") { text, event in
            switch event {
            case .contentDelta(let eventPhase, let delta, _) where eventPhase == phase:
                text += delta
            case .contentSnapshot(let eventPhase, let snapshot, _, _) where eventPhase == phase:
                text = snapshot
            default:
                break
            }
        }
    }
}

private extension String {
    var indicesIncludingEnd: [String.Index] {
        Array(indices) + [endIndex]
    }
}

private actor SuspendedToolGate {
    private var started = false
    private var released = false
    private var startedContinuation: CheckedContinuation<Void, Never>?
    private var releaseContinuation: CheckedContinuation<Void, Never>?

    func waitUntilStarted() async {
        if started { return }
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                startedContinuation = continuation
            }
        } onCancel: {
            Task { await self.cancelStartedWait() }
        }
    }

    func suspendUntilReleased() async {
        started = true
        startedContinuation?.resume()
        startedContinuation = nil

        if released { return }
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                releaseContinuation = continuation
            }
        } onCancel: {
            Task { await self.release() }
        }
    }

    func release() {
        released = true
        releaseContinuation?.resume()
        releaseContinuation = nil
    }

    private func cancelStartedWait() {
        startedContinuation?.resume()
        startedContinuation = nil
    }
}

private actor ToolGenerationSegmentScript {
    private var prompts: [String] = []
    private var templateModes: [LLMChatTemplateMode] = []
    private var isInternalContinuationFlags: [Bool] = []

    func nextSegment(
        for input: LlamaEngine.ToolAwareGenerationSegmentOverrideInput
    ) -> LlamaEngine.ToolAwareGenerationSegmentOverrideOutput {
        prompts.append(input.renderedPrompt)
        templateModes.append(input.templateMode)
        isInternalContinuationFlags.append(input.isInternalContinuation)

        if prompts.count == 1 {
            return LlamaEngine.ToolAwareGenerationSegmentOverrideOutput(
                toolCalls: [
                    LLMToolCall(
                        executionID: "model_lookup",
                        name: "lookup",
                        arguments: ["query": "sp500"],
                        triggerPhase: .final
                    )
                ],
                stopReason: "tool-call-complete",
                triggerPhase: .final
            )
        }

        return LlamaEngine.ToolAwareGenerationSegmentOverrideOutput(
            finalText: "Final answer after tool.",
            stopReason: "eog",
            triggerPhase: .final,
            generatedTokens: 4
        )
    }

    func snapshot() -> (
        count: Int,
        prompts: [String],
        templateModes: [LLMChatTemplateMode],
        isInternalContinuationFlags: [Bool]
    ) {
        (prompts.count, prompts, templateModes, isInternalContinuationFlags)
    }
}

private enum TestTimeoutError: Error {
    case timedOut
}

private func withTimeout<T: Sendable>(
    seconds: UInt64,
    operation: @escaping @Sendable () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask {
            try await operation()
        }
        group.addTask {
            try await Task.sleep(nanoseconds: seconds * 1_000_000_000)
            throw TestTimeoutError.timedOut
        }

        let result = try await group.next()!
        group.cancelAll()
        return result
    }
}
