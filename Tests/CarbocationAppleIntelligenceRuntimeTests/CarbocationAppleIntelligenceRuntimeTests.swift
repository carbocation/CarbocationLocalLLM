import CarbocationLocalLLM
import XCTest
@testable import CarbocationAppleIntelligenceRuntime

#if canImport(FoundationModels)
import FoundationModels
#endif

final class CarbocationAppleIntelligenceRuntimeTests: XCTestCase {
    func testAvailabilityReflectsBuildSDKWhenFoundationModelsIsMissing() {
        #if canImport(FoundationModels)
        XCTAssertTrue(AppleIntelligenceEngine.isBuiltWithFoundationModelsSDK)
        #else
        XCTAssertFalse(AppleIntelligenceEngine.isBuiltWithFoundationModelsSDK)
        XCTAssertEqual(AppleIntelligenceEngine.availability(), .unavailable(.sdkUnavailable))
        XCTAssertFalse(AppleIntelligenceEngine.availability().shouldOfferModelOption)
        #endif
    }

    func testSystemModelOptionOnlyAppearsWhenAvailable() {
        let availability = AppleIntelligenceEngine.availability()
        let option = AppleIntelligenceEngine.systemModelOption()

        if availability.isAvailable {
            XCTAssertEqual(option?.selection, .system(AppleIntelligenceEngine.systemModelID))
            XCTAssertEqual(option?.displayName, AppleIntelligenceEngine.displayName)
            XCTAssertEqual(option?.contextLength, availability.contextSize)
        } else {
            XCTAssertNil(option)
        }
    }

    func testOptionsMapperPrefersGreedyWhenTemperatureIsZero() {
        let options = GenerationOptions(
            temperature: 0,
            topP: 0.9,
            topK: 40,
            minP: 0.05,
            maxOutputTokens: 128,
            seed: 7,
            grammar: "root ::= object"
        )

        let resolved = AppleIntelligenceOptionsMapper.resolve(options)

        XCTAssertEqual(resolved.sampling, .greedy)
        XCTAssertNil(resolved.temperature)
        XCTAssertEqual(resolved.maximumResponseTokens, 128)
        XCTAssertEqual(resolved.unsupportedFeatures, [.grammar])
    }

    func testOptionsMapperUsesTopKBeforeTopPForNonGreedySampling() {
        let options = GenerationOptions(
            temperature: 0.4,
            topP: 0.7,
            topK: 12,
            seed: 99
        )

        let resolved = AppleIntelligenceOptionsMapper.resolve(options)

        XCTAssertEqual(resolved.sampling, .randomTopK(12, seed: 99))
        XCTAssertEqual(resolved.temperature, 0.4)
        XCTAssertNil(resolved.maximumResponseTokens)
        XCTAssertEqual(resolved.unsupportedFeatures, [.combinedSamplingFilters])
    }

    func testOptionsMapperFallsBackToProbabilityThreshold() {
        let options = GenerationOptions(
            temperature: 0.6,
            topP: 1.5,
            seed: 3
        )

        let resolved = AppleIntelligenceOptionsMapper.resolve(options)

        XCTAssertEqual(resolved.sampling, .randomProbabilityThreshold(1, seed: 3))
        XCTAssertEqual(resolved.temperature, 0.6)
    }

    func testOptionsMapperReportsUnsupportedLlamaOnlySamplingControls() {
        let options = GenerationOptions(
            temperature: 0.8,
            minP: 0.05,
            presencePenalty: 1.2,
            repetitionPenalty: 1.1
        )

        let resolved = AppleIntelligenceOptionsMapper.resolve(options)

        XCTAssertEqual(
            resolved.unsupportedFeatures,
            [.minP, .presencePenalty, .repetitionPenalty]
        )
    }

    func testOptionsMapperAllowsNeutralLlamaOnlySamplingControls() {
        let options = GenerationOptions(
            temperature: 0.8,
            topP: 1,
            topK: 20,
            minP: 0,
            presencePenalty: 0,
            repetitionPenalty: 1
        )

        let resolved = AppleIntelligenceOptionsMapper.resolve(options)

        XCTAssertEqual(resolved.sampling, .randomTopK(20, seed: nil))
        XCTAssertTrue(resolved.unsupportedFeatures.isEmpty)
    }

    func testConfigurationDefaultsPromptReserve() {
        XCTAssertEqual(
            AppleIntelligenceEngineConfiguration().promptReserveTokens,
            LLMGenerationBudget.outputTokenReserve
        )
    }

    func testPostProcessorTrimsAtEarliestStopSequence() {
        let options = GenerationOptions(stopSequences: ["END", "STOP"])
        let processed = AppleIntelligenceResponsePostProcessor.process(
            "alpha STOP beta END",
            options: options
        )

        XCTAssertEqual(processed.text, "alpha ")
        XCTAssertEqual(processed.stopReason, "stop-sequence")
    }

    func testPostProcessorTrimsAtBalancedJSONObject() {
        let options = GenerationOptions(stopAtBalancedJSON: true)
        let processed = AppleIntelligenceResponsePostProcessor.process(
            #"{"message":"brace } inside string","nested":{"ok":true}} trailing"#,
            options: options
        )

        XCTAssertEqual(processed.text, #"{"message":"brace } inside string","nested":{"ok":true}}"#)
        XCTAssertEqual(processed.stopReason, "json-complete")
    }

    func testPostProcessorSlicesBalancedJSONValueAfterPreamble() {
        let options = GenerationOptions(stopAtBalancedJSON: true)
        let object = AppleIntelligenceResponsePostProcessor.process(
            #"Here is JSON: {"items":[{"ok":true}]} trailing"#,
            options: options
        )
        let array = AppleIntelligenceResponsePostProcessor.process(
            #"Result: [{"ok":true},{"ok":false}] trailing"#,
            options: options
        )

        XCTAssertEqual(object.text, #"{"items":[{"ok":true}]}"#)
        XCTAssertEqual(object.stopReason, "json-complete")
        XCTAssertEqual(array.text, #"[{"ok":true},{"ok":false}]"#)
        XCTAssertEqual(array.stopReason, "json-complete")
    }

    func testPostProcessorChoosesEarliestBoundary() {
        let options = GenerationOptions(
            stopSequences: ["STOP"],
            stopAtBalancedJSON: true
        )
        let processed = AppleIntelligenceResponsePostProcessor.process(
            #"prefix STOP {"ok":true}"#,
            options: options
        )

        XCTAssertEqual(processed.text, "prefix ")
        XCTAssertEqual(processed.stopReason, "stop-sequence")
    }

    #if canImport(FoundationModels)
    func testNativeToolSchemaMapperAcceptsSupportedJSONSchemaShapes() throws {
        guard #available(iOS 26.0, macOS 26.0, *) else {
            throw XCTSkip("Foundation Models tool schemas require OS 26.")
        }

        let tool = LLMToolDefinition(
            name: "lookup",
            description: "Lookup records.",
            parameters: [
                "type": "object",
                "properties": [
                    "query": [
                        "type": "string",
                        "description": "Search query."
                    ],
                    "limit": ["type": "integer"],
                    "include_archived": ["type": "boolean"],
                    "scores": [
                        "type": "array",
                        "items": ["type": "number"]
                    ],
                    "mode": [
                        "type": "string",
                        "enum": ["fast", "deep"]
                    ]
                ],
                "required": ["query"]
            ]
        )

        XCTAssertNoThrow(try AppleNativeToolSchemaMapper.generationSchema(for: tool))
    }

    func testGeneratedContentRoundTripsToolJSON() throws {
        guard #available(iOS 26.0, macOS 26.0, *) else {
            throw XCTSkip("GeneratedContent JSON bridging requires OS 26.")
        }

        let original: LLMJSONValue = [
            "query": "swift",
            "limit": 3,
            "flags": ["exact": true]
        ]
        let content = try GeneratedContent(json: original.jsonString(prettyPrinted: false))
        let decoded = try LLMJSONValue(jsonString: content.jsonString)

        XCTAssertEqual(decoded.string(forKey: "query"), "swift")
        XCTAssertEqual(decoded.double(forKey: "limit"), 3)
        XCTAssertEqual(decoded.value(forKey: "flags")?.value(forKey: "exact"), .bool(true))
    }

    func testNativeToolRecorderRecordsLifecycleEvents() async throws {
        guard #available(iOS 26.0, macOS 26.0, *) else {
            throw XCTSkip("Foundation Models tool recording requires OS 26.")
        }

        let recorder = AppleToolEventRecorder()
        let nativeRecorder = AppleNativeToolRecorder(
            maxToolCalls: 2,
            onPhaseAwareEvent: { recorder.append($0) }
        )
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "lookup", description: "Lookup records.")
        ) { arguments in
            ["ok": true, "query": arguments.value(forKey: "query") ?? .null]
        }

        let output = try await nativeRecorder.execute(
            tool: tool,
            arguments: GeneratedContent(json: #"{"query":"swift"}"#)
        )
        let decoded = try LLMJSONValue(jsonString: output.jsonString)

        XCTAssertEqual(decoded.string(forKey: "query"), "swift")
        XCTAssertEqual(nativeRecorder.calls.map(\.executionID), ["call_1"])
        XCTAssertNil(nativeRecorder.calls.first?.triggerPhase)
        XCTAssertEqual(nativeRecorder.outputs.map(\.callID), ["call_1"])
        XCTAssertEqual(recorder.events.startedToolCallIDs, ["call_1"])
        XCTAssertEqual(recorder.events.completedToolCallIDs, ["call_1"])
    }

    func testNativeToolRecorderReturnsToolErrorsAsGeneratedContent() async throws {
        guard #available(iOS 26.0, macOS 26.0, *) else {
            throw XCTSkip("Foundation Models tool recording requires OS 26.")
        }

        let recorder = AppleToolEventRecorder()
        let nativeRecorder = AppleNativeToolRecorder(
            maxToolCalls: 1,
            onPhaseAwareEvent: { recorder.append($0) }
        )
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "failing", description: "Always fails.")
        ) { _ in
            throw AppleNativeToolTestError.expected
        }

        let output = try await nativeRecorder.execute(
            tool: tool,
            arguments: GeneratedContent(json: #"{"query":"swift"}"#)
        )
        let decoded = try LLMJSONValue(jsonString: output.jsonString)

        XCTAssertEqual(decoded.value(forKey: "error")?.string(forKey: "code"), "tool_execution_failed")
        XCTAssertEqual(nativeRecorder.outputs.first?.isError, true)
        XCTAssertEqual(recorder.events.startedToolCallIDs, ["call_1"])
        XCTAssertEqual(recorder.events.failedToolCallIDs, ["call_1"])
    }

    func testNativeToolRecorderReturnsLimitErrorsAsGeneratedContent() async throws {
        guard #available(iOS 26.0, macOS 26.0, *) else {
            throw XCTSkip("Foundation Models tool recording requires OS 26.")
        }

        let recorder = AppleToolEventRecorder()
        let nativeRecorder = AppleNativeToolRecorder(
            maxToolCalls: 0,
            onPhaseAwareEvent: { recorder.append($0) }
        )
        let tool = LLMTool(
            definition: LLMToolDefinition(name: "lookup", description: "Lookup records.")
        ) { _ in
            ["ok": true]
        }

        let output = try await nativeRecorder.execute(
            tool: tool,
            arguments: GeneratedContent(json: #"{"query":"swift"}"#)
        )
        let decoded = try LLMJSONValue(jsonString: output.jsonString)

        XCTAssertEqual(decoded.value(forKey: "error")?.string(forKey: "code"), "max_tool_rounds")
        XCTAssertEqual(nativeRecorder.executedToolCallCount, 0)
        XCTAssertTrue(nativeRecorder.reachedToolLimit)
        XCTAssertEqual(recorder.events.failedToolCallIDs, ["call_1"])
    }
    #endif

    func testLiveGenerationWhenExplicitlyEnabled() async throws {
        guard ProcessInfo.processInfo.environment["CARBOCATION_RUN_APPLE_INTELLIGENCE_LIVE_TEST"] == "1" else {
            throw XCTSkip("Set CARBOCATION_RUN_APPLE_INTELLIGENCE_LIVE_TEST=1 to run the live Apple Intelligence smoke test.")
        }

        let availability = AppleIntelligenceEngine.availability()
        guard availability.isAvailable else {
            throw XCTSkip(availability.displayMessage)
        }

        let engine = AppleIntelligenceEngine()
        let response = try await engine.generate(
            system: "Return only JSON matching the requested schema. Do not include prose.",
            prompt: #"Return {"ok": true, "message": "hello"}."#,
            options: GenerationOptions(maxOutputTokens: 96, stopAtBalancedJSON: true)
        ) { _ in }

        let payload = try JSONSalvage.decode(AppleIntelligenceLivePayload.self, from: response)
        XCTAssertNotNil(payload.ok)
        XCTAssertNotNil(payload.message)
    }

    #if !canImport(FoundationModels)
    func testEngineThrowsUnavailableWhenBuiltWithoutFoundationModels() async {
        let engine = AppleIntelligenceEngine()

        do {
            _ = try await engine.generate(
                system: "You are concise.",
                prompt: "Say hello.",
                options: .extractionSafe
            ) { _ in }
            XCTFail("Expected Apple Intelligence generation to be unavailable.")
        } catch let error as AppleIntelligenceEngineError {
            guard case .unavailable(.unavailable(.sdkUnavailable)) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testEnginePreflightThrowsUnavailableWhenBuiltWithoutFoundationModels() async {
        let engine = AppleIntelligenceEngine()

        do {
            _ = try await engine.preflight(
                system: "You are concise.",
                prompt: "Say hello.",
                options: .extractionSafe
            )
            XCTFail("Expected Apple Intelligence preflight to be unavailable.")
        } catch let error as AppleIntelligenceEngineError {
            guard case .unavailable(.unavailable(.sdkUnavailable)) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testSessionThrowsUnavailableWhenBuiltWithoutFoundationModels() async {
        let session = AppleIntelligenceSession(system: "You are concise.")

        do {
            _ = try await session.generate(
                prompt: "Say hello.",
                options: .extractionSafe
            ) { _ in }
            XCTFail("Expected Apple Intelligence session generation to be unavailable.")
        } catch let error as AppleIntelligenceEngineError {
            guard case .unavailable(.unavailable(.sdkUnavailable)) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testSessionPreflightThrowsUnavailableWhenBuiltWithoutFoundationModels() async {
        let session = AppleIntelligenceSession(system: "You are concise.")

        do {
            _ = try await session.preflight(
                prompt: "Say hello.",
                options: .extractionSafe
            )
            XCTFail("Expected Apple Intelligence session preflight to be unavailable.")
        } catch let error as AppleIntelligenceEngineError {
            guard case .unavailable(.unavailable(.sdkUnavailable)) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
    #endif
}

private struct AppleIntelligenceLivePayload: Decodable {
    var ok: Bool?
    var message: String?
}

private final class AppleToolEventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [LLMToolPhaseAwareStreamEvent] = []

    func append(_ event: LLMToolPhaseAwareStreamEvent) {
        lock.lock()
        storage.append(event)
        lock.unlock()
    }

    var events: [LLMToolPhaseAwareStreamEvent] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

private extension Array where Element == LLMToolPhaseAwareStreamEvent {
    var startedToolCallIDs: [String] {
        compactMap { event in
            if case .toolCallStarted(let call) = event {
                return call.id
            }
            return nil
        }
    }

    var failedToolCallIDs: [String] {
        compactMap { event in
            if case .toolCallFailed(let output) = event {
                return output.callID
            }
            return nil
        }
    }

    var completedToolCallIDs: [String] {
        compactMap { event in
            if case .toolCallCompleted(let output) = event {
                return output.callID
            }
            return nil
        }
    }
}

private enum AppleNativeToolTestError: Error {
    case expected
}
