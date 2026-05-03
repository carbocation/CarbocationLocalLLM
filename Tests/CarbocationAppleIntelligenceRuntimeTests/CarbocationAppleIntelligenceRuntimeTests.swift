import CarbocationAppleIntelligenceRuntime
import CarbocationLocalLLM
import XCTest

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
        XCTAssertTrue(resolved.unsupportedFeatures.isEmpty)
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
