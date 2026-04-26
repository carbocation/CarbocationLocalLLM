import CarbocationLocalLLM
import CarbocationLocalLLMRuntime
import XCTest

final class CarbocationLocalLLMRuntimeTests: XCTestCase {
    func testSelectionStorageRoundTripsInstalledAndSystemModels() throws {
        let id = UUID()
        XCTAssertEqual(LLMModelSelection(storageValue: id.uuidString), .installed(id))
        XCTAssertEqual(
            LLMModelSelection(storageValue: LLMSystemModelID.appleIntelligence.rawValue),
            .system(.appleIntelligence)
        )
        XCTAssertNil(LLMModelSelection(storageValue: "not-a-model"))
    }

    func testSelectionCodableUsesStorageValue() throws {
        let selection = LLMModelSelection.system(.appleIntelligence)
        let data = try JSONEncoder().encode(selection)

        XCTAssertEqual(String(data: data, encoding: .utf8), "\"system.apple-intelligence\"")
        XCTAssertEqual(try JSONDecoder().decode(LLMModelSelection.self, from: data), selection)
    }

    func testAvailableSystemModelsUseUnifiedSelectionIDs() {
        for option in LocalLLMEngine.availableSystemModels() {
            XCTAssertEqual(option.id, option.selection.storageValue)
            XCTAssertTrue(option.contextLength > 0)
        }
    }

    @MainActor
    func testCapabilitiesDescribeGrammarSupportByProvider() {
        let installedID = UUID()
        let installed = LocalLLMEngine.capabilities(for: .installed(installedID))
        XCTAssertTrue(installed.supportsGrammar)
        XCTAssertTrue(installed.usesExactTokenCounts)

        let system = LocalLLMEngine.capabilities(for: .system(.appleIntelligence))
        XCTAssertFalse(system.supportsGrammar)
        XCTAssertFalse(system.usesExactTokenCounts)
    }

    func testGenerateRequiresLoadedSelection() async {
        let engine = LocalLLMEngine()

        do {
            _ = try await engine.generate(
                system: "Return JSON.",
                prompt: #"Return {"ok": true, "message": "hello"}."#,
                options: GenerationOptions(maxOutputTokens: 16)
            ) { _ in }
            XCTFail("Expected generation to require a loaded selection.")
        } catch let error as LocalLLMEngineError {
            XCTAssertEqual(error.errorDescription, LocalLLMEngineError.noSelectionLoaded.errorDescription)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    @MainActor
    func testLiveSystemModelGenerationWhenExplicitlyEnabled() async throws {
        guard ProcessInfo.processInfo.environment["CARBOCATION_RUN_APPLE_INTELLIGENCE_LIVE_TEST"] == "1" else {
            throw XCTSkip("Set CARBOCATION_RUN_APPLE_INTELLIGENCE_LIVE_TEST=1 to run the live system-model smoke test.")
        }

        guard let option = LocalLLMEngine.availableSystemModels().first(where: {
            $0.selection == .system(.appleIntelligence)
        }) else {
            throw XCTSkip("Apple Intelligence is not available through LocalLLMEngine.")
        }

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("CarbocationLocalLLMRuntimeTests-\(UUID().uuidString)")
        let library = ModelLibrary(root: root, contextLengthProbe: { _ in nil })
        let engine = LocalLLMEngine()
        let loaded = try await engine.load(
            selection: option.selection,
            from: library,
            requestedContext: option.contextLength
        )

        XCTAssertEqual(loaded.selection, .system(.appleIntelligence))
        XCTAssertFalse(loaded.supportsGrammar)

        let response = try await engine.generate(
            system: "Return only JSON matching the requested schema. Do not include prose.",
            prompt: #"Return {"ok": true, "message": "hello"}."#,
            options: GenerationOptions(maxOutputTokens: 96, stopAtBalancedJSON: true)
        ) { _ in }

        let payload = try JSONSalvage.decode(RuntimeLivePayload.self, from: response)
        XCTAssertNotNil(payload.ok)
        XCTAssertNotNil(payload.message)
    }
}

private struct RuntimeLivePayload: Decodable {
    var ok: Bool?
    var message: String?
}
