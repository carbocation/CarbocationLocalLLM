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

    @MainActor
    func testLoadPlanRefreshesBeforeResolvingInstalledModel() async throws {
        let root = try makeTemporaryDirectory()
        let library = ModelLibrary(root: root, contextLengthProbe: { _ in nil })
        let model = try installFixtureModel(
            displayName: "Planned Model",
            contextLength: 32_768,
            in: root
        )

        let stalePlan = await LocalLLMEngine.loadPlan(
            from: model.id.uuidString,
            in: library,
            refreshingLibrary: false
        )
        XCTAssertNil(stalePlan)

        let resolvedPlan = await LocalLLMEngine.loadPlan(from: model.id.uuidString, in: library)
        let plan = try XCTUnwrap(resolvedPlan)

        XCTAssertEqual(plan.selection, .installed(model.id))
        XCTAssertEqual(plan.displayName, "Planned Model")
        XCTAssertEqual(plan.requestedContext, LlamaContextPolicy.defaultAutoCap)
        XCTAssertEqual(plan.capabilities.contextSize, 32_768)
        XCTAssertTrue(plan.capabilities.supportsGrammar)
        XCTAssertTrue(plan.capabilities.usesExactTokenCounts)
    }

    @MainActor
    func testLoadPlanReturnsNilForInvalidAndDeletedSelections() async throws {
        let root = try makeTemporaryDirectory()
        let library = ModelLibrary(root: root, contextLengthProbe: { _ in nil })
        let model = try installFixtureModel(displayName: "Deleted Model", contextLength: 8_192, in: root)

        let invalidPlan = await LocalLLMEngine.loadPlan(from: "not-a-model", in: library)
        XCTAssertNil(invalidPlan)

        let existingPlan = await LocalLLMEngine.loadPlan(from: model.id.uuidString, in: library)
        XCTAssertNotNil(existingPlan)

        try FileManager.default.removeItem(at: model.directory(in: root))

        let deletedPlan = await LocalLLMEngine.loadPlan(from: model.id.uuidString, in: library)
        XCTAssertNil(deletedPlan)
    }

    @MainActor
    func testLoadPlanHonorsAutoAndManualContextPolicy() async throws {
        let root = try makeTemporaryDirectory()
        let library = ModelLibrary(root: root, contextLengthProbe: { _ in nil })
        let model = try installFixtureModel(
            displayName: "Context Model",
            contextLength: 65_536,
            in: root
        )
        let suiteName = "CarbocationLocalLLMRuntimeTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let automaticPlan = await LocalLLMEngine.loadPlan(
            from: model.id.uuidString,
            in: library,
            defaults: defaults
        )
        let automatic = try XCTUnwrap(automaticPlan)
        XCTAssertEqual(automatic.requestedContext, LlamaContextPolicy.defaultAutoCap)

        defaults.set(LlamaContextMode.manual.rawValue, forKey: "llama.contextMode")
        defaults.set(32_768, forKey: "llama.numCtx")

        let manualPlan = await LocalLLMEngine.loadPlan(
            from: model.id.uuidString,
            in: library,
            defaults: defaults,
            refreshingLibrary: false
        )
        let manual = try XCTUnwrap(manualPlan)
        XCTAssertEqual(manual.requestedContext, 32_768)
    }

    @MainActor
    func testLoadPlanUsesCalibratedContextOnlyInAutoMode() async throws {
        let root = try makeTemporaryDirectory()
        let library = ModelLibrary(root: root, contextLengthProbe: { _ in nil })
        let model = try installFixtureModel(
            displayName: "Calibrated Context Model",
            contextLength: 131_072,
            in: root
        )
        let suiteName = "CarbocationLocalLLMRuntimeCalibrationTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        defaults.set("device-a", forKey: LlamaContextCalibrationStore.deviceIDDefaultsKey)
        let store = LlamaContextCalibrationStore(defaults: defaults)
        let runtime = LocalLLMEngine.contextCalibrationRuntimeFingerprint()
        store.save(LlamaContextCalibrationRecord(
            key: store.key(for: model, runtime: runtime),
            maximumSupportedContext: 65_536,
            probedTiers: [
                LlamaContextCalibrationProbe(context: 32_768, succeeded: true),
                LlamaContextCalibrationProbe(context: 65_536, succeeded: true),
                LlamaContextCalibrationProbe(context: 131_072, succeeded: false)
            ]
        ))

        let autoPlan = await LocalLLMEngine.loadPlan(
            from: model.id.uuidString,
            in: library,
            defaults: defaults,
            refreshingLibrary: true,
            calibrationStore: store
        )
        XCTAssertEqual(try XCTUnwrap(autoPlan).requestedContext, 65_536)

        defaults.set(LlamaContextMode.manual.rawValue, forKey: "llama.contextMode")
        defaults.set(32_768, forKey: "llama.numCtx")

        let manualPlan = await LocalLLMEngine.loadPlan(
            from: model.id.uuidString,
            in: library,
            defaults: defaults,
            refreshingLibrary: false,
            calibrationStore: store
        )
        XCTAssertEqual(try XCTUnwrap(manualPlan).requestedContext, 32_768)
    }

    @MainActor
    func testLoadPlanHandlesSystemModelAvailability() async throws {
        let root = try makeTemporaryDirectory()
        let library = ModelLibrary(root: root, contextLengthProbe: { _ in nil })
        let storageValue = LLMSystemModelID.appleIntelligence.rawValue

        guard let option = LocalLLMEngine.availableSystemModels().first(where: {
            $0.selection == .system(.appleIntelligence)
        }) else {
            let plan = await LocalLLMEngine.loadPlan(from: storageValue, in: library)
            XCTAssertNil(plan)
            return
        }

        let resolvedPlan = await LocalLLMEngine.loadPlan(from: storageValue, in: library)
        let plan = try XCTUnwrap(resolvedPlan)

        XCTAssertEqual(plan.selection, option.selection)
        XCTAssertEqual(plan.displayName, option.displayName)
        XCTAssertEqual(plan.requestedContext, plan.capabilities.contextSize)
        XCTAssertEqual(plan.capabilities.contextSize, option.contextLength)
        XCTAssertFalse(plan.capabilities.supportsGrammar)
        XCTAssertFalse(plan.capabilities.usesExactTokenCounts)
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

    func testPreflightRequiresLoadedSelection() async {
        let engine = LocalLLMEngine()

        do {
            _ = try await engine.preflight(
                system: "You are concise.",
                prompt: "Say hello.",
                options: GenerationOptions(maxOutputTokens: 16)
            )
            XCTFail("Expected preflight to require a loaded selection.")
        } catch let error as LocalLLMEngineError {
            XCTAssertEqual(error.errorDescription, LocalLLMEngineError.noSelectionLoaded.errorDescription)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    @MainActor
    func testSystemModelPreflightUsesEstimatedTokenCountsWhenAvailable() async throws {
        guard let option = LocalLLMEngine.availableSystemModels().first(where: {
            $0.selection == .system(.appleIntelligence)
        }) else {
            return
        }

        let root = try makeTemporaryDirectory()
        let library = ModelLibrary(root: root, contextLengthProbe: { _ in nil })
        let engine = LocalLLMEngine()
        _ = try await engine.load(
            selection: option.selection,
            from: library,
            requestedContext: option.contextLength
        )

        let preflight = try await engine.preflight(
            system: "Return only JSON.",
            prompt: #"Return {"ok": true}."#,
            options: GenerationOptions(maxOutputTokens: 96, stopAtBalancedJSON: true)
        )

        XCTAssertEqual(preflight.loadedContextSize, option.contextLength)
        XCTAssertEqual(preflight.modelTrainingContextSize, option.contextLength)
        XCTAssertFalse(preflight.usesExactTokenCounts)
        XCTAssertEqual(preflight.requestedMaxOutputTokens, 96)
        XCTAssertEqual(preflight.templateMode, .unavailable)
        XCTAssertGreaterThan(preflight.promptTokens, 0)
        XCTAssertGreaterThanOrEqual(preflight.effectiveMaxOutputTokens, 0)
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

private func makeTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("CarbocationLocalLLMRuntimeTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func installFixtureModel(
    id: UUID = UUID(),
    displayName: String,
    contextLength: Int,
    in root: URL
) throws -> InstalledModel {
    let filename = "Fixture-Q4_K_M.gguf"
    let payload = Data("fake gguf".utf8)
    let model = InstalledModel(
        id: id,
        displayName: displayName,
        filename: filename,
        sizeBytes: Int64(payload.count),
        contextLength: contextLength,
        quantization: "Q4_K_M",
        source: .imported
    )
    let directory = model.directory(in: root)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try payload.write(to: model.weightsURL(in: root))
    try LocalLLMJSON.makePrettyEncoder().encode(model).write(to: model.metadataURL(in: root))
    return model
}
