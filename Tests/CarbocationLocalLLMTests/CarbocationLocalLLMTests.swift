import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import CarbocationLocalLLM

final class CarbocationLocalLLMTests: XCTestCase {
    func testModelStorageKeepsCarbocationSharedGroupAlias() {
        XCTAssertEqual(ModelStorage.carbocationSharedGroupID, "group.com.carbocation.shared")
        XCTAssertEqual(ModelStorage.defaultSharedGroupID, ModelStorage.carbocationSharedGroupID)
    }

    func testModelStorageUsesCustomSharedGroupIdentifier() throws {
        let groupRoot = try makeTemporaryDirectory()
        var requestedIdentifier: String?

        let modelsDirectory = ModelStorage.modelsDirectory(
            sharedGroupIdentifier: "group.com.example.shared",
            appSupportFolderName: "ExampleApp",
            sharedGroupRootResolver: { identifier, _ in
                requestedIdentifier = identifier
                return identifier == "group.com.example.shared" ? groupRoot : nil
            }
        )

        let expected = groupRoot.appendingPathComponent("Models", isDirectory: true)
        XCTAssertEqual(requestedIdentifier, "group.com.example.shared")
        XCTAssertEqual(modelsDirectory.standardizedFileURL.path, expected.standardizedFileURL.path)
    }

    func testModelStorageFallsBackToApplicationSupportWhenSharedGroupUnavailable() {
        let appSupportFolderName = "ExampleApp-\(UUID().uuidString)"

        let modelsDirectory = ModelStorage.modelsDirectory(
            sharedGroupIdentifier: "group.com.example.missing",
            appSupportFolderName: appSupportFolderName,
            sharedGroupRootResolver: { _, _ in nil }
        )

        let expected = ModelStorage.appSupportDirectory(appSupportFolderName: appSupportFolderName)
            .appendingPathComponent("Models", isDirectory: true)
        XCTAssertEqual(modelsDirectory.standardizedFileURL.path, expected.standardizedFileURL.path)
    }

    #if os(macOS)
    func testHuggingFaceHubCacheDirectoryUsesEnvironmentPrecedence() throws {
        let root = try makeTemporaryDirectory()
        let direct = root.appendingPathComponent("direct-hub", isDirectory: true)
        let hfHome = root.appendingPathComponent("hf-home", isDirectory: true)
        let xdg = root.appendingPathComponent("xdg-cache", isDirectory: true)

        XCTAssertEqual(
            ModelStorage.huggingFaceHubCacheDirectory(
                environment: [
                    "HF_HUB_CACHE": direct.path,
                    "HF_HOME": hfHome.path,
                    "XDG_CACHE_HOME": xdg.path
                ]
            )?.path,
            direct.standardizedFileURL.path
        )

        XCTAssertEqual(
            ModelStorage.huggingFaceHubCacheDirectory(
                environment: [
                    "HF_HOME": hfHome.path,
                    "XDG_CACHE_HOME": xdg.path
                ]
            )?.path,
            hfHome.appendingPathComponent("hub", isDirectory: true).standardizedFileURL.path
        )

        XCTAssertEqual(
            ModelStorage.huggingFaceHubCacheDirectory(
                environment: ["XDG_CACHE_HOME": xdg.path]
            )?.path,
            xdg.appendingPathComponent("huggingface", isDirectory: true)
                .appendingPathComponent("hub", isDirectory: true)
                .standardizedFileURL.path
        )
    }
    #endif

    func testContextCalibrationStoreSeparatesDeviceModelAndRuntimeKeys() {
        let suiteName = "CarbocationLocalLLMCalibrationTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        defaults.set("device-a", forKey: LlamaContextCalibrationStore.deviceIDDefaultsKey)

        let store = LlamaContextCalibrationStore(defaults: defaults)
        let model = InstalledModel(
            displayName: "Calibration Model",
            filename: "model-Q4_K_M.gguf",
            sizeBytes: 2_000_000,
            contextLength: 65_536,
            quantization: "Q4_K_M",
            source: .imported,
            sha256: "abc123"
        )
        let runtime = calibrationRuntime(batchSizeLimit: 2_048)
        let key = store.key(for: model, runtime: runtime)
        let record = LlamaContextCalibrationRecord(
            key: key,
            maximumSupportedContext: 32_768,
            probedTiers: [
                LlamaContextCalibrationProbe(context: 16_384, succeeded: true),
                LlamaContextCalibrationProbe(context: 32_768, succeeded: true),
                LlamaContextCalibrationProbe(context: 65_536, succeeded: false)
            ]
        )

        store.save(record)

        XCTAssertEqual(
            store.record(for: model, runtime: runtime)?.maximumSupportedContext,
            32_768
        )
        XCTAssertNil(store.record(
            for: model,
            runtime: calibrationRuntime(batchSizeLimit: 1_024)
        ))

        defaults.set("device-b", forKey: LlamaContextCalibrationStore.deviceIDDefaultsKey)
        XCTAssertNil(store.record(for: model, runtime: runtime))
    }

    func testContextCalibrationAlgorithmVersionInvalidatesOlderRecords() {
        let suiteName = "CarbocationLocalLLMCalibrationVersionTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        defaults.set("device-a", forKey: LlamaContextCalibrationStore.deviceIDDefaultsKey)

        let store = LlamaContextCalibrationStore(defaults: defaults)
        let model = InstalledModel(
            displayName: "Calibration Model",
            filename: "model-Q4_K_M.gguf",
            sizeBytes: 2_000_000,
            contextLength: 65_536,
            quantization: "Q4_K_M",
            source: .imported,
            sha256: "abc123"
        )
        let initOnlyRuntime = calibrationRuntime(batchSizeLimit: 2_048, algorithmVersion: 1)
        let overlyConservativeRuntime = calibrationRuntime(batchSizeLimit: 2_048, algorithmVersion: 2)
        let overlyLiberalRuntime = calibrationRuntime(batchSizeLimit: 2_048, algorithmVersion: 3)
        let stillTooLiberalRuntime = calibrationRuntime(batchSizeLimit: 2_048, algorithmVersion: 4)
        let missingModelReserveRuntime = calibrationRuntime(batchSizeLimit: 2_048, algorithmVersion: 5)
        let currentRuntime = calibrationRuntime(batchSizeLimit: 2_048)
        let initOnlyKey = store.key(for: model, runtime: initOnlyRuntime)
        let overlyConservativeKey = store.key(for: model, runtime: overlyConservativeRuntime)
        let overlyLiberalKey = store.key(for: model, runtime: overlyLiberalRuntime)
        let stillTooLiberalKey = store.key(for: model, runtime: stillTooLiberalRuntime)
        let missingModelReserveKey = store.key(for: model, runtime: missingModelReserveRuntime)

        store.save(LlamaContextCalibrationRecord(
            key: initOnlyKey,
            maximumSupportedContext: 65_536,
            probedTiers: [
                LlamaContextCalibrationProbe(context: 65_536, succeeded: true)
            ]
        ))
        store.save(LlamaContextCalibrationRecord(
            key: overlyConservativeKey,
            maximumSupportedContext: 16_384,
            probedTiers: [
                LlamaContextCalibrationProbe(context: 16_384, succeeded: true)
            ]
        ))
        store.save(LlamaContextCalibrationRecord(
            key: overlyLiberalKey,
            maximumSupportedContext: 65_536,
            probedTiers: [
                LlamaContextCalibrationProbe(context: 65_536, succeeded: true)
            ]
        ))
        store.save(LlamaContextCalibrationRecord(
            key: stillTooLiberalKey,
            maximumSupportedContext: 65_536,
            probedTiers: [
                LlamaContextCalibrationProbe(context: 65_536, succeeded: true)
            ]
        ))
        store.save(LlamaContextCalibrationRecord(
            key: missingModelReserveKey,
            maximumSupportedContext: 65_536,
            probedTiers: [
                LlamaContextCalibrationProbe(context: 65_536, succeeded: true)
            ]
        ))

        XCTAssertGreaterThan(LlamaContextCalibrationAlgorithm.version, 5)
        XCTAssertEqual(
            store.record(for: model, runtime: initOnlyRuntime)?.maximumSupportedContext,
            65_536
        )
        XCTAssertEqual(
            store.record(for: model, runtime: overlyConservativeRuntime)?.maximumSupportedContext,
            16_384
        )
        XCTAssertEqual(
            store.record(for: model, runtime: overlyLiberalRuntime)?.maximumSupportedContext,
            65_536
        )
        XCTAssertEqual(
            store.record(for: model, runtime: stillTooLiberalRuntime)?.maximumSupportedContext,
            65_536
        )
        XCTAssertEqual(
            store.record(for: model, runtime: missingModelReserveRuntime)?.maximumSupportedContext,
            65_536
        )
        XCTAssertNil(store.record(for: model, runtime: currentRuntime))
    }

    func testContextCalibrationKeyDistinguishesMTPRuntime() {
        let model = InstalledModel(
            displayName: "Calibration Model",
            filename: "model-Q4_K_M.gguf",
            sizeBytes: 2_000_000,
            contextLength: 65_536,
            quantization: "Q4_K_M",
            source: .imported,
            sha256: "abc123"
        )
        func storageKey(mtpEnabled: Bool, draftTokens: Int) -> String {
            LlamaContextCalibrationKey(
                deviceID: "device-a",
                model: LlamaContextCalibrationModelFingerprint(model: model),
                runtime: LlamaContextCalibrationRuntimeFingerprint(
                    platform: "macOS",
                    gpuLayerCount: 999,
                    useMemoryMap: true,
                    batchSizeLimit: 2_048,
                    threadCount: 4,
                    mtpAccelerationEnabled: mtpEnabled,
                    mtpMaxDraftTokens: draftTokens
                )
            ).storageKey
        }

        // Toggling MTP, or changing the draft-token count, changes the runtime's
        // memory footprint, so calibration records must not be shared across them.
        XCTAssertNotEqual(
            storageKey(mtpEnabled: true, draftTokens: 1),
            storageKey(mtpEnabled: false, draftTokens: 0)
        )
        XCTAssertNotEqual(
            storageKey(mtpEnabled: true, draftTokens: 1),
            storageKey(mtpEnabled: true, draftTokens: 4)
        )
        XCTAssertEqual(
            storageKey(mtpEnabled: true, draftTokens: 1),
            storageKey(mtpEnabled: true, draftTokens: 1)
        )
    }

    func testContextCalibrationSearchUsesCoarsePowerOfTwoBisect() async throws {
        let candidates = LlamaContextCalibrationAlgorithm.powerOfTwoTiers(upTo: 65_536)
        var probed: [Int] = []

        let result = try await LlamaContextCalibrationAlgorithm.search(candidates: candidates) { context in
            probed.append(context)
            return context <= 16_384
        }

        XCTAssertEqual(candidates, [512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536])
        XCTAssertEqual(result.maximumSupportedContext, 16_384)
        XCTAssertLessThan(probed.count, candidates.count)
        XCTAssertTrue(result.probes.contains(LlamaContextCalibrationProbe(context: 16_384, succeeded: true)))
        XCTAssertTrue(result.probes.contains { !$0.succeeded })
    }

    func testContextCalibrationSearchUsesDecodeSuccessfulCandidate() async throws {
        let candidates = LlamaContextCalibrationAlgorithm.powerOfTwoTiers(upTo: 65_536)
        let initSuccessfulLimit = 65_536
        let decodeSuccessfulLimit = 16_384
        var probed: [(context: Int, initSucceeded: Bool, decodeSucceeded: Bool)] = []

        let result = try await LlamaContextCalibrationAlgorithm.search(candidates: candidates) { context in
            let initSucceeded = context <= initSuccessfulLimit
            let decodeSucceeded = context <= decodeSuccessfulLimit
            probed.append((
                context: context,
                initSucceeded: initSucceeded,
                decodeSucceeded: decodeSucceeded
            ))
            XCTAssertTrue(initSucceeded)
            return initSucceeded && decodeSucceeded
        }

        XCTAssertEqual(result.maximumSupportedContext, decodeSuccessfulLimit)
        XCTAssertNotEqual(result.maximumSupportedContext, initSuccessfulLimit)
        XCTAssertTrue(probed.contains { $0.initSucceeded && !$0.decodeSucceeded })
    }

    func testInstalledModelInfersQuantization() {
        XCTAssertEqual(InstalledModel.inferQuantization(from: "Qwen2.5-7B-Instruct-Q4_K_M.gguf"), "Q4_K_M")
        XCTAssertEqual(InstalledModel.inferQuantization(from: "Phi-3.5-mini-instruct-Q5_K_M.gguf"), "Q5_K_M")
        XCTAssertNil(InstalledModel.inferQuantization(from: "unknown-model.gguf"))
    }

    func testContextPolicySupportsAppSpecificAutoCaps() {
        XCTAssertEqual(
            LlamaContextPolicy.resolvedRequestedContext(
                trainingContext: 262_144,
                mode: .auto,
                manualContext: 8_192
            ),
            16_384
        )
        XCTAssertEqual(
            LlamaContextPolicy.resolvedRequestedContext(
                trainingContext: 262_144,
                mode: .auto,
                manualContext: 8_192,
                autoCap: 32_768
            ),
            32_768
        )
        XCTAssertEqual(
            LlamaContextPolicy.resolvedRequestedContext(
                trainingContext: 262_144,
                mode: .auto,
                manualContext: 8_192,
                autoCap: 131_072,
                maximumSupportedContext: 65_536
            ),
            65_536
        )
        XCTAssertEqual(
            LlamaContextPolicy.resolvedRequestedContext(
                trainingContext: 4_096,
                mode: .manual,
                manualContext: 32_768,
                maximumSupportedContext: 16_384
            ),
            32_768
        )
    }

    func testContextPolicyReadsAutoContextLimitPreference() {
        let suiteName = "CarbocationLocalLLMTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let keys = LlamaContextPreferenceKeys()

        XCTAssertEqual(
            LlamaContextPolicy.resolvedRequestedContext(
                trainingContext: 262_144,
                defaults: defaults,
                keys: keys
            ),
            LlamaContextPolicy.defaultAutoCap
        )

        defaults.set(65_536, forKey: keys.autoContextLimit)
        XCTAssertEqual(
            LlamaContextPolicy.resolvedRequestedContext(
                trainingContext: 262_144,
                defaults: defaults,
                keys: keys
            ),
            65_536
        )

        XCTAssertEqual(
            LlamaContextPolicy.resolvedRequestedContext(
                trainingContext: 262_144,
                defaults: defaults,
                keys: keys,
                maximumSupportedContext: 32_768
            ),
            32_768
        )
    }

    func testContextPolicyCanTrackSelectedModelMaximumInAutoMode() {
        let suiteName = "CarbocationLocalLLMTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let keys = LlamaContextPreferenceKeys()

        defaults.set(65_536, forKey: keys.autoContextLimit)
        defaults.set(true, forKey: keys.autoContextLimitUsesMaximum)

        XCTAssertEqual(
            LlamaContextPolicy.resolvedRequestedContext(
                trainingContext: 262_144,
                defaults: defaults,
                keys: keys,
                maximumSupportedContext: 65_536
            ),
            65_536
        )
        XCTAssertEqual(
            LlamaContextPolicy.resolvedRequestedContext(
                trainingContext: 262_144,
                defaults: defaults,
                keys: keys,
                maximumSupportedContext: 131_072
            ),
            131_072
        )
        XCTAssertEqual(
            LlamaContextPolicy.resolvedRequestedContext(
                trainingContext: 262_144,
                defaults: defaults,
                keys: keys
            ),
            LlamaContextPolicy.defaultAutoCap
        )

        defaults.set(false, forKey: keys.autoContextLimitUsesMaximum)
        XCTAssertEqual(
            LlamaContextPolicy.resolvedRequestedContext(
                trainingContext: 262_144,
                defaults: defaults,
                keys: keys,
                maximumSupportedContext: 131_072
            ),
            65_536
        )
    }

    func testGGUFMetadataReadsTrainingContextLength() throws {
        let root = try makeTemporaryDirectory()
        let url = root.appendingPathComponent("metadata-only.gguf")
        try makeMinimalGGUF(contextLength: 32_768).write(to: url)

        XCTAssertEqual(GGUFMetadata.trainingContextLength(at: url), 32_768)
        XCTAssertEqual(GGUFMetadata.modelMetadata(at: url).architecture, "llama")
    }

    func testGGUFMetadataDetectsMTPAccelerationSupport() throws {
        let root = try makeTemporaryDirectory()
        let url = root.appendingPathComponent("metadata-only.gguf")
        try makeMinimalGGUF(contextLength: 32_768, nextNPredictLayers: 4).write(to: url)

        let metadata = GGUFMetadata.modelMetadata(at: url)
        XCTAssertEqual(metadata.architecture, "llama")
        XCTAssertEqual(metadata.trainingContextLength, 32_768)
        XCTAssertEqual(metadata.nextNPredictLayers, 4)
        XCTAssertTrue(metadata.supportsMTPAcceleration)
        XCTAssertTrue(GGUFMetadata.supportsMTPAcceleration(at: url))
    }

    func testGenerationOptionsResolverUsesSafeDefaultsUnlessCustom() {
        let suiteName = "CarbocationLocalLLMTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        XCTAssertEqual(GenerationOptionsResolver.configuredExtractionOptions(defaults: defaults), .extractionSafe)

        defaults.set("custom", forKey: "llama.optionsMode")
        defaults.set(0.2, forKey: "llama.temperature")
        defaults.set(0.7, forKey: "llama.topP")
        defaults.set(64, forKey: "llama.topK")
        defaults.set(0.05, forKey: "llama.minP")
        defaults.set(1.5, forKey: "llama.presencePenalty")
        defaults.set(1.0, forKey: "llama.repetitionPenalty")
        defaults.set("1234", forKey: "llama.seed")

        let options = GenerationOptionsResolver.configuredExtractionOptions(defaults: defaults)
        XCTAssertEqual(options.temperature, 0.2)
        XCTAssertEqual(options.topP, 0.7)
        XCTAssertEqual(options.topK, 64)
        XCTAssertEqual(options.minP, 0.05)
        XCTAssertEqual(options.presencePenalty, 1.5)
        XCTAssertEqual(options.repetitionPenalty, 1.0)
        XCTAssertEqual(options.seed, 1234)
    }

    func testSamplingDefaultsMergeOverridesOnlySpecifiedFields() {
        let base = LLMSamplingDefaults(
            temperature: 0,
            topP: 0.9,
            topK: 40,
            minP: 0.05,
            presencePenalty: 0,
            repetitionPenalty: 1.3,
            seed: 123
        )
        let override = LLMSamplingDefaults(
            temperature: 0.7,
            topK: 20,
            minP: 0,
            repetitionPenalty: 1.0,
            seed: 456
        )

        let merged = base.merged(with: override)

        XCTAssertEqual(merged.temperature, 0.7)
        XCTAssertEqual(merged.topP, 0.9)
        XCTAssertEqual(merged.topK, 20)
        XCTAssertEqual(merged.minP, 0)
        XCTAssertEqual(merged.presencePenalty, 0)
        XCTAssertEqual(merged.repetitionPenalty, 1.0)
        XCTAssertEqual(merged.seed, 456)
    }

    func testSamplingDefaultsResolverLayersGlobalCuratedAppAndRequestOptions() {
        let curated = CuratedModel(
            id: "lab",
            displayName: "Lab",
            subtitle: "",
            hfRepo: "example/lab",
            hfFilename: "lab-Q4_K_M.gguf",
            approxSizeBytes: 1,
            contextLength: 8_192,
            quantization: "Q4_K_M",
            recommendedRAMGB: 8,
            sha256: nil,
            samplingDefaults: LLMSamplingDefaults(temperature: 0.7, minP: 0)
        )
        let installed = InstalledModel(
            displayName: "Lab",
            filename: "lab-Q4_K_M.gguf",
            sizeBytes: 1,
            contextLength: 8_192,
            quantization: "Q4_K_M",
            source: .curated,
            hfRepo: curated.hfRepo,
            hfFilename: curated.hfFilename,
            sha256: nil
        )

        let resolved = LLMSamplingDefaultsResolver.resolvedOptions(
            globalDefaults: .extractionSafe,
            installedModel: installed,
            curatedModels: [curated],
            appOverrides: [curated.reference: LLMSamplingDefaults(topK: 20, presencePenalty: 1.5)],
            requestOptions: GenerationOptions(topP: 0.95, repetitionPenalty: 1.0, maxOutputTokens: 128)
        )

        XCTAssertEqual(resolved.temperature, 0.7)
        XCTAssertEqual(resolved.topP, 0.95)
        XCTAssertEqual(resolved.topK, 20)
        XCTAssertEqual(resolved.minP, 0)
        XCTAssertEqual(resolved.presencePenalty, 1.5)
        XCTAssertEqual(resolved.repetitionPenalty, 1.0)
        XCTAssertEqual(resolved.maxOutputTokens, 128)
    }

    func testSamplingDefaultsResolverDoesNotApplyCuratedDefaultsToCustomHFModel() {
        let curated = CuratedModel(
            id: "lab",
            displayName: "Lab",
            subtitle: "",
            hfRepo: "example/lab",
            hfFilename: "lab-Q4_K_M.gguf",
            approxSizeBytes: 1,
            contextLength: 8_192,
            quantization: "Q4_K_M",
            recommendedRAMGB: 8,
            sha256: nil,
            samplingDefaults: LLMSamplingDefaults(temperature: 0.7)
        )
        let installed = InstalledModel(
            displayName: "Lab",
            filename: "lab-Q4_K_M.gguf",
            sizeBytes: 1,
            contextLength: 8_192,
            quantization: "Q4_K_M",
            source: .customHF,
            hfRepo: curated.hfRepo,
            hfFilename: curated.hfFilename,
            sha256: nil
        )

        let resolved = LLMSamplingDefaultsResolver.resolvedDefaults(
            globalDefaults: .extractionSafe,
            installedModel: installed,
            curatedModels: [curated]
        )

        XCTAssertEqual(resolved, .extractionSafe)
    }

    func testGenerationOptionsDecodeLegacyPayloadWithNewDefaults() throws {
        let data = Data(#"{"temperature":0.1,"topP":0.8,"topK":32}"#.utf8)
        let options = try JSONDecoder().decode(GenerationOptions.self, from: data)

        XCTAssertEqual(options.temperature, 0.1)
        XCTAssertEqual(options.topP, 0.8)
        XCTAssertEqual(options.topK, 32)
        XCTAssertNil(options.minP)
        XCTAssertNil(options.presencePenalty)
        XCTAssertNil(options.repetitionPenalty)
        XCTAssertNil(options.seed)
        XCTAssertEqual(options.stopSequences, [])
        XCTAssertFalse(options.stopAtBalancedJSON)
        XCTAssertFalse(options.enableThinking)
        XCTAssertNil(options.thinkingBudgetTokens)
        XCTAssertEqual(options.thinkingBudgetMessage, "")
        XCTAssertEqual(options.streamPhaseConfiguration, .automatic)
    }

    func testGenerationOptionsOnlyEncodesEnableThinkingWhenTrue() throws {
        let data = try JSONEncoder().encode(GenerationOptions(enableThinking: true))
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        XCTAssertEqual(object?["enableThinking"] as? Bool, true)
        XCTAssertNil((try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(GenerationOptions())
        ) as? [String: Any])?["enableThinking"])
    }

    func testGenerationOptionsSamplingParametersRoundTrip() throws {
        let options = GenerationOptions(
            temperature: 0.7,
            topP: 0.8,
            topK: 20,
            minP: 0,
            presencePenalty: 1.5,
            repetitionPenalty: 1.0,
            seed: 1234
        )

        let data = try JSONEncoder().encode(options)
        let decoded = try JSONDecoder().decode(GenerationOptions.self, from: data)

        XCTAssertEqual(decoded.temperature, 0.7)
        XCTAssertEqual(decoded.topP, 0.8)
        XCTAssertEqual(decoded.topK, 20)
        XCTAssertEqual(decoded.minP, 0)
        XCTAssertEqual(decoded.presencePenalty, 1.5)
        XCTAssertEqual(decoded.repetitionPenalty, 1.0)
        XCTAssertEqual(decoded.seed, 1234)
    }

    func testGenerationOptionsThinkingBudgetRoundTrips() throws {
        let options = GenerationOptions(
            enableThinking: true,
            thinkingBudgetTokens: 0,
            thinkingBudgetMessage: "Budget reached."
        )

        let data = try JSONEncoder().encode(options)
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertEqual(object?["thinkingBudgetTokens"] as? Int, 0)
        XCTAssertEqual(object?["thinkingBudgetMessage"] as? String, "Budget reached.")

        let decoded = try JSONDecoder().decode(GenerationOptions.self, from: data)
        XCTAssertEqual(decoded.thinkingBudgetTokens, 0)
        XCTAssertEqual(decoded.thinkingBudgetMessage, "Budget reached.")
    }

    func testGenerationOptionsDefaultOmitsThinkingBudgetFields() throws {
        let object = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(GenerationOptions())
        ) as? [String: Any]

        XCTAssertNil(object?["thinkingBudgetTokens"])
        XCTAssertNil(object?["thinkingBudgetMessage"])
        XCTAssertNil(object?["streamPhaseConfiguration"])
    }

    func testGenerationOptionsStreamPhaseConfigurationRoundTrips() throws {
        let options = GenerationOptions(
            enableThinking: true,
            streamPhaseConfiguration: LLMStreamPhaseConfiguration(
                thinkingPairs: [OutputDelimiterPair(open: "<reason>", close: "</reason>")],
                finalMarkers: ["<final>"],
                startsInThinking: true
            )
        )

        let data = try JSONEncoder().encode(options)
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let configuration = object?["streamPhaseConfiguration"] as? [String: Any]

        XCTAssertNotNil(configuration?["thinkingPairs"])
        XCTAssertEqual(configuration?["finalMarkers"] as? [String], ["<final>"])
        XCTAssertEqual(configuration?["startsInThinking"] as? Bool, true)

        let decoded = try JSONDecoder().decode(GenerationOptions.self, from: data)
        XCTAssertEqual(decoded.streamPhaseConfiguration, options.streamPhaseConfiguration)
    }

    func testGenerationOptionsRejectsNegativeThinkingBudgetPayload() {
        let data = Data(#"{"thinkingBudgetTokens":-1}"#.utf8)

        XCTAssertThrowsError(try JSONDecoder().decode(GenerationOptions.self, from: data))
    }

    func testGenerationControlScopesThinkingTerminationRequestsToActiveGeneration() {
        let control = LLMGenerationControl()

        XCTAssertFalse(control.requestThinkingTermination())
        XCTAssertEqual(control.thinkingTerminationRequestCount, 0)

        let firstGeneration = control.beginGeneration()
        XCTAssertTrue(control.requestThinkingTermination(message: "Stop."))
        XCTAssertEqual(control.thinkingTerminationRequestCount, 1)
        XCTAssertEqual(
            control.takePendingThinkingTerminationRequest(for: firstGeneration),
            LLMThinkingTerminationRequest(
                generationID: firstGeneration,
                requestID: 1,
                message: "Stop."
            )
        )
        XCTAssertNil(control.takePendingThinkingTerminationRequest(for: firstGeneration))

        XCTAssertTrue(control.requestThinkingTermination())
        XCTAssertEqual(control.thinkingTerminationRequestCount, 2)
        control.finishGeneration(firstGeneration)
        XCTAssertEqual(control.thinkingTerminationRequestCount, 0)

        let secondGeneration = control.beginGeneration()
        XCTAssertNil(control.takePendingThinkingTerminationRequest(for: secondGeneration))
        XCTAssertEqual(control.thinkingTerminationRequestCount, 0)
        control.finishGeneration(secondGeneration)
    }

    func testGenerationPreflightComputesOutputBudget() {
        let uncapped = LLMGenerationPreflight(
            loadedContextSize: 4_096,
            modelTrainingContextSize: 262_144,
            promptTokens: 3_000,
            reservedOutputTokens: 1_024,
            requestedMaxOutputTokens: nil,
            usesExactTokenCounts: true,
            templateMode: .embedded
        )
        XCTAssertEqual(uncapped.loadedContextSize, 4_096)
        XCTAssertEqual(uncapped.modelTrainingContextSize, 262_144)
        XCTAssertEqual(uncapped.availableOutputTokens, 72)
        XCTAssertEqual(uncapped.effectiveMaxOutputTokens, 72)
        XCTAssertTrue(uncapped.canGenerate)

        let capped = LLMGenerationPreflight(
            loadedContextSize: 4_096,
            modelTrainingContextSize: 262_144,
            promptTokens: 3_000,
            reservedOutputTokens: 1_024,
            requestedMaxOutputTokens: 64,
            usesExactTokenCounts: true,
            templateMode: .embedded
        )
        XCTAssertEqual(capped.availableOutputTokens, 72)
        XCTAssertEqual(capped.effectiveMaxOutputTokens, 64)
        XCTAssertTrue(capped.canGenerate)

        let overRequested = LLMGenerationPreflight(
            loadedContextSize: 4_096,
            modelTrainingContextSize: 262_144,
            promptTokens: 3_000,
            reservedOutputTokens: 1_024,
            requestedMaxOutputTokens: 256,
            usesExactTokenCounts: true,
            templateMode: .embedded
        )
        XCTAssertEqual(overRequested.availableOutputTokens, 72)
        XCTAssertEqual(overRequested.effectiveMaxOutputTokens, 72)

        let zeroRequested = LLMGenerationPreflight(
            loadedContextSize: 4_096,
            modelTrainingContextSize: 262_144,
            promptTokens: 3_000,
            reservedOutputTokens: 1_024,
            requestedMaxOutputTokens: 0,
            usesExactTokenCounts: true,
            templateMode: .embedded
        )
        XCTAssertEqual(zeroRequested.requestedMaxOutputTokens, 0)
        XCTAssertEqual(zeroRequested.effectiveMaxOutputTokens, 72)

        let negativeRequested = LLMGenerationPreflight(
            loadedContextSize: 4_096,
            modelTrainingContextSize: 262_144,
            promptTokens: 3_000,
            reservedOutputTokens: 1_024,
            requestedMaxOutputTokens: -1,
            usesExactTokenCounts: true,
            templateMode: .embedded
        )
        XCTAssertEqual(negativeRequested.effectiveMaxOutputTokens, 72)

        let exactFit = LLMGenerationPreflight(
            loadedContextSize: 4_096,
            modelTrainingContextSize: 262_144,
            promptTokens: 3_072,
            reservedOutputTokens: 1_024,
            requestedMaxOutputTokens: nil,
            usesExactTokenCounts: true,
            templateMode: .embedded
        )
        XCTAssertEqual(exactFit.availableOutputTokens, 0)
        XCTAssertEqual(exactFit.effectiveMaxOutputTokens, 0)
        XCTAssertFalse(exactFit.canGenerate)

        let overflow = LLMGenerationPreflight(
            loadedContextSize: 4_096,
            modelTrainingContextSize: 262_144,
            promptTokens: 4_097,
            reservedOutputTokens: 0,
            requestedMaxOutputTokens: nil,
            usesExactTokenCounts: true,
            templateMode: .embedded
        )
        XCTAssertEqual(overflow.availableOutputTokens, 0)
        XCTAssertFalse(overflow.canGenerate)
    }

    @MainActor
    func testModelLibraryImportsSyncsAndDeletesGGUF() async throws {
        let root = try makeTemporaryDirectory()
        let source = root.appendingPathComponent("source-Q4_K_M.gguf")
        try Data("fake gguf".utf8).write(to: source)
        let modelsRoot = root.appendingPathComponent("Models", isDirectory: true)

        let library = ModelLibrary(root: modelsRoot, searchConfiguration: .managedOnly) { url in
            url.lastPathComponent == "source-Q4_K_M.gguf" ? 32_768 : nil
        }

        let model = try await library.importFile(at: source, displayName: "Test Model")

        XCTAssertEqual(library.models.count, 1)
        XCTAssertEqual(model.displayName, "Test Model")
        XCTAssertEqual(model.quantization, "Q4_K_M")
        XCTAssertEqual(model.contextLength, 32_768)
        XCTAssertTrue(FileManager.default.fileExists(atPath: model.weightsURL(in: modelsRoot).path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: model.metadataURL(in: modelsRoot).path))

        try await library.syncContextLength(65_536, for: model.id)
        XCTAssertEqual(library.model(id: model.id)?.contextLength, 65_536)

        try await library.delete(id: model.id)
        XCTAssertTrue(library.models.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: model.directory(in: modelsRoot).path))
    }

    @MainActor
    func testModelLibraryImportIncludesCompanionMMProj() async throws {
        let root = try makeTemporaryDirectory()
        let source = root.appendingPathComponent("model.q4_k_m.gguf")
        let autoMMProj = root.appendingPathComponent("model-mmproj.q8_0.gguf")
        let explicitMMProj = root.appendingPathComponent("model-mmproj.bf16.gguf")
        let primaryData = makeMinimalGGUF(contextLength: 262_144)
        let autoMMProjData = Data("q8 mmproj".utf8)
        let explicitMMProjData = Data("bf16 mmproj".utf8)
        try primaryData.write(to: source)
        try autoMMProjData.write(to: autoMMProj)
        try explicitMMProjData.write(to: explicitMMProj)
        let modelsRoot = root.appendingPathComponent("Models", isDirectory: true)
        let library = ModelLibrary(root: modelsRoot, searchConfiguration: .managedOnly)

        let model = try await library.importFiles(at: [source, explicitMMProj])

        XCTAssertEqual(model.artifacts.map(\.role), [.primaryModel, .mmproj])
        XCTAssertEqual(
            model.artifacts.map(\.relativePath),
            [
                "model.q4_k_m.gguf",
                "model-mmproj.bf16.gguf"
            ]
        )
        XCTAssertEqual(model.sizeBytes, Int64(primaryData.count + explicitMMProjData.count))
        XCTAssertTrue(FileManager.default.fileExists(atPath: model.weightsURL(in: modelsRoot).path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: model.mmprojURL(in: modelsRoot)?.path ?? ""))
    }

    @MainActor
    func testModelLibraryDiscoversHuggingFaceCacheSnapshotsReadOnly() async throws {
        let root = try makeTemporaryDirectory()
        let hubRoot = root.appendingPathComponent("hub", isDirectory: true)
        let modelsRoot = root.appendingPathComponent("Models", isDirectory: true)
        let commit = Self.mockCommit
        let primaryHash = String(repeating: "b", count: 64)
        let mmprojHash = String(repeating: "c", count: 64)
        let primaryData = makeMinimalGGUF(contextLength: 32_768)
        let mmprojData = Data("mmproj".utf8)

        let snapshot = try makeHuggingFaceCacheRepo(
            hubRoot: hubRoot,
            repoFolder: "models--owner--repo",
            commit: commit,
            files: [
                "Qwen3.6-27B-Q8_0.gguf": (primaryHash, primaryData),
                "mmproj-BF16.gguf": (mmprojHash, mmprojData)
            ]
        )

        let library = ModelLibrary(
            root: modelsRoot,
            searchConfiguration: ModelLibrarySearchConfiguration(
                externalHuggingFaceHubCacheDirectories: [hubRoot]
            )
        )

        await library.refresh()

        XCTAssertEqual(library.models.count, 1)
        let model = try XCTUnwrap(library.models.first)
        let originalID = model.id
        XCTAssertTrue(model.isReadOnly)
        XCTAssertEqual(model.hfRepo, "owner/repo")
        XCTAssertEqual(model.hfFilename, "Qwen3.6-27B-Q8_0.gguf")
        XCTAssertEqual(model.sha256, primaryHash)
        XCTAssertEqual(model.sizeBytes, Int64(primaryData.count + mmprojData.count))
        XCTAssertEqual(model.contextLength, 32_768)
        XCTAssertEqual(model.quantization, "Q8_0")
        XCTAssertEqual(
            model.weightsURL(in: modelsRoot).standardizedFileURL.path,
            snapshot.appendingPathComponent("Qwen3.6-27B-Q8_0.gguf").standardizedFileURL.path
        )
        XCTAssertEqual(model.artifacts.map(\.role), [.primaryModel, .mmproj])

        await library.refresh()
        XCTAssertEqual(library.models.first?.id, originalID)

        do {
            try await library.delete(id: originalID)
            XCTFail("Expected deleting a Hugging Face cache model to fail.")
        } catch ModelLibraryError.readOnlyModel {
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: snapshot.appendingPathComponent("Qwen3.6-27B-Q8_0.gguf").path
            ))
        }

        do {
            try await library.syncContextLength(65_536, for: originalID)
            XCTFail("Expected syncing read-only model metadata to fail.")
        } catch ModelLibraryError.readOnlyModel {
        }

        do {
            try await library.writeMetadata(model)
            XCTFail("Expected writing read-only model metadata to fail.")
        } catch ModelLibraryError.readOnlyModel {
        }
    }

    @MainActor
    func testModelLibraryGroupsHuggingFaceCacheSplitModelsAndMMProj() async throws {
        let root = try makeTemporaryDirectory()
        let hubRoot = root.appendingPathComponent("hub", isDirectory: true)
        let modelsRoot = root.appendingPathComponent("Models", isDirectory: true)
        let primaryData = makeMinimalGGUF(contextLength: 65_536)
        let splitData = Data("split".utf8)
        let mmprojData = Data("mmproj".utf8)

        _ = try makeHuggingFaceCacheRepo(
            hubRoot: hubRoot,
            repoFolder: "models--owner--split-repo",
            commit: Self.mockCommit,
            files: [
                "nested/model-Q4_K_M-00001-of-00002.gguf": (String(repeating: "d", count: 64), primaryData),
                "nested/model-Q4_K_M-00002-of-00002.gguf": (String(repeating: "e", count: 64), splitData),
                "nested/mmproj-model-Q4_K_M.gguf": (String(repeating: "f", count: 64), mmprojData)
            ]
        )

        let library = ModelLibrary(
            root: modelsRoot,
            searchConfiguration: ModelLibrarySearchConfiguration(
                externalHuggingFaceHubCacheDirectories: [hubRoot]
            )
        )

        await library.refresh()

        XCTAssertEqual(library.models.count, 1)
        let model = try XCTUnwrap(library.models.first)
        XCTAssertEqual(model.displayName, "model-Q4_K_M")
        XCTAssertEqual(model.filename, "nested/model-Q4_K_M-00001-of-00002.gguf")
        XCTAssertEqual(model.contextLength, 65_536)
        XCTAssertEqual(model.sizeBytes, Int64(primaryData.count + splitData.count + mmprojData.count))
        XCTAssertEqual(model.artifacts.map(\.role), [.primaryModel, .splitModel, .mmproj])
        XCTAssertEqual(
            model.artifacts.map(\.relativePath),
            [
                "nested/model-Q4_K_M-00001-of-00002.gguf",
                "nested/model-Q4_K_M-00002-of-00002.gguf",
                "nested/mmproj-model-Q4_K_M.gguf"
            ]
        )
    }

    @MainActor
    func testModelLibraryPrefersManagedModelOverMatchingHuggingFaceCacheEntry() async throws {
        let root = try makeTemporaryDirectory()
        let hubRoot = root.appendingPathComponent("hub", isDirectory: true)
        let modelsRoot = root.appendingPathComponent("Models", isDirectory: true)
        let source = root.appendingPathComponent("managed-Q4_K_M.gguf")
        let sha = String(repeating: "a", count: 64)
        let payload = makeMinimalGGUF(contextLength: 16_384)
        try payload.write(to: source)

        _ = try makeHuggingFaceCacheRepo(
            hubRoot: hubRoot,
            repoFolder: "models--owner--dupe",
            commit: Self.mockCommit,
            files: [
                "managed-Q4_K_M.gguf": (sha, payload)
            ]
        )

        let library = ModelLibrary(
            root: modelsRoot,
            searchConfiguration: ModelLibrarySearchConfiguration(
                externalHuggingFaceHubCacheDirectories: [hubRoot]
            )
        )

        let managed = try await library.add(
            weightsAt: source,
            displayName: "Managed",
            filename: "managed-Q4_K_M.gguf",
            sizeBytes: Int64(payload.count),
            source: .customHF,
            hfRepo: "owner/dupe",
            hfFilename: "managed-Q4_K_M.gguf",
            sha256: sha,
            contextLength: 16_384
        )

        await library.refresh()

        XCTAssertEqual(library.models.count, 1)
        XCTAssertEqual(library.models.first?.id, managed.id)
        XCTAssertFalse(try XCTUnwrap(library.models.first).isReadOnly)
    }

    @MainActor
    func testModelLibraryTrustsProvidedContextLengthWithoutProbe() async throws {
        let root = try makeTemporaryDirectory()
        let source = root.appendingPathComponent("download-Q4_K_M.gguf")
        try Data("fake gguf".utf8).write(to: source)
        let modelsRoot = root.appendingPathComponent("Models", isDirectory: true)
        let recorder = ThreadProbeRecorder()

        let library = ModelLibrary(root: modelsRoot, searchConfiguration: .managedOnly) { _ in
            recorder.record(true)
            return 16_384
        }

        let model = try await library.add(
            weightsAt: source,
            displayName: "Known Context Model",
            filename: "download-Q4_K_M.gguf",
            sizeBytes: 9,
            source: .curated,
            contextLength: 32_768
        )

        XCTAssertEqual(model.contextLength, 32_768)
        XCTAssertNil(recorder.value)
    }

    @MainActor
    func testModelLibrarySynthesizesOrphanMetadata() async throws {
        let root = try makeTemporaryDirectory()
        let modelID = UUID()
        let directory = root.appendingPathComponent(modelID.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("fake".utf8).write(to: directory.appendingPathComponent("Orphan-Q5_K_M.gguf"))

        let library = ModelLibrary(root: root, searchConfiguration: .managedOnly)
        await library.refresh()

        XCTAssertEqual(library.models.count, 1)
        XCTAssertEqual(library.models[0].id, modelID)
        XCTAssertEqual(library.models[0].displayName, "Orphan-Q5_K_M")
        XCTAssertEqual(library.models[0].quantization, "Q5_K_M")
    }

    @MainActor
    func testModelLibraryOrdersInstalledModelsBySizeThenNameThenID() async throws {
        let root = try makeTemporaryDirectory()
        let sourcesRoot = root.appendingPathComponent("Sources", isDirectory: true)
        let modelsRoot = root.appendingPathComponent("Models", isDirectory: true)
        try FileManager.default.createDirectory(at: sourcesRoot, withIntermediateDirectories: true)
        let library = ModelLibrary(root: modelsRoot, searchConfiguration: .managedOnly)

        let largest = try await addFakeModel(
            to: library,
            sourcesRoot: sourcesRoot,
            displayName: "Zulu",
            filename: "zulu-Q8_0.gguf",
            sizeBytes: 300
        )
        let nameTieSecond = try await addFakeModel(
            to: library,
            sourcesRoot: sourcesRoot,
            displayName: "bravo",
            filename: "bravo-Q4_K_M.gguf",
            sizeBytes: 100
        )
        let nameTieFirst = try await addFakeModel(
            to: library,
            sourcesRoot: sourcesRoot,
            displayName: "Alpha",
            filename: "alpha-Q4_K_M.gguf",
            sizeBytes: 100
        )
        let idTieFirst = try await addFakeModel(
            to: library,
            sourcesRoot: sourcesRoot,
            displayName: "Same",
            filename: "same-a-Q5_K_M.gguf",
            sizeBytes: 200
        )
        let idTieSecond = try await addFakeModel(
            to: library,
            sourcesRoot: sourcesRoot,
            displayName: "Same",
            filename: "same-b-Q5_K_M.gguf",
            sizeBytes: 200
        )

        let idTieOrder = [idTieFirst, idTieSecond].sorted {
            $0.id.uuidString < $1.id.uuidString
        }
        XCTAssertEqual(
            library.models.map(\.id),
            [nameTieFirst.id, nameTieSecond.id] + idTieOrder.map(\.id) + [largest.id]
        )
        XCTAssertEqual(library.models.map(\.sizeBytes), [100, 100, 200, 200, 300])
    }

    @MainActor
    func testModelLibraryResolveInstalledModelRefreshesOnDemand() async throws {
        let root = try makeTemporaryDirectory()
        let modelID = UUID()
        let directory = root.appendingPathComponent(modelID.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("fake".utf8).write(to: directory.appendingPathComponent("Resolvable-Q4_K_M.gguf"))

        let library = ModelLibrary(root: root, searchConfiguration: .managedOnly)

        XCTAssertNil(library.model(id: modelID))
        let cachedOnly = await library.resolveInstalledModel(id: modelID, refreshing: false)
        XCTAssertNil(cachedOnly)

        let resolved = await library.resolveInstalledModel(id: modelID)

        XCTAssertEqual(resolved?.id, modelID)
        XCTAssertEqual(resolved?.displayName, "Resolvable-Q4_K_M")
        XCTAssertEqual(library.model(id: modelID)?.id, modelID)
    }

    @MainActor
    func testModelLibraryContextProbeRunsOffMainThread() async throws {
        let root = try makeTemporaryDirectory()
        let source = root.appendingPathComponent("probe-Q4_K_M.gguf")
        try Data("fake gguf".utf8).write(to: source)
        let modelsRoot = root.appendingPathComponent("Models", isDirectory: true)
        let recorder = ThreadProbeRecorder()

        let library = ModelLibrary(root: modelsRoot, searchConfiguration: .managedOnly) { _ in
            recorder.record(Thread.isMainThread)
            return 16_384
        }

        let model = try await library.importFile(at: source, displayName: "Probe Model")

        XCTAssertEqual(model.contextLength, 16_384)
        XCTAssertEqual(recorder.value, false)
    }

    @MainActor
    func testModelLibraryRefreshIgnoresHiddenStagingDirectories() async throws {
        let root = try makeTemporaryDirectory()
        let modelID = UUID()
        let stagingDirectory = root
            .appendingPathComponent(".staging", isDirectory: true)
            .appendingPathComponent("\(modelID.uuidString)-staged", isDirectory: true)
        try FileManager.default.createDirectory(at: stagingDirectory, withIntermediateDirectories: true)
        try Data("fake".utf8).write(to: stagingDirectory.appendingPathComponent("Staged-Q5_K_M.gguf"))

        let library = ModelLibrary(root: root, searchConfiguration: .managedOnly)
        await library.refresh()

        XCTAssertTrue(library.models.isEmpty)
    }

    @MainActor
    func testModelLibraryFailedInstallLeavesNoVisibleModel() async throws {
        let root = try makeTemporaryDirectory()
        let source = root.appendingPathComponent("source-Q4_K_M.gguf")
        let modelsRoot = root.appendingPathComponent("Models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsRoot, withIntermediateDirectories: true)
        try Data("fake gguf".utf8).write(to: source)
        try Data("not a directory".utf8).write(to: modelsRoot.appendingPathComponent(".staging"))

        let library = ModelLibrary(root: modelsRoot, searchConfiguration: .managedOnly)

        do {
            _ = try await library.importFile(at: source, displayName: "Broken")
            XCTFail("Expected import to fail when staging cannot be created.")
        } catch {
            await library.refresh()
            XCTAssertTrue(library.models.isEmpty)
            XCTAssertFalse(
                try FileManager.default.contentsOfDirectory(
                    at: modelsRoot,
                    includingPropertiesForKeys: nil,
                    options: [.skipsHiddenFiles]
                ).contains { $0.pathExtension == "gguf" || UUID(uuidString: $0.lastPathComponent) != nil }
            )
        }
    }

    func testHuggingFaceURLParsesExpectedForms() {
        let resolve = HuggingFaceURL.parse("https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf?download=true")
        XCTAssertEqual(resolve?.repo, "bartowski/Qwen2.5-7B-Instruct-GGUF")
        XCTAssertEqual(resolve?.filename, "Qwen2.5-7B-Instruct-Q4_K_M.gguf")

        let nested = HuggingFaceURL.parse("https://huggingface.co/bartowski/model/resolve/main/quantized/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
        XCTAssertEqual(nested?.repo, "bartowski/model")
        XCTAssertEqual(nested?.filename, "quantized/Qwen2.5-7B-Instruct-Q4_K_M.gguf")

        let blob = HuggingFaceURL.parse("https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/blob/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
        XCTAssertEqual(blob?.repo, "bartowski/Qwen2.5-7B-Instruct-GGUF")
        XCTAssertEqual(blob?.filename, "Qwen2.5-7B-Instruct-Q4_K_M.gguf")

        let compact = HuggingFaceURL.parse("bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
        XCTAssertEqual(compact?.repo, "bartowski/Qwen2.5-7B-Instruct-GGUF")
        XCTAssertEqual(compact?.filename, "Qwen2.5-7B-Instruct-Q4_K_M.gguf")

        XCTAssertNil(HuggingFaceURL.parse("https://huggingface.co/bartowski/model/blob/main/README.md"))
    }

    func testHuggingFaceModelReferenceParsesLlamaCppForms() {
        let repoQuant = HuggingFaceModelReference.parse("unsloth/gemma-4-31B-it-GGUF:UD-Q8_K_XL")
        XCTAssertEqual(repoQuant?.repo, "unsloth/gemma-4-31B-it-GGUF")
        XCTAssertEqual(repoQuant?.quantization, "UD-Q8_K_XL")
        XCTAssertNil(repoQuant?.file)

        let repoOnly = HuggingFaceModelReference.parse("ggml-org/gemma-3-1b-it-GGUF")
        XCTAssertEqual(repoOnly?.repo, "ggml-org/gemma-3-1b-it-GGUF")
        XCTAssertNil(repoOnly?.quantization)

        let exact = HuggingFaceModelReference.parse("bartowski/model/nested/foo-Q4_K_M.gguf")
        XCTAssertEqual(exact?.repo, "bartowski/model")
        XCTAssertEqual(exact?.file, "nested/foo-Q4_K_M.gguf")

        let url = HuggingFaceModelReference.parse("https://huggingface.co/bartowski/model/blob/dev/nested/foo-Q4_K_M.gguf")
        XCTAssertEqual(url?.repo, "bartowski/model")
        XCTAssertEqual(url?.revision, "dev")
        XCTAssertEqual(url?.file, "nested/foo-Q4_K_M.gguf")

        XCTAssertNil(HuggingFaceModelReference.parse("owner/repo/not-a-gguf.txt"))
        XCTAssertNil(HuggingFaceModelReference.parse("owner/repo/too/many/parts"))
    }

    func testHuggingFaceResolverChoosesDefaultQuantAndSendsAuthorization() async throws {
        let client = MockHuggingFaceHTTPClient(
            refsJSON: refsJSON(commit: Self.mockCommit),
            treeJSON: #"""
            [
              {"type":"file","path":"README.md","size":10},
              {"type":"file","path":"model-Q8_0.gguf","size":800},
              {"type":"file","path":"model-Q4_K_M.gguf","lfs":{"oid":"1111111111111111111111111111111111111111","size":400}},
              {"type":"file","path":"mmproj-model-Q4_K_M.gguf","size":50}
            ]
            """#
        )
        let endpoint = URL(string: "https://huggingface.test")!
        let resolver = HuggingFaceModelResolver(endpoint: endpoint, httpClient: client)
        let reference = HuggingFaceModelReference(repo: "org/model", endpoint: endpoint)

        let resolution = try await resolver.resolve(reference, token: "hf_testtoken")

        XCTAssertEqual(resolution.commit, Self.mockCommit)
        XCTAssertEqual(resolution.primaryArtifact.path, "model-Q4_K_M.gguf")
        XCTAssertEqual(resolution.quantization, "Q4_K_M")
        XCTAssertEqual(resolution.splitCount, 1)
        XCTAssertEqual(resolution.mmprojArtifact?.path, "mmproj-model-Q4_K_M.gguf")
        XCTAssertEqual(resolution.totalSizeBytes, 450)
        XCTAssertTrue(client.requests.contains {
            $0.value(forHTTPHeaderField: "Authorization") == "Bearer hf_testtoken"
        })
    }

    func testHuggingFaceResolverPairsRootModelWithNearestMMProjQuant() async throws {
        let client = MockHuggingFaceHTTPClient(
            refsJSON: refsJSON(commit: Self.mockCommit),
            treeJSON: #"""
            [
              {"type":"file","path":"README.md","size":10},
              {"type":"file","path":"model-mmproj.bf16.gguf","size":1190000000},
              {"type":"file","path":"model-mmproj.f16.gguf","size":1190000000},
              {"type":"file","path":"model-mmproj.f32.gguf","size":2290000000},
              {"type":"file","path":"model-mmproj.q8_0.gguf","size":806000000},
              {"type":"file","path":"model.q4_k_m.gguf","size":16900000000},
              {"type":"file","path":"model.q6_k.gguf","size":23200000000}
            ]
            """#
        )
        let endpoint = URL(string: "https://huggingface.test")!
        let resolver = HuggingFaceModelResolver(endpoint: endpoint, httpClient: client)
        let reference = HuggingFaceModelReference(
            repo: "org/model",
            endpoint: endpoint
        )

        let resolution = try await resolver.resolve(reference)

        XCTAssertEqual(resolution.primaryArtifact.path, "model.q4_k_m.gguf")
        XCTAssertEqual(resolution.mmprojArtifact?.path, "model-mmproj.q8_0.gguf")
        XCTAssertEqual(resolution.totalSizeBytes, 17_706_000_000)
    }

    func testHuggingFaceResolverChoosesRequestedSplitQuant() async throws {
        let client = MockHuggingFaceHTTPClient(
            refsJSON: refsJSON(commit: Self.mockCommit),
            treeJSON: #"""
            [
              {"type":"file","path":"model-Q4_K_M.gguf","size":400},
              {"type":"file","path":"nested/model-UD-Q8_K_XL-00002-of-00003.gguf","size":200},
              {"type":"file","path":"nested/model-UD-Q8_K_XL-00001-of-00003.gguf","size":200},
              {"type":"file","path":"nested/model-UD-Q8_K_XL-00003-of-00003.gguf","size":200},
              {"type":"file","path":"nested/mmproj-model-UD-Q8_K_XL.gguf","size":25},
              {"type":"file","path":"nested/model-UD-Q8_K_XL-imatrix.dat","size":1}
            ]
            """#
        )
        let endpoint = URL(string: "https://huggingface.test")!
        let resolver = HuggingFaceModelResolver(endpoint: endpoint, httpClient: client)
        let reference = HuggingFaceModelReference(
            repo: "org/model",
            quantization: "ud-q8_k_xl",
            endpoint: endpoint
        )

        let resolution = try await resolver.resolve(reference)

        XCTAssertEqual(resolution.primaryArtifact.path, "nested/model-UD-Q8_K_XL-00001-of-00003.gguf")
        XCTAssertEqual(resolution.quantization, "UD-Q8_K_XL")
        XCTAssertEqual(resolution.splitCount, 3)
        XCTAssertEqual(
            resolution.artifacts.filter { $0.role == .splitModel }.map(\.path),
            [
                "nested/model-UD-Q8_K_XL-00002-of-00003.gguf",
                "nested/model-UD-Q8_K_XL-00003-of-00003.gguf"
            ]
        )
        XCTAssertEqual(resolution.mmprojArtifact?.path, "nested/mmproj-model-UD-Q8_K_XL.gguf")
        XCTAssertEqual(resolution.totalSizeBytes, 625)
    }

    func testHuggingFaceResolverReportsMissingQuantAndNoGGUF() async throws {
        let endpoint = URL(string: "https://huggingface.test")!
        let missingQuant = HuggingFaceModelResolver(
            endpoint: endpoint,
            httpClient: MockHuggingFaceHTTPClient(
                refsJSON: refsJSON(commit: Self.mockCommit),
                treeJSON: #"[{"type":"file","path":"model-Q4_K_M.gguf","size":400}]"#
            )
        )

        do {
            _ = try await missingQuant.resolve(HuggingFaceModelReference(
                repo: "org/model",
                quantization: "Q8_0",
                endpoint: endpoint
            ))
            XCTFail("Expected missing quantization error.")
        } catch HuggingFaceModelResolverError.quantizationNotFound(let quantization) {
            XCTAssertEqual(quantization, "Q8_0")
        }

        let noGGUF = HuggingFaceModelResolver(
            endpoint: endpoint,
            httpClient: MockHuggingFaceHTTPClient(
                refsJSON: refsJSON(commit: Self.mockCommit),
                treeJSON: #"[{"type":"file","path":"README.md","size":1}]"#
            )
        )

        do {
            _ = try await noGGUF.resolve(HuggingFaceModelReference(repo: "org/model", endpoint: endpoint))
            XCTFail("Expected no GGUF error.")
        } catch HuggingFaceModelResolverError.noGGUFFiles(let repo) {
            XCTAssertEqual(repo, "org/model")
        }
    }

    @MainActor
    func testModelLibraryInstallsMultipleArtifactsAndMetadataStaysCompatible() async throws {
        let root = try makeTemporaryDirectory()
        let primary = root.appendingPathComponent("primary.gguf")
        let split = root.appendingPathComponent("split.gguf")
        let mmproj = root.appendingPathComponent("mmproj.gguf")
        try Data("primary".utf8).write(to: primary)
        try Data("split".utf8).write(to: split)
        try Data("mmproj".utf8).write(to: mmproj)

        let modelsRoot = root.appendingPathComponent("Models", isDirectory: true)
        let library = ModelLibrary(root: modelsRoot, searchConfiguration: .managedOnly)
        let model = try await library.add(
            artifacts: [
                ModelLibraryInstallArtifact(
                    sourceURL: primary,
                    role: .primaryModel,
                    relativePath: "nested/model-Q4_K_M-00001-of-00002.gguf",
                    sizeBytes: 7,
                    sha256: "primary-sha"
                ),
                ModelLibraryInstallArtifact(
                    sourceURL: split,
                    role: .splitModel,
                    relativePath: "nested/model-Q4_K_M-00002-of-00002.gguf",
                    sizeBytes: 5
                ),
                ModelLibraryInstallArtifact(
                    sourceURL: mmproj,
                    role: .mmproj,
                    relativePath: "nested/mmproj-model-Q4_K_M.gguf",
                    sizeBytes: 6
                )
            ],
            displayName: "Split Model",
            source: .customHF,
            hfRepo: "org/model",
            hfFilename: "nested/model-Q4_K_M-00001-of-00002.gguf",
            sha256: "primary-sha",
            quantization: "Q4_K_M"
        )

        XCTAssertEqual(model.sizeBytes, 18)
        XCTAssertEqual(model.filename, "nested/model-Q4_K_M-00001-of-00002.gguf")
        XCTAssertEqual(model.artifacts.map(\.role), [.primaryModel, .splitModel, .mmproj])
        XCTAssertTrue(FileManager.default.fileExists(atPath: model.weightsURL(in: modelsRoot).path))
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: model.directory(in: modelsRoot)
                .appendingPathComponent("nested/mmproj-model-Q4_K_M.gguf")
                .path
        ))

        let oldMetadata = #"""
        {
          "id": "00000000-0000-0000-0000-000000000001",
          "displayName": "Old",
          "filename": "old-Q4_K_M.gguf",
          "sizeBytes": 123,
          "contextLength": 0,
          "quantization": "Q4_K_M",
          "source": "imported",
          "sha256": "abc",
          "installedAt": "2026-01-01T00:00:00Z"
        }
        """#
        let decoded = try LocalLLMJSON.makeDecoder().decode(InstalledModel.self, from: Data(oldMetadata.utf8))
        XCTAssertEqual(decoded.artifacts.count, 1)
        XCTAssertEqual(decoded.artifacts[0].role, .primaryModel)
        XCTAssertEqual(decoded.artifacts[0].relativePath, "old-Q4_K_M.gguf")
        XCTAssertEqual(decoded.artifacts[0].sha256, "abc")
    }

    func testModelDownloaderAuthorizationHeaderDoesNotLeakIntoSidecars() throws {
        var request = URLRequest(url: URL(string: "https://huggingface.co/org/model/resolve/main/model.gguf")!)
        ModelDownloader.applyStandardHeaders(to: &request, bearerToken: " hf_secret ")

        XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer hf_secret")
        XCTAssertNil(request.value(forHTTPHeaderField: "hf_secret"))
    }

    func testChunkPlanComputesPendingRangesAndCompletedBytes() {
        let plan = ChunkPlan(totalBytes: 25, chunkSize: 10, doneChunks: [0, 2])

        XCTAssertEqual(plan.chunkCount, 3)
        XCTAssertEqual(plan.completedBytes(), 15)
        XCTAssertEqual(plan.pendingRanges(), [
            ChunkRange(index: 1, start: 10, end: 19)
        ])
    }

    func testModelDownloadConfigurationNormalizesUnsafeValues() {
        let defaults = ModelDownloadConfiguration.default
        XCTAssertEqual(defaults.parallelConnections, 12)
        XCTAssertEqual(defaults.chunkSize, 16 * 1_024 * 1_024)

        let normalized = ModelDownloadConfiguration(
            parallelConnections: 100,
            chunkSize: 128,
            requestTimeout: 1
        )
        XCTAssertEqual(
            normalized.parallelConnections,
            ModelDownloadConfiguration.maximumParallelConnections
        )
        XCTAssertEqual(normalized.chunkSize, 1_024 * 1_024)
        XCTAssertEqual(normalized.requestTimeout, 30)
    }

    func testListPartialsUsesDoneChunksForPreallocatedChunkedFiles() throws {
        let root = try makeTemporaryDirectory()
        let partialsRoot = try ModelDownloader.partialsDirectory(in: root)
        let stem = "cllm-partial-abcdef123456"
        let sidecarURL = partialsRoot.appendingPathComponent("\(stem).json")
        let partialURL = partialsRoot.appendingPathComponent("\(stem).gguf")

        FileManager.default.createFile(atPath: partialURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: partialURL)
        try handle.truncate(atOffset: 100)
        try handle.close()

        let sidecar = #"""
        {
          "url": "https://huggingface.co/bartowski/model/resolve/main/nested/foo-Q4_K_M.gguf",
          "totalBytes": 100,
          "displayName": "Foo",
          "schemaVersion": 2,
          "chunkSize": 10,
          "doneChunks": [0, 2, 3]
        }
        """#
        try Data(sidecar.utf8).write(to: sidecarURL)

        let partials = ModelDownloader.listPartials(in: root)
        XCTAssertEqual(partials.count, 1)
        XCTAssertEqual(partials[0].displayName, "Foo")
        XCTAssertEqual(partials[0].hfRepo, "bartowski/model")
        XCTAssertEqual(partials[0].hfFilename, "nested/foo-Q4_K_M.gguf")
        XCTAssertEqual(partials[0].totalBytes, 100)
        XCTAssertEqual(partials[0].bytesOnDisk, 30)
        XCTAssertEqual(partials[0].fractionComplete, 0.3, accuracy: 0.000_001)
    }

    func testListPartialsUsesFileSizeForSingleStreamPartials() throws {
        let root = try makeTemporaryDirectory()
        let partialsRoot = try ModelDownloader.partialsDirectory(in: root)
        let stem = "cllm-partial-fedcba654321"
        let sidecarURL = partialsRoot.appendingPathComponent("\(stem).json")
        let partialURL = partialsRoot.appendingPathComponent("\(stem).gguf")

        try Data("partial data".utf8).write(to: partialURL)
        let sidecar = #"""
        {
          "url": "https://huggingface.co/bartowski/model/resolve/main/foo-Q4_K_M.gguf",
          "totalBytes": 100,
          "displayName": "Foo",
          "schemaVersion": 1
        }
        """#
        try Data(sidecar.utf8).write(to: sidecarURL)

        let partials = ModelDownloader.listPartials(in: root)
        XCTAssertEqual(partials.count, 1)
        XCTAssertEqual(partials[0].bytesOnDisk, 12)
    }

    func testDeletePartialRemovesSidecarAndPartialFile() throws {
        let root = try makeTemporaryDirectory()
        let partialsRoot = try ModelDownloader.partialsDirectory(in: root)
        let stem = "cllm-partial-001122334455"
        let sidecarURL = partialsRoot.appendingPathComponent("\(stem).json")
        let partialURL = partialsRoot.appendingPathComponent("\(stem).gguf")

        try Data("partial data".utf8).write(to: partialURL)
        let sidecar = #"""
        {
          "url": "https://huggingface.co/bartowski/model/resolve/main/foo-Q4_K_M.gguf",
          "totalBytes": 100,
          "displayName": "Foo",
          "schemaVersion": 1
        }
        """#
        try Data(sidecar.utf8).write(to: sidecarURL)

        let partial = try XCTUnwrap(ModelDownloader.listPartials(in: root).first)
        ModelDownloader.deletePartial(partial)

        XCTAssertFalse(FileManager.default.fileExists(atPath: partialURL.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: sidecarURL.path))
    }

    func testSanitizerAndJSONSalvageStripReasoningAndFences() throws {
        struct Payload: Decodable, Equatable {
            let title: String
        }

        let raw = """
        <think>private reasoning</think>
        ```json
        {"title":"Example"}
        ```
        """

        XCTAssertEqual(try JSONSalvage.decode(Payload.self, from: raw), Payload(title: "Example"))
    }

    func testOutputProfileDerivationGemma4() throws {
        let template = try String(contentsOf: Self.gemma4TemplateURL, encoding: .utf8)
        let profile = OutputSanitizationProfile.derived(fromChatTemplate: template)

        XCTAssertTrue(profile.thinkingPairs.contains(OutputDelimiterPair(
            open: "<|channel>thought",
            close: "<channel|>"
        )))
        XCTAssertTrue(profile.extraStopStrings.contains("<turn|>"))
        XCTAssertTrue(profile.extraStopStrings.contains("<|turn>"))
    }

    func testOutputProfileDerivationChatML() {
        let profile = OutputSanitizationProfile.derived(fromChatTemplate: """
        {% for message in messages %}<|im_start|>{{ message.role }}
        {{ message.content }}<|im_end|>
        {% endfor %}<|im_start|>assistant
        """)

        XCTAssertEqual(profile.extraStopStrings, ["<|im_end|>", "<|im_start|>"])
        XCTAssertTrue(profile.thinkingPairs.isEmpty)
    }

    func testOutputProfileDerivationLegacyGemma() {
        let profile = OutputSanitizationProfile.derived(fromChatTemplate: """
        {{ '<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>' }}
        """)

        XCTAssertEqual(profile.extraStopStrings, ["<end_of_turn>", "<start_of_turn>"])
    }

    func testOutputProfileDerivationHarmony() {
        let profile = OutputSanitizationProfile.derived(fromChatTemplate: """
        <|channel|>final<|message|>{{ content }}<|return|><|end|>
        """)

        XCTAssertEqual(profile.sliceAfterMarker, "<|channel|>final<|message|>")
        XCTAssertEqual(profile.allFinalMarkers, ["<|channel|>final<|message|>"])
        XCTAssertEqual(profile.scrubTokens, ["<|return|>", "<|end|>"])
    }

    func testOutputProfileMergesExplicitStreamPhaseConfiguration() {
        let profile = OutputSanitizationProfile(thinkingPairs: [
            OutputDelimiterPair(open: "<think>", close: "</think>")
        ])
        let merged = profile.merging(LLMStreamPhaseConfiguration(
            thinkingPairs: [
                OutputDelimiterPair(open: "<reason>", close: "</reason>"),
                OutputDelimiterPair(open: "<think>", close: "</think>")
            ],
            finalMarkers: ["<final>"]
        ))

        XCTAssertEqual(merged.thinkingPairs, [
            OutputDelimiterPair(open: "<think>", close: "</think>"),
            OutputDelimiterPair(open: "<reason>", close: "</reason>")
        ])
        XCTAssertEqual(merged.allFinalMarkers, ["<final>"])
    }

    func testPhaseAwareStreamEventMapsToLegacyStreamEvents() {
        let phaseAwareEvents: [LLMPhaseAwareStreamEvent] = [
            .requestSent(phase: .thinking),
            .phaseChanged(from: .thinking, to: .final),
            .finalAnswerDelta(text: "answer", bytesSoFar: 6),
            .finalAnswerSnapshot(text: "answer", bytesSoFar: 6, reason: .completed),
            .tokenChunk(preview: "answer", bytesSoFar: 6, phase: .final),
            .diagnostic(message: "mtp-diagnostic accepted=0/3"),
            .accelerationStats(LLMGenerationAccelerationStats(
                status: .active,
                accelerator: "mtp",
                maxDraftTokens: 3,
                draftCalls: 2,
                draftTokensGenerated: 6,
                draftTokensAccepted: 3
            )),
            .done(totalBytes: 6, duration: 1, phase: .final)
        ]

        let mapped = phaseAwareEvents.compactMap(\.streamEvent)

        XCTAssertEqual(mapped.count, 3)
        guard case .requestSent = mapped[0] else {
            return XCTFail("Expected requestSent.")
        }
        guard case .tokenChunk(let preview, let bytesSoFar) = mapped[1] else {
            return XCTFail("Expected tokenChunk.")
        }
        XCTAssertEqual(preview, "answer")
        XCTAssertEqual(bytesSoFar, 6)
        guard case .done(let totalBytes, _) = mapped[2] else {
            return XCTFail("Expected done.")
        }
        XCTAssertEqual(totalBytes, 6)
    }

    func testGenerationStreamEventMapsFinalContentToPhaseAwareEvents() {
        let generationEvents: [LLMGenerationStreamEvent] = [
            .contentDelta(phase: .thinking, text: "draft", bytesSoFar: 5),
            .contentDelta(phase: .final, text: "answer", bytesSoFar: 6),
            .contentSnapshot(
                phase: .thinking,
                text: "revised draft",
                bytesSoFar: 13,
                reason: .streamCorrection
            ),
            .contentSnapshot(
                phase: .final,
                text: "final answer",
                bytesSoFar: 12,
                reason: .completed
            )
        ]

        let mapped = generationEvents.compactMap(\.phaseAwareEvent)

        XCTAssertEqual(mapped.count, 2)
        guard case .finalAnswerDelta(let delta, let deltaBytes) = mapped[0] else {
            return XCTFail("Expected final delta.")
        }
        XCTAssertEqual(delta, "answer")
        XCTAssertEqual(deltaBytes, 6)
        guard case .finalAnswerSnapshot(let snapshot, let snapshotBytes, let reason) = mapped[1] else {
            return XCTFail("Expected final snapshot.")
        }
        XCTAssertEqual(snapshot, "final answer")
        XCTAssertEqual(snapshotBytes, 12)
        XCTAssertEqual(reason, .completed)
    }

    func testGenerationResultAccumulatorAppliesDeltaAndSnapshotSemanticsPerPhase() {
        let accumulator = LLMGenerationResultAccumulator()

        accumulator.record(.contentDelta(phase: .thinking, text: "draft", bytesSoFar: 5))
        accumulator.record(.contentDelta(phase: .final, text: "ans", bytesSoFar: 3))
        accumulator.record(.contentDelta(phase: .final, text: "wer", bytesSoFar: 6))
        accumulator.record(.contentSnapshot(
            phase: .thinking,
            text: "revised draft",
            bytesSoFar: 13,
            reason: .streamCorrection
        ))
        accumulator.record(.contentSnapshot(
            phase: .final,
            text: "final answer",
            bytesSoFar: 12,
            reason: .completed
        ))
        accumulator.record(.generationStats(
            promptTokens: 2,
            generatedTokens: 4,
            stopReason: "complete",
            templateMode: .embedded,
            phase: .final
        ))

        let result = accumulator.result()

        XCTAssertEqual(result.thinkingText, "revised draft")
        XCTAssertEqual(result.finalText, "final answer")
        XCTAssertEqual(result.stopReason, "complete")
        XCTAssertEqual(result.promptTokens, 2)
        XCTAssertEqual(result.generatedTokens, 4)
        XCTAssertEqual(result.phaseSegments, [
            LLMGenerationPhaseSegment(phase: .thinking, text: "revised draft"),
            LLMGenerationPhaseSegment(phase: .final, text: "final answer")
        ])
    }

    func testGenerationResultAccumulatorPreservesOrderedSegmentsAcrossPhaseReentry() {
        let accumulator = LLMGenerationResultAccumulator()

        accumulator.record(.contentDelta(phase: .thinking, text: "draft", bytesSoFar: 5))
        accumulator.record(.contentDelta(phase: .final, text: "answer", bytesSoFar: 6))
        accumulator.record(.contentDelta(phase: .thinking, text: " revised", bytesSoFar: 13))
        accumulator.record(.contentSnapshot(
            phase: .thinking,
            text: "draft revised",
            bytesSoFar: 13,
            reason: .streamCorrection
        ))

        let result = accumulator.result()

        XCTAssertEqual(result.thinkingText, "draft revised")
        XCTAssertEqual(result.finalText, "answer")
        XCTAssertEqual(result.phaseSegments, [
            LLMGenerationPhaseSegment(phase: .thinking, text: "draft"),
            LLMGenerationPhaseSegment(phase: .final, text: "answer"),
            LLMGenerationPhaseSegment(phase: .thinking, text: " revised")
        ])
    }

    func testGenerationAccelerationStatsAcceptanceRate() {
        let active = LLMGenerationAccelerationStats(
            status: .active,
            accelerator: "mtp",
            maxDraftTokens: 3,
            draftCalls: 4,
            draftTokensGenerated: 8,
            draftTokensAccepted: 6
        )
        XCTAssertEqual(active.acceptanceRate, 0.75)

        let unsupported = LLMGenerationAccelerationStats(
            status: .unsupported,
            accelerator: "mtp"
        )
        XCTAssertNil(unsupported.acceptanceRate)

        var aggregate = unsupported
        aggregate.merge(active)
        XCTAssertEqual(aggregate.status, .active)
        XCTAssertEqual(aggregate.maxDraftTokens, 3)
        XCTAssertEqual(aggregate.draftCalls, 4)
        XCTAssertEqual(aggregate.draftTokensGenerated, 8)
        XCTAssertEqual(aggregate.draftTokensAccepted, 6)
    }

    func testOutputProfileDerivationStartEndThinkingPair() {
        let profile = OutputSanitizationProfile.derived(fromChatTemplate: """
        <|START_THINKING|>{{ reasoning }}<|END_THINKING|>{{ content }}
        """)

        XCTAssertTrue(profile.thinkingPairs.contains(OutputDelimiterPair(
            open: "<|START_THINKING|>",
            close: "<|END_THINKING|>"
        )))
    }

    func testOutputProfileDerivationAdditionalThinkingDialects() {
        let cases: [(template: String, pair: OutputDelimiterPair)] = [
            (
                "[THINK]{{ reasoning }}[/THINK]{{ content }}",
                OutputDelimiterPair(open: "[THINK]", close: "[/THINK]")
            ),
            (
                "<seed:think>{{ reasoning }}</seed:think>{{ content }}",
                OutputDelimiterPair(open: "<seed:think>", close: "</seed:think>")
            ),
            (
                "<|inner_prefix|>{{ reasoning }}<|inner_suffix|>{{ content }}",
                OutputDelimiterPair(open: "<|inner_prefix|>", close: "<|inner_suffix|>")
            ),
            (
                "<|begin|>assistant<|think|>{{ reasoning }}<|end|><|content|>{{ content }}",
                OutputDelimiterPair(open: "<|think|>", close: "<|end|>")
            )
        ]

        for testCase in cases {
            let profile = OutputSanitizationProfile.derived(fromChatTemplate: testCase.template)

            XCTAssertTrue(
                profile.thinkingPairs.contains(testCase.pair),
                "Expected \(testCase.pair) for template \(testCase.template)"
            )
        }
    }

    func testOutputProfileDerivationFinalMarkerThinkingDialect() {
        let profile = OutputSanitizationProfile.derived(fromChatTemplate: """
        Here are my reasoning steps:
        [BEGIN FINAL RESPONSE]{{ content }}[END FINAL RESPONSE]
        """)

        XCTAssertEqual(profile.allFinalMarkers, ["[BEGIN FINAL RESPONSE]"])
        XCTAssertEqual(profile.scrubTokens, ["[END FINAL RESPONSE]"])
    }

    func testOutputProfileDerivationSolarContentMarker() {
        let profile = OutputSanitizationProfile.derived(fromChatTemplate: """
        {{ "<|begin|>assistant<|think|>" + reasoning + "<|end|>" }}
        {{ "<|begin|>assistant<|content|>" + content + "<|end|>" }}
        """)

        XCTAssertTrue(profile.thinkingPairs.contains(OutputDelimiterPair(
            open: "<|think|>",
            close: "<|end|>"
        )))
        XCTAssertEqual(profile.allFinalMarkers, ["<|content|>"])
        XCTAssertEqual(profile.scrubTokens, ["<|end|>", "<|content|>"])
    }

    func testOutputProfileDerivationUnknownModelIsEmpty() {
        let profile = OutputSanitizationProfile.derived(fromChatTemplate: "plain template")

        XCTAssertEqual(profile, .empty)
    }

    func testProfileDrivenSanitizerDoesNotStripUnknownThinkContent() {
        let raw = "The literal <think> tag can be discussed safely."

        XCTAssertEqual(
            LLMResponseSanitizer.unwrapStructuredOutput(raw, using: .empty),
            raw
        )
    }

    func testProfileDrivenSanitizerStripsPromptPrefilledThinkingPrefix() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let profile = OutputSanitizationProfile(thinkingPairs: [pair])

        XCTAssertEqual(
            LLMResponseSanitizer.unwrapStructuredOutput(
                "private reasoning\n</think>\nfinal answer",
                using: profile,
                continuingOpenThinkingPairs: [pair]
            ),
            "final answer"
        )
    }

    func testProfileDrivenSanitizerReturnsEmptyForUnclosedPromptPrefilledThinking() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let profile = OutputSanitizationProfile(thinkingPairs: [pair])

        XCTAssertEqual(
            LLMResponseSanitizer.unwrapStructuredOutput(
                "private reasoning that never closed",
                using: profile,
                continuingOpenThinkingPairs: [pair]
            ),
            ""
        )
    }

    func testProfileDrivenSanitizerDoesNotStripStrayThinkingCloseWithoutContinuation() {
        let pair = OutputDelimiterPair(open: "<think>", close: "</think>")
        let profile = OutputSanitizationProfile(thinkingPairs: [pair])
        let raw = "</think> final answer"

        XCTAssertEqual(
            LLMResponseSanitizer.unwrapStructuredOutput(raw, using: profile),
            raw
        )
    }

    func testProfileDrivenSanitizerStripsGemma4ThinkingBlock() {
        let profile = OutputSanitizationProfile(
            thinkingPairs: [
                OutputDelimiterPair(open: "<|channel>thought", close: "<channel|>")
            ]
        )
        let raw = """
        <|channel>thought
        hidden notes
        <channel|>
        visible answer
        """

        XCTAssertEqual(
            LLMResponseSanitizer.unwrapStructuredOutput(raw, using: profile),
            "visible answer"
        )
    }

    func testProfileDrivenSanitizerStripsStartEndThinkingBlock() {
        let profile = OutputSanitizationProfile(
            thinkingPairs: [
                OutputDelimiterPair(open: "<|START_THINKING|>", close: "<|END_THINKING|>")
            ]
        )

        XCTAssertEqual(
            LLMResponseSanitizer.unwrapStructuredOutput(
                "<|START_THINKING|>hidden notes<|END_THINKING|>visible answer",
                using: profile
            ),
            "visible answer"
        )
    }

    func testProfileDrivenSanitizerStripsAdditionalThinkingDialects() {
        let cases: [(pair: OutputDelimiterPair, raw: String)] = [
            (
                OutputDelimiterPair(open: "[THINK]", close: "[/THINK]"),
                "[THINK]hidden notes[/THINK]visible answer"
            ),
            (
                OutputDelimiterPair(open: "<seed:think>", close: "</seed:think>"),
                "<seed:think>hidden notes</seed:think>visible answer"
            ),
            (
                OutputDelimiterPair(open: "<|inner_prefix|>", close: "<|inner_suffix|>"),
                "<|inner_prefix|>hidden notes<|inner_suffix|>visible answer"
            ),
            (
                OutputDelimiterPair(open: "<|think|>", close: "<|end|>"),
                "<|think|>hidden notes<|end|><|content|>visible answer<|end|>"
            )
        ]

        for testCase in cases {
            let profile = OutputSanitizationProfile(
                thinkingPairs: [testCase.pair],
                scrubTokens: ["<|end|>", "<|content|>"],
                finalMarkers: testCase.pair.open == "<|think|>" ? ["<|content|>"] : []
            )

            XCTAssertEqual(
                LLMResponseSanitizer.unwrapStructuredOutput(testCase.raw, using: profile),
                "visible answer",
                "Expected sanitizer to strip \(testCase.pair)"
            )
        }
    }

    func testProfileDrivenSanitizerSlicesFinalMarkerThinkingDialect() {
        let profile = OutputSanitizationProfile(
            scrubTokens: ["[END FINAL RESPONSE]"],
            finalMarkers: ["[BEGIN FINAL RESPONSE]"]
        )

        XCTAssertEqual(
            LLMResponseSanitizer.unwrapStructuredOutput(
                "hidden reasoning[BEGIN FINAL RESPONSE]visible answer[END FINAL RESPONSE]",
                using: profile
            ),
            "visible answer"
        )
    }

    func testLegacySanitizerStripsKnownThinkingDialects() {
        let raw = """
        [THINK]hidden[/THINK]
        <seed:think>hidden</seed:think>
        <|inner_prefix|>hidden<|inner_suffix|>
        <|think|>hidden<|end|><|content|>visible answer<|end|>
        """

        XCTAssertEqual(
            LLMResponseSanitizer.unwrapStructuredOutput(raw),
            "visible answer"
        )
    }

    func testProfileDrivenSanitizerSlicesHarmonyFinalMarker() {
        let profile = OutputSanitizationProfile(
            sliceAfterMarker: "<|channel|>final<|message|>",
            scrubTokens: ["<|return|>"]
        )

        XCTAssertEqual(
            LLMResponseSanitizer.unwrapStructuredOutput(
                "analysis<|channel|>final<|message|>{\"ok\":true}<|return|>",
                using: profile
            ),
            "{\"ok\":true}"
        )
    }

    func testPreviewDescribesEmptyAndWhitespaceResponses() {
        XCTAssertEqual(LLMResponsePreview.describe(""), "<empty response>")
        XCTAssertEqual(LLMResponsePreview.describe("  \n"), "<whitespace-only response: ..\\n>")
    }

    func testCuratedRecommendationChoosesLargestModelWithinRAM() {
        let models = [
            CuratedModel(
                id: "small",
                displayName: "Small",
                subtitle: "",
                hfRepo: "org/small",
                hfFilename: "small.gguf",
                approxSizeBytes: 1,
                contextLength: 1,
                quantization: "Q4",
                recommendedRAMGB: 8,
                sha256: nil
            ),
            CuratedModel(
                id: "large",
                displayName: "Large",
                subtitle: "",
                hfRepo: "org/large",
                hfFilename: "large.gguf",
                approxSizeBytes: 1,
                contextLength: 1,
                quantization: "Q4",
                recommendedRAMGB: 32,
                sha256: nil
            )
        ]

        let recommendation = CuratedModelCatalog.recommendedModel(
            forPhysicalMemoryBytes: 16 * 1_073_741_824,
            among: models
        )
        XCTAssertEqual(recommendation?.id, "small")
    }

    func testDefaultCuratedCatalogRecommendationTiers() {
        let belowSmallestTier = CuratedModelCatalog.recommendedModel(
            forPhysicalMemoryBytes: UInt64(8) * 1_073_741_824 - 1
        )
        let smallTier = CuratedModelCatalog.recommendedModel(
            forPhysicalMemoryBytes: UInt64(8) * 1_073_741_824
        )
        let mediumTier = CuratedModelCatalog.recommendedModel(
            forPhysicalMemoryBytes: UInt64(16) * 1_073_741_824
        )
        let largeTier = CuratedModelCatalog.recommendedModel(
            forPhysicalMemoryBytes: UInt64(32) * 1_073_741_824
        )
        let topTier = CuratedModelCatalog.recommendedModel(
            forPhysicalMemoryBytes: UInt64(48) * 1_073_741_824
        )

        XCTAssertNil(belowSmallestTier)
        XCTAssertEqual(smallTier?.id, "gemma-4-e2b-it-q4km")
        XCTAssertEqual(smallTier?.displayName, "Gemma 4 E2B Instruct (Q4_K_M)")
        XCTAssertEqual(smallTier?.hfRepo, "bartowski/google_gemma-4-E2B-it-GGUF")
        XCTAssertEqual(smallTier?.hfFilename, "google_gemma-4-E2B-it-Q4_K_M.gguf")
        XCTAssertEqual(smallTier?.approxSizeBytes, 3_500_000_000)
        XCTAssertEqual(smallTier?.contextLength, 131_072)
        XCTAssertEqual(smallTier?.recommendedRAMGB, 8)
        XCTAssertEqual(smallTier?.samplingDefaults?.temperature, 1.0)
        XCTAssertEqual(smallTier?.samplingDefaults?.topP, 0.95)
        XCTAssertEqual(smallTier?.samplingDefaults?.topK, 64)
        XCTAssertEqual(mediumTier?.id, "qwen3.5-9b-instruct-q4km")
        XCTAssertEqual(largeTier?.id, "gemma-4-26b-a4b-it-q4km")
        XCTAssertEqual(largeTier?.samplingDefaults?.temperature, 1.0)
        XCTAssertEqual(largeTier?.samplingDefaults?.topP, 0.95)
        XCTAssertEqual(largeTier?.samplingDefaults?.topK, 64)
        XCTAssertEqual(topTier?.id, "gemma-4-26b-a4b-it-heretic-q6k")
        XCTAssertEqual(topTier?.displayName, "Gemma 4 26B A4B Uncensored Instruct (Q6_K)")
        XCTAssertEqual(topTier?.hfRepo, "nohurry/gemma-4-26B-A4B-it-heretic-GUFF")
        XCTAssertEqual(topTier?.hfFilename, "gemma-4-26b-a4b-it-heretic.q6_k.gguf")
        XCTAssertEqual(topTier?.approxSizeBytes, 23_172_471_776)
        XCTAssertEqual(topTier?.contextLength, 262_144)
        XCTAssertEqual(topTier?.recommendedRAMGB, 48)
        XCTAssertEqual(topTier?.samplingDefaults?.temperature, 1.0)
        XCTAssertEqual(topTier?.samplingDefaults?.topP, 0.95)
        XCTAssertEqual(topTier?.samplingDefaults?.topK, 64)
        XCTAssertTrue(CuratedModelCatalog.all.contains { $0.id == "qwen3.6-27b-dense-q4km" })
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("CarbocationLocalLLMTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeHuggingFaceCacheRepo(
        hubRoot: URL,
        repoFolder: String,
        commit: String,
        files: [String: (blobHash: String, data: Data)]
    ) throws -> URL {
        let repoRoot = hubRoot.appendingPathComponent(repoFolder, isDirectory: true)
        let blobs = repoRoot.appendingPathComponent("blobs", isDirectory: true)
        let refs = repoRoot.appendingPathComponent("refs", isDirectory: true)
        let snapshot = repoRoot
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(commit, isDirectory: true)

        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: refs, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try Data(commit.utf8).write(to: refs.appendingPathComponent("main"))

        for (relativePath, file) in files {
            try file.data.write(to: blobs.appendingPathComponent(file.blobHash))
            let linkURL = snapshot.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: linkURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let depth = relativePath.split(separator: "/").dropLast().count
            let prefix = Array(repeating: "..", count: depth + 2).joined(separator: "/")
            try FileManager.default.createSymbolicLink(
                atPath: linkURL.path,
                withDestinationPath: "\(prefix)/blobs/\(file.blobHash)"
            )
        }

        return snapshot
    }

    private func addFakeModel(
        to library: ModelLibrary,
        sourcesRoot: URL,
        displayName: String,
        filename: String,
        sizeBytes: Int64
    ) async throws -> InstalledModel {
        let source = sourcesRoot.appendingPathComponent("\(UUID().uuidString)-\(filename)")
        try Data("fake gguf".utf8).write(to: source)
        return try await library.add(
            weightsAt: source,
            displayName: displayName,
            filename: filename,
            sizeBytes: sizeBytes,
            source: .imported,
            contextLength: 4_096
        )
    }

    private static let mockCommit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    private func refsJSON(commit: String) -> String {
        #"""
        {
          "branches": [
            {"name": "main", "targetCommit": "\#(commit)"}
          ],
          "tags": []
        }
        """#
    }

    private func makeMinimalGGUF(
        contextLength: UInt32,
        nextNPredictLayers: UInt32? = nil,
        architecture: String = "llama"
    ) -> Data {
        var data = Data([0x47, 0x47, 0x55, 0x46])
        appendUInt32(3, to: &data)
        appendInt64(0, to: &data)
        appendInt64(nextNPredictLayers == nil ? 2 : 3, to: &data)

        appendGGUFString("general.architecture", to: &data)
        appendInt32(8, to: &data)
        appendGGUFString(architecture, to: &data)

        appendGGUFString("llama.context_length", to: &data)
        appendInt32(4, to: &data)
        appendUInt32(contextLength, to: &data)

        if let nextNPredictLayers {
            appendGGUFString("gemma4.nextn_predict_layers", to: &data)
            appendInt32(4, to: &data)
            appendUInt32(nextNPredictLayers, to: &data)
        }
        return data
    }

    private func appendGGUFString(_ string: String, to data: inout Data) {
        let bytes = Array(string.utf8)
        appendUInt64(UInt64(bytes.count), to: &data)
        data.append(contentsOf: bytes)
    }

    private func appendUInt32(_ value: UInt32, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) {
            data.append(contentsOf: $0)
        }
    }

    private func appendInt32(_ value: Int32, to data: inout Data) {
        appendUInt32(UInt32(bitPattern: value), to: &data)
    }

    private func appendUInt64(_ value: UInt64, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) {
            data.append(contentsOf: $0)
        }
    }

    private func appendInt64(_ value: Int64, to data: inout Data) {
        appendUInt64(UInt64(bitPattern: value), to: &data)
    }

    private func calibrationRuntime(
        batchSizeLimit: Int,
        algorithmVersion: Int = LlamaContextCalibrationAlgorithm.version
    ) -> LlamaContextCalibrationRuntimeFingerprint {
        LlamaContextCalibrationRuntimeFingerprint(
            platform: "macOS",
            gpuLayerCount: 999,
            useMemoryMap: true,
            batchSizeLimit: batchSizeLimit,
            threadCount: 4,
            algorithmVersion: algorithmVersion
        )
    }
    func testRGB8ImageValidationRequiresTightThreeByteLayout() throws {
        let valid = try LLMImageInput.rgb8(
            width: 2,
            height: 1,
            data: Data([255, 0, 0, 0, 255, 0])
        ).normalizedRGB8()

        XCTAssertEqual(valid.width, 2)
        XCTAssertEqual(valid.height, 1)
        XCTAssertEqual(valid.data.count, 6)

        do {
            _ = try LLMImageInput.rgb8(width: 2, height: 2, data: Data(repeating: 0, count: 11))
                .normalizedRGB8(location: LLMContentLocation(messageIndex: 3, partIndex: 1))
            XCTFail("Expected RGB8 layout validation to fail.")
        } catch let error as LLMEngineError {
            guard case .invalidImageData(_, let location) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(location, LLMContentLocation(messageIndex: 3, partIndex: 1))
        }
    }

    func testEncodedImageMIMEMismatchIncludesLocation() throws {
        let pngData = try makeOnePixelPNGData()
        let location = LLMContentLocation(messageIndex: 0, partIndex: 2)

        do {
            _ = try LLMImageInput.encoded(data: pngData, mimeType: "image/jpeg")
                .normalizedRGB8(location: location)
            XCTFail("Expected MIME mismatch.")
        } catch let error as LLMEngineError {
            guard case .imageMIMEMismatch(let declared, let detected, let errorLocation) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(declared, "image/jpeg")
            XCTAssertEqual(detected, "image/png")
            XCTAssertEqual(errorLocation, location)
        }
    }

    func testTextOnlyEngineRejectsImageMessagesWithRequestLocation() async throws {
        let engine = TextOnlyScriptedEngine()
        let messages = [
            LLMChatMessage(role: .user, content: [
                .text("Look at this"),
                .image(.rgb8(width: 1, height: 1, data: Data([0, 0, 0])))
            ])
        ]

        do {
            _ = try await engine.generate(messages: messages, options: .extractionSafe) { _ in }
            XCTFail("Expected image modality rejection.")
        } catch let error as LLMEngineError {
            guard case .unsupportedInputModality(.image, let location) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(location, LLMContentLocation(messageIndex: 0, partIndex: 1))
        }
    }

    func testMessageAPIKeepsTextOnlyEnginesSourceCompatible() async throws {
        let engine = TextOnlyScriptedEngine(response: "ok")

        let response = try await engine.generate(
            messages: [
                LLMChatMessage(role: .system, text: "System rule"),
                LLMChatMessage(role: .user, text: "Hello")
            ],
            options: .extractionSafe
        ) { _ in }

        let invocation = await engine.lastInvocation()
        XCTAssertEqual(response, "ok")
        XCTAssertEqual(invocation?.system, "System rule")
        XCTAssertEqual(invocation?.prompt, "Hello")
    }

    private func makeOnePixelPNGData() throws -> Data {
        let rgba = Data([0x11, 0x22, 0x33, 0xFF])
        let provider = CGDataProvider(data: rgba as CFData)!
        let image = CGImage(
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )!

        let output = NSMutableData()
        let destination = CGImageDestinationCreateWithData(
            output,
            UTType.png.identifier as CFString,
            1,
            nil
        )!
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return output as Data
    }
}

private actor TextOnlyScriptedEngine: LLMEngine {
    private var response: String
    private var invocation: (system: String, prompt: String)?

    init(response: String = "") {
        self.response = response
    }

    func currentModelID() async -> UUID? {
        nil
    }

    func currentContextSize() async -> Int {
        4_096
    }

    func generate(
        system: String,
        prompt: String,
        options: GenerationOptions,
        onEvent: @Sendable (LLMStreamEvent) -> Void
    ) async throws -> String {
        invocation = (system, prompt)
        onEvent(.requestSent)
        onEvent(.generationStats(
            promptTokens: 1,
            generatedTokens: response.isEmpty ? 0 : 1,
            stopReason: "complete",
            templateMode: .unavailable
        ))
        onEvent(.done(totalBytes: response.utf8.count, duration: 0))
        return response
    }

    func lastInvocation() -> (system: String, prompt: String)? {
        invocation
    }
}

private extension CarbocationLocalLLMTests {
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
}

private final class ThreadProbeRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recordedValue: Bool?

    var value: Bool? {
        lock.lock()
        defer { lock.unlock() }
        return recordedValue
    }

    func record(_ value: Bool) {
        lock.lock()
        recordedValue = value
        lock.unlock()
    }
}

private final class MockHuggingFaceHTTPClient: HuggingFaceModelResolverHTTPClient, @unchecked Sendable {
    private let refsData: Data
    private let treeData: Data
    private let queue = DispatchQueue(label: "MockHuggingFaceHTTPClient")
    private var recordedRequests: [URLRequest] = []

    init(refsJSON: String, treeJSON: String) {
        self.refsData = Data(refsJSON.utf8)
        self.treeData = Data(treeJSON.utf8)
    }

    var requests: [URLRequest] {
        queue.sync { recordedRequests }
    }

    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        queue.sync {
            recordedRequests.append(request)
        }

        let path = request.url?.absoluteString ?? ""
        let data = path.contains("/refs") ? refsData : treeData
        let response = HTTPURLResponse(
            url: request.url ?? URL(string: "https://huggingface.test")!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: nil
        )!
        return (data, response)
    }
}
