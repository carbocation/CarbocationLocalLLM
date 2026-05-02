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
                trainingContext: 4_096,
                mode: .manual,
                manualContext: 32_768
            ),
            32_768
        )
    }

    func testGGUFMetadataReadsTrainingContextLength() throws {
        let root = try makeTemporaryDirectory()
        let url = root.appendingPathComponent("metadata-only.gguf")
        try makeMinimalGGUF(contextLength: 32_768).write(to: url)

        XCTAssertEqual(GGUFMetadata.trainingContextLength(at: url), 32_768)
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

        let options = GenerationOptionsResolver.configuredExtractionOptions(defaults: defaults)
        XCTAssertEqual(options.temperature, 0.2)
        XCTAssertEqual(options.topP, 0.7)
        XCTAssertEqual(options.topK, 64)
    }

    func testGenerationOptionsDecodeLegacyPayloadWithNewDefaults() throws {
        let data = Data(#"{"temperature":0.1,"topP":0.8,"topK":32}"#.utf8)
        let options = try JSONDecoder().decode(GenerationOptions.self, from: data)

        XCTAssertEqual(options.temperature, 0.1)
        XCTAssertEqual(options.topP, 0.8)
        XCTAssertEqual(options.topK, 32)
        XCTAssertNil(options.seed)
        XCTAssertEqual(options.stopSequences, [])
        XCTAssertFalse(options.stopAtBalancedJSON)
        XCTAssertFalse(options.enableThinking)
    }

    func testGenerationOptionsOnlyEncodesEnableThinkingWhenTrue() throws {
        let data = try JSONEncoder().encode(GenerationOptions(enableThinking: true))
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        XCTAssertEqual(object?["enableThinking"] as? Bool, true)
        XCTAssertNil((try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(GenerationOptions())
        ) as? [String: Any])?["enableThinking"])
    }

    @MainActor
    func testModelLibraryImportsSyncsAndDeletesGGUF() async throws {
        let root = try makeTemporaryDirectory()
        let source = root.appendingPathComponent("source-Q4_K_M.gguf")
        try Data("fake gguf".utf8).write(to: source)
        let modelsRoot = root.appendingPathComponent("Models", isDirectory: true)

        let library = ModelLibrary(root: modelsRoot) { url in
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
    func testModelLibraryTrustsProvidedContextLengthWithoutProbe() async throws {
        let root = try makeTemporaryDirectory()
        let source = root.appendingPathComponent("download-Q4_K_M.gguf")
        try Data("fake gguf".utf8).write(to: source)
        let modelsRoot = root.appendingPathComponent("Models", isDirectory: true)
        let recorder = ThreadProbeRecorder()

        let library = ModelLibrary(root: modelsRoot) { _ in
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

        let library = ModelLibrary(root: root)
        await library.refresh()

        XCTAssertEqual(library.models.count, 1)
        XCTAssertEqual(library.models[0].id, modelID)
        XCTAssertEqual(library.models[0].displayName, "Orphan-Q5_K_M")
        XCTAssertEqual(library.models[0].quantization, "Q5_K_M")
    }

    @MainActor
    func testModelLibraryResolveInstalledModelRefreshesOnDemand() async throws {
        let root = try makeTemporaryDirectory()
        let modelID = UUID()
        let directory = root.appendingPathComponent(modelID.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("fake".utf8).write(to: directory.appendingPathComponent("Resolvable-Q4_K_M.gguf"))

        let library = ModelLibrary(root: root)

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

        let library = ModelLibrary(root: modelsRoot) { _ in
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

        let library = ModelLibrary(root: root)
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

        let library = ModelLibrary(root: modelsRoot)

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
        XCTAssertEqual(profile.scrubTokens, ["<|return|>", "<|end|>"])
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

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("CarbocationLocalLLMTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeMinimalGGUF(contextLength: UInt32) -> Data {
        var data = Data([0x47, 0x47, 0x55, 0x46])
        appendUInt32(3, to: &data)
        appendInt64(0, to: &data)
        appendInt64(2, to: &data)

        appendGGUFString("general.architecture", to: &data)
        appendInt32(8, to: &data)
        appendGGUFString("llama", to: &data)

        appendGGUFString("llama.context_length", to: &data)
        appendInt32(4, to: &data)
        appendUInt32(contextLength, to: &data)
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
