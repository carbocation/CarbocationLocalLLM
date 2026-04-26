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
    }

    @MainActor
    func testModelLibraryImportsSyncsAndDeletesGGUF() throws {
        let root = try makeTemporaryDirectory()
        let source = root.appendingPathComponent("source-Q4_K_M.gguf")
        try Data("fake gguf".utf8).write(to: source)
        let modelsRoot = root.appendingPathComponent("Models", isDirectory: true)

        let library = ModelLibrary(root: modelsRoot) { url in
            url.lastPathComponent == "source-Q4_K_M.gguf" ? 32_768 : nil
        }

        let model = try library.importFile(at: source, displayName: "Test Model")

        XCTAssertEqual(library.models.count, 1)
        XCTAssertEqual(model.displayName, "Test Model")
        XCTAssertEqual(model.quantization, "Q4_K_M")
        XCTAssertEqual(model.contextLength, 32_768)
        XCTAssertTrue(FileManager.default.fileExists(atPath: model.weightsURL(in: modelsRoot).path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: model.metadataURL(in: modelsRoot).path))

        try library.syncContextLength(65_536, for: model.id)
        XCTAssertEqual(library.model(id: model.id)?.contextLength, 65_536)

        try library.delete(id: model.id)
        XCTAssertTrue(library.models.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: model.directory(in: modelsRoot).path))
    }

    @MainActor
    func testModelLibrarySynthesizesOrphanMetadata() throws {
        let root = try makeTemporaryDirectory()
        let modelID = UUID()
        let directory = root.appendingPathComponent(modelID.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("fake".utf8).write(to: directory.appendingPathComponent("Orphan-Q5_K_M.gguf"))

        let library = ModelLibrary(root: root)

        XCTAssertEqual(library.models.count, 1)
        XCTAssertEqual(library.models[0].id, modelID)
        XCTAssertEqual(library.models[0].displayName, "Orphan-Q5_K_M")
        XCTAssertEqual(library.models[0].quantization, "Q5_K_M")
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
}
