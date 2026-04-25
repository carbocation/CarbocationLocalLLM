import XCTest
@testable import CarbocationLocalLLM

final class CarbocationLocalLLMTests: XCTestCase {
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

        let blob = HuggingFaceURL.parse("https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/blob/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
        XCTAssertEqual(blob?.repo, "bartowski/Qwen2.5-7B-Instruct-GGUF")
        XCTAssertEqual(blob?.filename, "Qwen2.5-7B-Instruct-Q4_K_M.gguf")

        let compact = HuggingFaceURL.parse("bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
        XCTAssertEqual(compact?.repo, "bartowski/Qwen2.5-7B-Instruct-GGUF")
        XCTAssertEqual(compact?.filename, "Qwen2.5-7B-Instruct-Q4_K_M.gguf")

        XCTAssertNil(HuggingFaceURL.parse("https://huggingface.co/bartowski/model/blob/main/README.md"))
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
