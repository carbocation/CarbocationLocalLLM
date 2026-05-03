import CarbocationLocalLLM
import CarbocationLocalLLMUI
import XCTest

final class ModelLibraryPickerLabelPolicyTests: XCTestCase {
    private let physicalMemoryBytes = UInt64(16) * 1_073_741_824

    func testRecommendedCuratedModelInstalledGetsRecommendedLabel() {
        let installedModel = installedModel(for: mediumModel)
        let recommended = CuratedModelCatalog.recommendedModel(
            forPhysicalMemoryBytes: physicalMemoryBytes,
            among: curatedModels
        )
        let bestInstalled = ModelLibraryPickerLabelPolicy.bestInstalledCuratedModel(
            forPhysicalMemoryBytes: physicalMemoryBytes,
            installedModels: [installedModel],
            curatedModels: curatedModels
        )

        let label = ModelLibraryPickerLabelPolicy.default.installedModelLabel(
            for: installedModel,
            recommendedCuratedModel: recommended,
            bestInstalledCuratedModel: bestInstalled
        )

        XCTAssertEqual(label?.title, "Recommended")
        XCTAssertEqual(label?.tone, .accent)
    }

    func testBestInstalledCuratedModelGetsLabelWhenRecommendedModelIsMissing() {
        let installedModel = installedModel(for: smallModel)
        let recommended = CuratedModelCatalog.recommendedModel(
            forPhysicalMemoryBytes: physicalMemoryBytes,
            among: curatedModels
        )
        let bestInstalled = ModelLibraryPickerLabelPolicy.bestInstalledCuratedModel(
            forPhysicalMemoryBytes: physicalMemoryBytes,
            installedModels: [installedModel],
            curatedModels: curatedModels
        )

        let label = ModelLibraryPickerLabelPolicy.default.installedModelLabel(
            for: installedModel,
            recommendedCuratedModel: recommended,
            bestInstalledCuratedModel: bestInstalled
        )

        XCTAssertEqual(recommended?.id, mediumModel.id)
        XCTAssertEqual(bestInstalled?.id, smallModel.id)
        XCTAssertEqual(label?.title, "Best Installed")
        XCTAssertEqual(label?.tone, .positive)
    }

    func testNoBestInstalledLabelForNonCuratedInstalledModels() {
        let importedModel = installedModel(for: smallModel, source: .imported)
        let customModel = installedModel(for: smallModel, source: .customHF)
        let recommended = CuratedModelCatalog.recommendedModel(
            forPhysicalMemoryBytes: physicalMemoryBytes,
            among: curatedModels
        )
        let bestInstalled = ModelLibraryPickerLabelPolicy.bestInstalledCuratedModel(
            forPhysicalMemoryBytes: physicalMemoryBytes,
            installedModels: [importedModel, customModel],
            curatedModels: curatedModels
        )

        let importedLabel = ModelLibraryPickerLabelPolicy.default.installedModelLabel(
            for: importedModel,
            recommendedCuratedModel: recommended,
            bestInstalledCuratedModel: bestInstalled
        )
        let customLabel = ModelLibraryPickerLabelPolicy.default.installedModelLabel(
            for: customModel,
            recommendedCuratedModel: recommended,
            bestInstalledCuratedModel: bestInstalled
        )

        XCTAssertNil(bestInstalled)
        XCTAssertNil(importedLabel)
        XCTAssertNil(customLabel)
    }

    func testLegacySystemModelLabelMarksAppleIntelligenceNotRecommended() {
        let label = ModelLibraryPickerLabelPolicy.default.systemModelLabel(for: appleIntelligenceOption)

        XCTAssertEqual(label?.title, "Not Recommended")
        XCTAssertEqual(label?.tone, .warning)
    }

    func testDefaultSystemModelLabelMarksAppleIntelligenceNotRecommendedWhenCuratedModelFits() {
        let label = ModelLibraryPickerLabelPolicy.default.systemModelLabel(
            for: appleIntelligenceOption,
            recommendedCuratedModel: smallModel
        )

        XCTAssertEqual(label?.title, "Not Recommended")
        XCTAssertEqual(label?.tone, .warning)
    }

    func testDefaultSystemModelLabelRecommendsAppleIntelligenceWhenNoCuratedModelFits() {
        let label = ModelLibraryPickerLabelPolicy.default.systemModelLabel(
            for: appleIntelligenceOption,
            recommendedCuratedModel: nil
        )

        XCTAssertEqual(label?.title, "Recommended")
        XCTAssertEqual(label?.tone, .accent)
    }

    func testSystemModelLabelCanBeRemovedOrReplaced() {
        let noSystemLabels = ModelLibraryPickerLabelPolicy(systemModelLabels: [:])
        let customLabel = ModelLibraryPickerStatusLabel("Use Carefully", tone: .secondary)
        let customSystemLabels = ModelLibraryPickerLabelPolicy(
            systemModelLabels: [.system(.appleIntelligence): customLabel]
        )

        XCTAssertNil(noSystemLabels.systemModelLabel(
            for: appleIntelligenceOption,
            recommendedCuratedModel: nil
        ))
        XCTAssertEqual(
            customSystemLabels.systemModelLabel(
                for: appleIntelligenceOption,
                recommendedCuratedModel: nil
            ),
            customLabel
        )
    }

    func testRecommendedLabelNilSuppressesAppleIntelligenceFallbackLabel() {
        let noRecommendedLabels = ModelLibraryPickerLabelPolicy(recommendedLabel: nil)

        XCTAssertNil(noRecommendedLabels.systemModelLabel(
            for: appleIntelligenceOption,
            recommendedCuratedModel: nil
        ))
    }

    private var curatedModels: [CuratedModel] {
        [smallModel, mediumModel, largeModel]
    }

    private var smallModel: CuratedModel {
        CuratedModel(
            id: "small",
            displayName: "Small",
            subtitle: "Small model",
            hfRepo: "example/small",
            hfFilename: "small-Q4_K_M.gguf",
            approxSizeBytes: 1_000_000,
            contextLength: 8_192,
            quantization: "Q4_K_M",
            recommendedRAMGB: 8,
            sha256: nil
        )
    }

    private var mediumModel: CuratedModel {
        CuratedModel(
            id: "medium",
            displayName: "Medium",
            subtitle: "Medium model",
            hfRepo: "example/medium",
            hfFilename: "medium-Q4_K_M.gguf",
            approxSizeBytes: 2_000_000,
            contextLength: 16_384,
            quantization: "Q4_K_M",
            recommendedRAMGB: 16,
            sha256: nil
        )
    }

    private var largeModel: CuratedModel {
        CuratedModel(
            id: "large",
            displayName: "Large",
            subtitle: "Large model",
            hfRepo: "example/large",
            hfFilename: "large-Q4_K_M.gguf",
            approxSizeBytes: 3_000_000,
            contextLength: 32_768,
            quantization: "Q4_K_M",
            recommendedRAMGB: 32,
            sha256: nil
        )
    }

    private var appleIntelligenceOption: LLMSystemModelOption {
        LLMSystemModelOption(
            selection: .system(.appleIntelligence),
            displayName: "Apple Intelligence",
            subtitle: "Built-in system model",
            contextLength: 4_096,
            systemImageName: "sparkles"
        )
    }

    private func installedModel(
        for curatedModel: CuratedModel,
        source: ModelSource = .curated
    ) -> InstalledModel {
        InstalledModel(
            displayName: curatedModel.displayName,
            filename: curatedModel.hfFilename,
            sizeBytes: curatedModel.approxSizeBytes,
            contextLength: curatedModel.contextLength,
            quantization: curatedModel.quantization,
            source: source,
            hfRepo: curatedModel.hfRepo,
            hfFilename: curatedModel.hfFilename
        )
    }
}
