import CarbocationLocalLLM
import Foundation

public struct ModelLibraryPickerStatusLabel: Equatable, Hashable, Sendable {
    public enum Tone: Equatable, Hashable, Sendable {
        case accent
        case positive
        case warning
        case secondary
    }

    public var title: String
    public var systemImageName: String?
    public var tone: Tone

    public init(
        _ title: String,
        systemImageName: String? = nil,
        tone: Tone = .accent
    ) {
        self.title = title
        self.systemImageName = systemImageName
        self.tone = tone
    }
}

public struct ModelLibraryPickerLabelPolicy: Equatable, Sendable {
    public static let recommendedLabel = ModelLibraryPickerStatusLabel("Recommended", tone: .accent)
    public static let bestInstalledLabel = ModelLibraryPickerStatusLabel("Best Installed", tone: .positive)
    public static let notRecommendedLabel = ModelLibraryPickerStatusLabel("Not Recommended", tone: .warning)

    public static let defaultSystemModelLabels: [LLMModelSelection: ModelLibraryPickerStatusLabel] = [
        .system(.appleIntelligence): notRecommendedLabel
    ]

    public static let `default` = ModelLibraryPickerLabelPolicy()

    public var recommendedLabel: ModelLibraryPickerStatusLabel?
    public var bestInstalledLabel: ModelLibraryPickerStatusLabel?
    public var systemModelLabels: [LLMModelSelection: ModelLibraryPickerStatusLabel]

    public init(
        recommendedLabel: ModelLibraryPickerStatusLabel? = Self.recommendedLabel,
        bestInstalledLabel: ModelLibraryPickerStatusLabel? = Self.bestInstalledLabel,
        systemModelLabels: [LLMModelSelection: ModelLibraryPickerStatusLabel] = Self.defaultSystemModelLabels
    ) {
        self.recommendedLabel = recommendedLabel
        self.bestInstalledLabel = bestInstalledLabel
        self.systemModelLabels = systemModelLabels
    }

    public func systemModelLabel(for model: LLMSystemModelOption) -> ModelLibraryPickerStatusLabel? {
        systemModelLabels[model.selection]
    }

    public func systemModelLabel(
        for model: LLMSystemModelOption,
        recommendedCuratedModel: CuratedModel?
    ) -> ModelLibraryPickerStatusLabel? {
        guard let label = systemModelLabel(for: model) else {
            return nil
        }

        if model.selection == .system(.appleIntelligence),
           recommendedCuratedModel == nil,
           label == Self.notRecommendedLabel {
            return recommendedLabel
        }

        return label
    }

    public func installedModelLabel(
        for model: InstalledModel,
        recommendedCuratedModel: CuratedModel?,
        bestInstalledCuratedModel: CuratedModel?
    ) -> ModelLibraryPickerStatusLabel? {
        if let recommendedCuratedModel,
           Self.installedModel(model, matches: recommendedCuratedModel) {
            return recommendedLabel
        }

        if let bestInstalledCuratedModel,
           Self.installedModel(model, matches: bestInstalledCuratedModel) {
            return bestInstalledLabel
        }

        return nil
    }

    public static func bestInstalledCuratedModel(
        forPhysicalMemoryBytes physicalMemoryBytes: UInt64,
        installedModels: [InstalledModel],
        curatedModels: [CuratedModel]
    ) -> CuratedModel? {
        guard physicalMemoryBytes > 0 else { return nil }

        var bestFit: CuratedModel?
        for curatedModel in curatedModels where curatedModel.recommendedRAMBytes <= physicalMemoryBytes {
            guard installedModels.contains(where: { installedModel($0, matches: curatedModel) }) else {
                continue
            }

            if bestFit == nil || curatedModel.recommendedRAMBytes > bestFit!.recommendedRAMBytes {
                bestFit = curatedModel
            }
        }
        return bestFit
    }

    public static func installedModel(
        _ installedModel: InstalledModel,
        matches curatedModel: CuratedModel
    ) -> Bool {
        installedModel.source == .curated
            && installedModel.hfRepo == curatedModel.hfRepo
            && installedModel.hfFilename == curatedModel.hfFilename
    }
}
