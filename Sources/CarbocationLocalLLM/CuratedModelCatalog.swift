import Foundation

public struct CuratedModelReference: Codable, Hashable, Sendable {
    public var hfRepo: String
    public var hfFilename: String

    public init(hfRepo: String, hfFilename: String) {
        self.hfRepo = hfRepo
        self.hfFilename = hfFilename
    }
}

public struct CuratedModel: Identifiable, Hashable, Sendable {
    public let id: String
    public let displayName: String
    public let subtitle: String
    public let hfRepo: String
    public let hfFilename: String
    public let approxSizeBytes: Int64
    public let contextLength: Int
    public let quantization: String
    public let recommendedRAMGB: Int
    public let sha256: String?
    public let samplingDefaults: LLMSamplingDefaults?

    public init(
        id: String,
        displayName: String,
        subtitle: String,
        hfRepo: String,
        hfFilename: String,
        approxSizeBytes: Int64,
        contextLength: Int,
        quantization: String,
        recommendedRAMGB: Int,
        sha256: String?,
        samplingDefaults: LLMSamplingDefaults? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.subtitle = subtitle
        self.hfRepo = hfRepo
        self.hfFilename = hfFilename
        self.approxSizeBytes = approxSizeBytes
        self.contextLength = contextLength
        self.quantization = quantization
        self.recommendedRAMGB = recommendedRAMGB
        self.sha256 = sha256
        self.samplingDefaults = samplingDefaults
    }

    public init(
        id: String,
        displayName: String,
        subtitle: String,
        hfRepo: String,
        hfFilename: String,
        approxSizeBytes: Int64,
        contextLength: Int,
        quantization: String,
        recommendedRAMGB: Int,
        sha256: String?
    ) {
        self.init(
            id: id,
            displayName: displayName,
            subtitle: subtitle,
            hfRepo: hfRepo,
            hfFilename: hfFilename,
            approxSizeBytes: approxSizeBytes,
            contextLength: contextLength,
            quantization: quantization,
            recommendedRAMGB: recommendedRAMGB,
            sha256: sha256,
            samplingDefaults: nil
        )
    }

    public var recommendedRAMBytes: UInt64 {
        UInt64(recommendedRAMGB) * 1_073_741_824
    }

    public var reference: CuratedModelReference {
        CuratedModelReference(hfRepo: hfRepo, hfFilename: hfFilename)
    }

    public func matches(installedModel: InstalledModel) -> Bool {
        installedModel.source == .curated
            && installedModel.hfRepo == hfRepo
            && installedModel.hfFilename == hfFilename
    }
}

public enum CuratedModelCatalog {
    public static let all: [CuratedModel] = [
        CuratedModel(
            id: "gemma-4-e2b-it-q4km",
            displayName: "Gemma 4 E2B Instruct (Q4_K_M)",
            subtitle: "Smallest practical on-device option. Fast local fallback for low-memory devices.",
            hfRepo: "bartowski/google_gemma-4-E2B-it-GGUF",
            hfFilename: "google_gemma-4-E2B-it-Q4_K_M.gguf",
            approxSizeBytes: 3_500_000_000,
            contextLength: 131_072,
            quantization: "Q4_K_M",
            recommendedRAMGB: 8,
            sha256: nil,
            samplingDefaults: LLMSamplingDefaults(
                temperature: 1.0,
                topP: 0.95,
                topK: 64
            )
        ),
        CuratedModel(
            id: "qwen3.5-9b-instruct-q4km",
            displayName: "Qwen3.5 9B Instruct (Q4_K_M)",
            subtitle: "Balanced quality jump. Strong extraction and reasoning for 16 GB devices.",
            hfRepo: "bartowski/Qwen_Qwen3.5-9B-GGUF",
            hfFilename: "Qwen_Qwen3.5-9B-Q4_K_M.gguf",
            approxSizeBytes: 5_300_000_000,
            contextLength: 32_768,
            quantization: "Q4_K_M",
            recommendedRAMGB: 16,
            sha256: nil
        ),
        CuratedModel(
            id: "gemma-4-26b-a4b-it-q4km",
            displayName: "Gemma 4 26B A4B Instruct (Q4_K_M)",
            subtitle: "MoE mid-tier: 26B total / about 4B active. High quality at fast speeds.",
            hfRepo: "bartowski/google_gemma-4-26B-A4B-it-GGUF",
            hfFilename: "google_gemma-4-26B-A4B-it-Q4_K_M.gguf",
            approxSizeBytes: 16_900_000_000,
            contextLength: 262_144,
            quantization: "Q4_K_M",
            recommendedRAMGB: 32,
            sha256: nil,
            samplingDefaults: LLMSamplingDefaults(
                temperature: 1.0,
                topP: 0.95,
                topK: 64
            )
        ),
        CuratedModel(
            id: "qwen3.6-27b-dense-q4km",
            displayName: "Qwen3.6 27B Dense (Q4_K_M)",
            subtitle: "Dense high-memory Qwen option. Strong coding and reasoning; needs about 48 GB RAM.",
            hfRepo: "bartowski/Qwen_Qwen3.6-27B-GGUF",
            hfFilename: "Qwen_Qwen3.6-27B-Q4_K_M.gguf",
            approxSizeBytes: 17_500_000_000,
            contextLength: 262_144,
            quantization: "Q4_K_M",
            recommendedRAMGB: 48,
            sha256: nil,
            samplingDefaults: LLMSamplingDefaults(
                temperature: 0.7,
                topP: 0.8,
                topK: 20,
                minP: 0,
                presencePenalty: 1.5,
                repetitionPenalty: 1.0
            )
        )
    ]

    public static func entry(id: String, among models: [CuratedModel] = all) -> CuratedModel? {
        models.first { $0.id == id }
    }

    public static func recommendedModel(
        forPhysicalMemoryBytes physicalMemoryBytes: UInt64,
        among models: [CuratedModel] = all
    ) -> CuratedModel? {
        guard physicalMemoryBytes > 0 else { return nil }

        var bestFit: CuratedModel?
        for model in models where model.recommendedRAMBytes <= physicalMemoryBytes {
            if bestFit == nil || model.recommendedRAMBytes > bestFit!.recommendedRAMBytes {
                bestFit = model
            }
        }
        return bestFit
    }

    public static func entry(
        for installedModel: InstalledModel,
        among models: [CuratedModel] = all
    ) -> CuratedModel? {
        models.first { $0.matches(installedModel: installedModel) }
    }
}

public enum LLMSamplingDefaultsResolver {
    public static func resolvedDefaults(
        globalDefaults: LLMSamplingDefaults = .extractionSafe,
        curatedDefaults: LLMSamplingDefaults? = nil,
        appOverrides: LLMSamplingDefaults? = nil
    ) -> LLMSamplingDefaults {
        globalDefaults
            .merged(with: curatedDefaults)
            .merged(with: appOverrides)
    }

    public static func resolvedDefaults(
        globalDefaults: LLMSamplingDefaults = .extractionSafe,
        installedModel: InstalledModel?,
        curatedModels: [CuratedModel] = CuratedModelCatalog.all,
        appOverrides: [CuratedModelReference: LLMSamplingDefaults] = [:]
    ) -> LLMSamplingDefaults {
        guard let installedModel,
              let curatedModel = CuratedModelCatalog.entry(for: installedModel, among: curatedModels)
        else {
            return globalDefaults
        }

        return resolvedDefaults(
            globalDefaults: globalDefaults,
            curatedDefaults: curatedModel.samplingDefaults,
            appOverrides: appOverrides[curatedModel.reference]
        )
    }

    public static func resolvedOptions(
        globalDefaults: LLMSamplingDefaults = .extractionSafe,
        installedModel: InstalledModel?,
        curatedModels: [CuratedModel] = CuratedModelCatalog.all,
        appOverrides: [CuratedModelReference: LLMSamplingDefaults] = [:],
        requestOptions: GenerationOptions
    ) -> GenerationOptions {
        resolvedDefaults(
            globalDefaults: globalDefaults,
            installedModel: installedModel,
            curatedModels: curatedModels,
            appOverrides: appOverrides
        ).applying(to: requestOptions)
    }
}
