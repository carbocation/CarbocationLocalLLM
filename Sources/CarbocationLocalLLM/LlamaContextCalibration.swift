import Foundation

public enum LlamaContextCalibrationStatus: String, Codable, Hashable, Sendable {
    case completed
}

public struct LlamaContextCalibrationModelFingerprint: Codable, Hashable, Sendable {
    public var modelID: UUID
    public var filename: String
    public var sizeBytes: Int64
    public var sha256: String?
    public var trainingContext: Int

    public init(
        modelID: UUID,
        filename: String,
        sizeBytes: Int64,
        sha256: String?,
        trainingContext: Int
    ) {
        self.modelID = modelID
        self.filename = filename
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
        self.trainingContext = trainingContext
    }

    public init(model: InstalledModel) {
        self.init(
            modelID: model.id,
            filename: model.filename,
            sizeBytes: model.sizeBytes,
            sha256: model.sha256,
            trainingContext: model.contextLength
        )
    }
}

public struct LlamaContextCalibrationRuntimeFingerprint: Codable, Hashable, Sendable {
    public var platform: String
    public var gpuLayerCount: Int
    public var useMemoryMap: Bool
    public var batchSizeLimit: Int
    public var threadCount: Int
    public var algorithmVersion: Int

    public init(
        platform: String,
        gpuLayerCount: Int,
        useMemoryMap: Bool,
        batchSizeLimit: Int,
        threadCount: Int,
        algorithmVersion: Int = LlamaContextCalibrationAlgorithm.version
    ) {
        self.platform = platform
        self.gpuLayerCount = gpuLayerCount
        self.useMemoryMap = useMemoryMap
        self.batchSizeLimit = batchSizeLimit
        self.threadCount = threadCount
        self.algorithmVersion = algorithmVersion
    }
}

public struct LlamaContextCalibrationKey: Codable, Hashable, Sendable {
    public var deviceID: String
    public var model: LlamaContextCalibrationModelFingerprint
    public var runtime: LlamaContextCalibrationRuntimeFingerprint

    public init(
        deviceID: String,
        model: LlamaContextCalibrationModelFingerprint,
        runtime: LlamaContextCalibrationRuntimeFingerprint
    ) {
        self.deviceID = deviceID
        self.model = model
        self.runtime = runtime
    }

    public var storageKey: String {
        [
            "v1",
            "device=\(deviceID)",
            "modelID=\(model.modelID.uuidString)",
            "filename=\(model.filename)",
            "size=\(model.sizeBytes)",
            "sha256=\(model.sha256 ?? "-")",
            "trainingContext=\(model.trainingContext)",
            "platform=\(runtime.platform)",
            "gpuLayers=\(runtime.gpuLayerCount)",
            "mmap=\(runtime.useMemoryMap)",
            "batchLimit=\(runtime.batchSizeLimit)",
            "threads=\(runtime.threadCount)",
            "algorithm=\(runtime.algorithmVersion)"
        ].joined(separator: "|")
    }
}

public struct LlamaContextCalibrationProbe: Codable, Hashable, Sendable {
    public var context: Int
    public var succeeded: Bool

    public init(context: Int, succeeded: Bool) {
        self.context = context
        self.succeeded = succeeded
    }
}

public struct LlamaContextCalibrationRecord: Codable, Hashable, Sendable {
    public var key: LlamaContextCalibrationKey
    public var maximumSupportedContext: Int
    public var probedTiers: [LlamaContextCalibrationProbe]
    public var status: LlamaContextCalibrationStatus
    public var createdAt: Date
    public var updatedAt: Date

    public init(
        key: LlamaContextCalibrationKey,
        maximumSupportedContext: Int,
        probedTiers: [LlamaContextCalibrationProbe],
        status: LlamaContextCalibrationStatus = .completed,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.key = key
        self.maximumSupportedContext = maximumSupportedContext
        self.probedTiers = probedTiers
        self.status = status
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

public enum LlamaContextCalibrationProgressPhase: String, Codable, Hashable, Sendable {
    case loadingModel
    case probing
    case probeSucceeded
    case probeFailed
    case saving
    case completed
    case cancelled
}

public struct LlamaContextCalibrationProgress: Hashable, Sendable {
    public var phase: LlamaContextCalibrationProgressPhase
    public var currentContext: Int?
    public var lastSuccessfulContext: Int?
    public var completedProbeCount: Int
    public var totalProbeCount: Int
    public var message: String?

    public init(
        phase: LlamaContextCalibrationProgressPhase,
        currentContext: Int? = nil,
        lastSuccessfulContext: Int? = nil,
        completedProbeCount: Int = 0,
        totalProbeCount: Int = 0,
        message: String? = nil
    ) {
        self.phase = phase
        self.currentContext = currentContext
        self.lastSuccessfulContext = lastSuccessfulContext
        self.completedProbeCount = completedProbeCount
        self.totalProbeCount = totalProbeCount
        self.message = message
    }

    public var fractionCompleted: Double? {
        guard totalProbeCount > 0 else { return nil }
        return min(1, max(0, Double(completedProbeCount) / Double(totalProbeCount)))
    }
}

public struct LlamaContextCalibrationSearchResult: Hashable, Sendable {
    public var maximumSupportedContext: Int?
    public var probes: [LlamaContextCalibrationProbe]

    public init(
        maximumSupportedContext: Int?,
        probes: [LlamaContextCalibrationProbe]
    ) {
        self.maximumSupportedContext = maximumSupportedContext
        self.probes = probes
    }
}

public enum LlamaContextCalibrationAlgorithm {
    public static let version = 1

    public static func powerOfTwoTiers(
        upTo upperBound: Int,
        minimumContext: Int = LlamaContextPolicy.minimumContext
    ) -> [Int] {
        let minimum = max(1, minimumContext)
        let clampedUpperBound = max(minimum, upperBound)
        var tiers: [Int] = []
        var next = minimum
        while next <= clampedUpperBound {
            tiers.append(next)
            guard next <= Int.max / 2 else { break }
            next *= 2
        }
        return tiers
    }

    public static func maximumProbeCount(candidateCount: Int) -> Int {
        guard candidateCount > 1 else { return max(1, candidateCount) }
        return Int(ceil(log2(Double(candidateCount)))) + 1
    }

    public static func search(
        candidates: [Int],
        probe: (Int) async throws -> Bool
    ) async throws -> LlamaContextCalibrationSearchResult {
        guard !candidates.isEmpty else {
            return LlamaContextCalibrationSearchResult(maximumSupportedContext: nil, probes: [])
        }

        var low = 0
        var high = candidates.count - 1
        var best: Int?
        var probes: [LlamaContextCalibrationProbe] = []

        while low <= high {
            let index = low + ((high - low) / 2)
            let context = candidates[index]
            let succeeded = try await probe(context)
            probes.append(LlamaContextCalibrationProbe(context: context, succeeded: succeeded))

            if succeeded {
                best = context
                low = index + 1
            } else {
                high = index - 1
            }
        }

        return LlamaContextCalibrationSearchResult(
            maximumSupportedContext: best,
            probes: probes
        )
    }
}

public final class LlamaContextCalibrationStore: @unchecked Sendable {
    public static let recordsDefaultsKey = "llama.contextCalibration.records.v1"
    public static let deviceIDDefaultsKey = "llama.contextCalibration.deviceID.v1"

    public static var shared: LlamaContextCalibrationStore {
        LlamaContextCalibrationStore()
    }

    private let defaults: UserDefaults
    private let recordsKey: String
    private let deviceIDKey: String
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    public init(
        defaults: UserDefaults = ModelStorage.sharedSettingsDefaults(),
        recordsKey: String = LlamaContextCalibrationStore.recordsDefaultsKey,
        deviceIDKey: String = LlamaContextCalibrationStore.deviceIDDefaultsKey
    ) {
        self.defaults = defaults
        self.recordsKey = recordsKey
        self.deviceIDKey = deviceIDKey
    }

    public func deviceID() -> String {
        if let existing = defaults.string(forKey: deviceIDKey),
           !existing.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return existing
        }

        let generated = UUID().uuidString
        defaults.set(generated, forKey: deviceIDKey)
        return generated
    }

    public func key(
        for model: InstalledModel,
        runtime: LlamaContextCalibrationRuntimeFingerprint
    ) -> LlamaContextCalibrationKey {
        LlamaContextCalibrationKey(
            deviceID: deviceID(),
            model: LlamaContextCalibrationModelFingerprint(model: model),
            runtime: runtime
        )
    }

    public func record(
        for model: InstalledModel,
        runtime: LlamaContextCalibrationRuntimeFingerprint
    ) -> LlamaContextCalibrationRecord? {
        record(for: key(for: model, runtime: runtime))
    }

    public func record(for key: LlamaContextCalibrationKey) -> LlamaContextCalibrationRecord? {
        records()[key.storageKey]
    }

    public func save(_ record: LlamaContextCalibrationRecord) {
        var allRecords = records()
        allRecords[record.key.storageKey] = record
        saveRecords(allRecords)
    }

    public func removeRecord(for key: LlamaContextCalibrationKey) {
        var allRecords = records()
        allRecords.removeValue(forKey: key.storageKey)
        saveRecords(allRecords)
    }

    public func records() -> [String: LlamaContextCalibrationRecord] {
        guard let data = defaults.data(forKey: recordsKey) else {
            return [:]
        }
        return (try? decoder.decode([String: LlamaContextCalibrationRecord].self, from: data)) ?? [:]
    }

    private func saveRecords(_ records: [String: LlamaContextCalibrationRecord]) {
        guard let data = try? encoder.encode(records) else { return }
        defaults.set(data, forKey: recordsKey)
    }
}
