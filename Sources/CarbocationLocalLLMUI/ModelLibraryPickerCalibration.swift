import CarbocationLocalLLM
import Foundation

public struct ModelLibraryPickerCalibrationAdapter {
    public var store: LlamaContextCalibrationStore
    public var runtimeFingerprint: LlamaContextCalibrationRuntimeFingerprint
    public var onCalibrationStarted: @MainActor (_ model: InstalledModel) -> Void
    public var onCalibrationCompleted: @MainActor (
        _ model: InstalledModel,
        _ record: LlamaContextCalibrationRecord
    ) -> Void
    public var calibrate: @MainActor (
        _ model: InstalledModel,
        _ onProgress: @escaping @MainActor (LlamaContextCalibrationProgress) -> Void
    ) async throws -> LlamaContextCalibrationRecord

    public init(
        store: LlamaContextCalibrationStore = .shared,
        runtimeFingerprint: LlamaContextCalibrationRuntimeFingerprint,
        onCalibrationStarted: @escaping @MainActor (_ model: InstalledModel) -> Void = { _ in },
        onCalibrationCompleted: @escaping @MainActor (
            _ model: InstalledModel,
            _ record: LlamaContextCalibrationRecord
        ) -> Void = { _, _ in },
        calibrate: @escaping @MainActor (
            _ model: InstalledModel,
            _ onProgress: @escaping @MainActor (LlamaContextCalibrationProgress) -> Void
        ) async throws -> LlamaContextCalibrationRecord
    ) {
        self.store = store
        self.runtimeFingerprint = runtimeFingerprint
        self.onCalibrationStarted = onCalibrationStarted
        self.onCalibrationCompleted = onCalibrationCompleted
        self.calibrate = calibrate
    }

    @MainActor
    public func runCalibration(
        _ model: InstalledModel,
        onProgress: @escaping @MainActor (LlamaContextCalibrationProgress) -> Void
    ) async throws -> LlamaContextCalibrationRecord {
        onCalibrationStarted(model)
        let record = try await calibrate(model, onProgress)
        onCalibrationCompleted(model, record)
        return record
    }
}

public struct ModelLibraryPickerContextCalibrationSummary: Hashable, Sendable {
    public var context: Int
    public var isCalibrated: Bool

    public init(context: Int, isCalibrated: Bool) {
        self.context = context
        self.isCalibrated = isCalibrated
    }

    public var actionTitle: String {
        isCalibrated ? "Recalibrate" : "Calibrate"
    }

    public var statusTitle: String {
        isCalibrated ? "Calibrated" : "Uncalibrated"
    }

    public var displayText: String {
        let prefix = isCalibrated ? "max context" : "auto context"
        return "\(prefix) \(context.formatted()) (\(statusTitle))"
    }
}

public enum ModelLibraryPickerContextCalibrationPresentation {
    public static func summary(
        for model: InstalledModel,
        record: LlamaContextCalibrationRecord?,
        defaultAutoCap: Int = LlamaContextPolicy.defaultAutoCap
    ) -> ModelLibraryPickerContextCalibrationSummary {
        if let record {
            return ModelLibraryPickerContextCalibrationSummary(
                context: record.maximumSupportedContext,
                isCalibrated: true
            )
        }

        return ModelLibraryPickerContextCalibrationSummary(
            context: LlamaContextPolicy.autoContext(
                for: model.contextLength,
                autoCap: defaultAutoCap
            ),
            isCalibrated: false
        )
    }
}
