import CarbocationLocalLLM
import Foundation

public struct ModelLibraryPickerCalibrationAdapter {
    public var store: LlamaContextCalibrationStore
    public var runtimeFingerprint: LlamaContextCalibrationRuntimeFingerprint
    public var calibrate: @MainActor (
        _ model: InstalledModel,
        _ onProgress: @escaping @MainActor (LlamaContextCalibrationProgress) -> Void
    ) async throws -> LlamaContextCalibrationRecord

    public init(
        store: LlamaContextCalibrationStore = .shared,
        runtimeFingerprint: LlamaContextCalibrationRuntimeFingerprint,
        calibrate: @escaping @MainActor (
            _ model: InstalledModel,
            _ onProgress: @escaping @MainActor (LlamaContextCalibrationProgress) -> Void
        ) async throws -> LlamaContextCalibrationRecord
    ) {
        self.store = store
        self.runtimeFingerprint = runtimeFingerprint
        self.calibrate = calibrate
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
        "context \(context.formatted()) (\(statusTitle))"
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
