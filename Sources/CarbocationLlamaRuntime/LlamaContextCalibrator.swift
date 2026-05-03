import CarbocationLocalLLM
import Foundation
import llama

private enum LlamaContextCalibrationError: Error, LocalizedError {
    case noSupportedContext([LlamaContextCalibrationProbe])

    var errorDescription: String? {
        switch self {
        case .noSupportedContext:
            return "Calibration could not initialize a context at any probed size."
        }
    }
}

public extension LlamaEngine {
    static func contextCalibrationRuntimeFingerprint(
        configuration: LlamaEngineConfiguration = LlamaEngineConfiguration()
    ) -> LlamaContextCalibrationRuntimeFingerprint {
        LlamaContextCalibrationRuntimeFingerprint(
            platform: contextCalibrationPlatformName(),
            gpuLayerCount: Int(configuration.gpuLayerCount),
            useMemoryMap: configuration.useMemoryMap,
            batchSizeLimit: configuration.batchSizeLimit,
            threadCount: Int(effectiveThreadCount(for: configuration)),
            algorithmVersion: LlamaContextCalibrationAlgorithm.version
        )
    }

    func contextCalibrationRuntimeFingerprint() -> LlamaContextCalibrationRuntimeFingerprint {
        Self.contextCalibrationRuntimeFingerprint(configuration: configuration)
    }

    func calibrateContext(
        model installed: InstalledModel,
        from root: URL,
        store: LlamaContextCalibrationStore = .shared,
        onProgress: @escaping @Sendable (LlamaContextCalibrationProgress) async -> Void = { _ in }
    ) async throws -> LlamaContextCalibrationRecord {
        try await LlamaContextCalibrator.calibrate(
            model: installed,
            root: root,
            configuration: configuration,
            store: store,
            onProgress: onProgress
        )
    }

    static func contextCalibrationPlatformName() -> String {
#if os(iOS)
        "iOS"
#elseif os(macOS)
        "macOS"
#elseif os(tvOS)
        "tvOS"
#elseif os(watchOS)
        "watchOS"
#elseif os(visionOS)
        "visionOS"
#else
        "unknown"
#endif
    }

    static func effectiveThreadCount(for configuration: LlamaEngineConfiguration) -> Int32 {
        configuration.threadCount
            ?? Int32(max(1, ProcessInfo.processInfo.activeProcessorCount / 2))
    }
}

private enum LlamaContextCalibrator {
    static func calibrate(
        model installed: InstalledModel,
        root: URL,
        configuration: LlamaEngineConfiguration,
        store: LlamaContextCalibrationStore,
        onProgress: @escaping @Sendable (LlamaContextCalibrationProgress) async -> Void
    ) async throws -> LlamaContextCalibrationRecord {
        try Task.checkCancellation()

        let runtime = LlamaEngine.contextCalibrationRuntimeFingerprint(configuration: configuration)
        let key = store.key(for: installed, runtime: runtime)
        let modelURL = installed.weightsURL(in: root)
        let descriptor = LlamaModelDescriptor(model: installed, root: root)

        await onProgress(LlamaContextCalibrationProgress(
            phase: .loadingModel,
            message: "Loading \(descriptor.displayName ?? descriptor.filename)"
        ))

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = configuration.gpuLayerCount
        modelParams.use_mmap = configuration.useMemoryMap
        if configuration.gpuLayerCount <= 0 {
            modelParams.configureForCPUOnly()
        }

        guard let loadedModel = modelURL.path.withCString({ cPath in
            llama_model_load_from_file(cPath, modelParams)
        }) else {
            throw LLMEngineError.modelLoadFailed("llama_model_load_from_file returned null")
        }
        defer { llama_model_free(loadedModel) }

        try Task.checkCancellation()

        let probedTrainingContext = Int(llama_model_n_ctx_train(loadedModel))
        let knownTrainingContext = installed.contextLength > 0
            ? installed.contextLength
            : max(0, probedTrainingContext)
        let upperBound = knownTrainingContext > 0
            ? knownTrainingContext
            : LlamaContextPolicy.unknownTrainingFallback
        let candidates = LlamaContextCalibrationAlgorithm.powerOfTwoTiers(upTo: upperBound)
        let totalProbeCount = LlamaContextCalibrationAlgorithm.maximumProbeCount(
            candidateCount: candidates.count
        )
        let threads = LlamaEngine.effectiveThreadCount(for: configuration)
        var completedProbeCount = 0
        var lastSuccessfulContext: Int?

        let search = try await LlamaContextCalibrationAlgorithm.search(candidates: candidates) { context in
            try Task.checkCancellation()
            await onProgress(LlamaContextCalibrationProgress(
                phase: .probing,
                currentContext: context,
                lastSuccessfulContext: lastSuccessfulContext,
                completedProbeCount: completedProbeCount,
                totalProbeCount: totalProbeCount,
                message: "Probing \(context.formatted()) tokens"
            ))

            let succeeded = canInitializeContext(
                model: loadedModel,
                contextSize: context,
                batchSizeLimit: configuration.batchSizeLimit,
                threads: threads
            )

            completedProbeCount += 1
            if succeeded {
                lastSuccessfulContext = context
            }

            await onProgress(LlamaContextCalibrationProgress(
                phase: succeeded ? .probeSucceeded : .probeFailed,
                currentContext: context,
                lastSuccessfulContext: lastSuccessfulContext,
                completedProbeCount: completedProbeCount,
                totalProbeCount: totalProbeCount,
                message: succeeded
                    ? "\(context.formatted()) tokens succeeded"
                    : "\(context.formatted()) tokens failed"
            ))
            return succeeded
        }

        try Task.checkCancellation()

        guard let maximumSupportedContext = search.maximumSupportedContext else {
            throw LlamaContextCalibrationError.noSupportedContext(search.probes)
        }

        await onProgress(LlamaContextCalibrationProgress(
            phase: .saving,
            currentContext: maximumSupportedContext,
            lastSuccessfulContext: maximumSupportedContext,
            completedProbeCount: completedProbeCount,
            totalProbeCount: totalProbeCount,
            message: "Saving calibrated context"
        ))

        let now = Date()
        let record = LlamaContextCalibrationRecord(
            key: key,
            maximumSupportedContext: maximumSupportedContext,
            probedTiers: search.probes,
            createdAt: store.record(for: key)?.createdAt ?? now,
            updatedAt: now
        )
        store.save(record)

        await onProgress(LlamaContextCalibrationProgress(
            phase: .completed,
            currentContext: maximumSupportedContext,
            lastSuccessfulContext: maximumSupportedContext,
            completedProbeCount: completedProbeCount,
            totalProbeCount: totalProbeCount,
            message: "Calibrated to \(maximumSupportedContext.formatted()) tokens"
        ))

        return record
    }

    private static func canInitializeContext(
        model: OpaquePointer,
        contextSize: Int,
        batchSizeLimit: Int,
        threads: Int32
    ) -> Bool {
        let batchCandidates = LlamaEngine.contextBatchCandidates(
            contextSize: contextSize,
            batchSizeLimit: batchSizeLimit
        )
        for batchSize in batchCandidates {
            let params = LlamaEngine.contextParams(
                contextSize: contextSize,
                batchSize: batchSize,
                threads: threads
            )
            if let context = llama_init_from_model(model, params) {
                llama_free(context)
                return true
            }
        }
        return false
    }
}
