import CarbocationLocalLLM
import Foundation
import llama

#if canImport(Metal)
import Metal
#endif

private enum LlamaContextCalibrationError: Error, LocalizedError {
    case noSupportedContext([LlamaContextCalibrationProbe])

    var errorDescription: String? {
        switch self {
        case .noSupportedContext:
            return "Calibration could not run a decode probe at any probed context size."
        }
    }
}

private struct LlamaContextDecodeProbeOutcome {
    var succeeded: Bool
    var shouldReloadModelBeforeNextProbe: Bool
}

struct LlamaContextMemoryGuardrail {
    struct ModelProfile: Hashable {
        var modelTensorBytes: UInt64
        var layerCount: Int
        var embeddingCount: Int
        var headCount: Int
        var kvHeadCount: Int
        var keyCacheBytesPerElement: Double
        var valueCacheBytesPerElement: Double
    }

    struct Estimate: Hashable {
        var modelTensorBytes: UInt64
        var modelReserveBytes: UInt64
        var kvCacheBytes: UInt64
        var decodeWorkspaceBytes: UInt64
        var safetyMarginBytes: UInt64
        var incrementalBytes: UInt64
        var requiredBytes: UInt64
        var totalBytes: UInt64
        var budgetBytes: UInt64
    }

    static let minimumDecodeWorkspaceBytes: UInt64 = 512 * 1_024 * 1_024
    static let maximumDecodeWorkspaceBytes: UInt64 = 8 * 1_024 * 1_024 * 1_024
    static let minimumSafetyMarginBytes: UInt64 = 512 * 1_024 * 1_024
    static let maximumSafetyMarginBytes: UInt64 = 8 * 1_024 * 1_024 * 1_024
    static let incrementalSafetyMarginRatio = 0.20
    static let maximumModelReserveBudgetFraction = 0.50
    static let cpuBudgetUtilization = 0.75
    static let acceleratedBudgetUtilization = 0.85
    static let minimumAcceleratedPhysicalMemoryFraction = 0.75

    static func profile(
        model: OpaquePointer,
        contextParams: llama_context_params
    ) -> ModelProfile {
        let headCount = max(1, Int(llama_model_n_head(model)))
        let kvHeadCount = max(1, Int(llama_model_n_head_kv(model)))
        return ModelProfile(
            modelTensorBytes: llama_model_size(model),
            layerCount: max(1, Int(llama_model_n_layer(model))),
            embeddingCount: max(1, Int(llama_model_n_embd(model))),
            headCount: headCount,
            kvHeadCount: min(max(1, kvHeadCount), headCount),
            keyCacheBytesPerElement: cacheElementSize(for: contextParams.type_k),
            valueCacheBytesPerElement: cacheElementSize(for: contextParams.type_v)
        )
    }

    static func estimate(
        profile: ModelProfile,
        contextSize: Int,
        batchSize: Int,
        budgetBytes: UInt64
    ) -> Estimate {
        let groupedEmbeddingCount = ceil(
            Double(profile.embeddingCount)
            * Double(profile.kvHeadCount)
            / Double(max(1, profile.headCount))
        )
        let kvBytesPerToken = Double(profile.layerCount)
            * groupedEmbeddingCount
            * (profile.keyCacheBytesPerElement + profile.valueCacheBytesPerElement)
        let kvCacheBytes = byteCount(kvBytesPerToken * Double(max(1, contextSize)))

        let batchActivationBytes = byteCount(
            Double(max(1, batchSize))
            * Double(profile.embeddingCount)
            * Double(profile.layerCount)
            * 4
        )
        let decodeWorkspaceBytes = min(
            maximumDecodeWorkspaceBytes,
            max(minimumDecodeWorkspaceBytes, batchActivationBytes)
        )
        let safetyBasis = safeAdd(kvCacheBytes, decodeWorkspaceBytes)
        let safetyMarginBytes = min(
            maximumSafetyMarginBytes,
            max(minimumSafetyMarginBytes, byteCount(Double(safetyBasis) * incrementalSafetyMarginRatio))
        )
        let incrementalBytes = safeAdd(
            kvCacheBytes,
            safeAdd(decodeWorkspaceBytes, safetyMarginBytes)
        )
        let modelReserveBytes = min(
            profile.modelTensorBytes,
            byteCount(Double(budgetBytes) * maximumModelReserveBudgetFraction)
        )
        let requiredBytes = safeAdd(modelReserveBytes, incrementalBytes)
        let totalBytes = safeAdd(
            profile.modelTensorBytes,
            incrementalBytes
        )

        return Estimate(
            modelTensorBytes: profile.modelTensorBytes,
            modelReserveBytes: modelReserveBytes,
            kvCacheBytes: kvCacheBytes,
            decodeWorkspaceBytes: decodeWorkspaceBytes,
            safetyMarginBytes: safetyMarginBytes,
            incrementalBytes: incrementalBytes,
            requiredBytes: requiredBytes,
            totalBytes: totalBytes,
            budgetBytes: budgetBytes
        )
    }

    static func allowsProbe(_ estimate: Estimate) -> Bool {
        // The model has already loaded before calibration chooses a context tier.
        // Count a bounded reserve for resident model resources without requiring
        // the full tensor size to fit again as if it were a fresh allocation.
        estimate.requiredBytes <= estimate.budgetBytes
    }

    static func defaultBudgetBytes(gpuLayerCount: Int32) -> UInt64 {
        var availableBytes = ProcessInfo.processInfo.physicalMemory

#if canImport(Metal)
        if gpuLayerCount != 0,
           let recommendedWorkingSet = MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize,
           recommendedWorkingSet > 0 {
            let physicalFloor = byteCount(
                Double(ProcessInfo.processInfo.physicalMemory)
                * minimumAcceleratedPhysicalMemoryFraction
            )
            availableBytes = max(recommendedWorkingSet, physicalFloor)
        }
#endif

        let utilization = gpuLayerCount == 0 ? cpuBudgetUtilization : acceleratedBudgetUtilization
        return byteCount(Double(availableBytes) * utilization)
    }

    private static func cacheElementSize(for type: ggml_type) -> Double {
        let blockSize = ggml_blck_size(type)
        guard blockSize > 0 else { return 2 }
        let rowSize = ggml_row_size(type, blockSize)
        guard rowSize > 0 else { return 2 }
        return Double(rowSize) / Double(blockSize)
    }

    private static func byteCount(_ value: Double) -> UInt64 {
        guard value.isFinite, value > 0 else { return 0 }
        guard value < Double(UInt64.max) else { return UInt64.max }
        return UInt64(value.rounded(.up))
    }

    private static func safeAdd(_ lhs: UInt64, _ rhs: UInt64) -> UInt64 {
        let (result, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? UInt64.max : result
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

        var loadedModel: OpaquePointer?
        var loadedVocabulary: OpaquePointer?
        defer {
            if let loadedModel {
                llama_model_free(loadedModel)
            }
        }

        func reloadModel() throws {
            if let model = loadedModel {
                llama_model_free(model)
            }
            loadedModel = nil
            loadedVocabulary = nil

            let reloadedModel = try Self.loadModel(at: modelURL, params: modelParams)
            do {
                loadedVocabulary = try Self.vocabulary(for: reloadedModel)
                loadedModel = reloadedModel
            } catch {
                llama_model_free(reloadedModel)
                throw error
            }
        }

        try reloadModel()
        guard let initialModel = loadedModel else {
            throw LLMEngineError.modelLoadFailed("llama_model_load_from_file returned null")
        }

        try Task.checkCancellation()

        let probedTrainingContext = Int(llama_model_n_ctx_train(initialModel))
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
        var modelNeedsReload = false

        let search = try await LlamaContextCalibrationAlgorithm.search(candidates: candidates) { context in
            try Task.checkCancellation()
            if modelNeedsReload {
                try reloadModel()
                modelNeedsReload = false
            }

            await onProgress(LlamaContextCalibrationProgress(
                phase: .probing,
                currentContext: context,
                lastSuccessfulContext: lastSuccessfulContext,
                completedProbeCount: completedProbeCount,
                totalProbeCount: totalProbeCount,
                message: "Probing \(context.formatted()) tokens"
            ))

            guard let model = loadedModel,
                  let vocabulary = loadedVocabulary
            else {
                throw LLMEngineError.modelLoadFailed("llama_model_load_from_file returned null")
            }

            let outcome = canRunDecodeProbe(
                model: model,
                vocabulary: vocabulary,
                contextSize: context,
                batchSizeLimit: configuration.batchSizeLimit,
                threads: threads,
                memoryBudgetBytes: LlamaContextMemoryGuardrail.defaultBudgetBytes(
                    gpuLayerCount: configuration.gpuLayerCount
                )
            )
            let succeeded = outcome.succeeded

            completedProbeCount += 1
            if succeeded {
                lastSuccessfulContext = context
            } else if outcome.shouldReloadModelBeforeNextProbe {
                modelNeedsReload = true
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

    private static func loadModel(
        at url: URL,
        params: llama_model_params
    ) throws -> OpaquePointer {
        guard let model = url.path.withCString({ cPath in
            llama_model_load_from_file(cPath, params)
        }) else {
            throw LLMEngineError.modelLoadFailed("llama_model_load_from_file returned null")
        }
        return model
    }

    private static func vocabulary(for model: OpaquePointer) throws -> OpaquePointer {
        guard let vocabulary = llama_model_get_vocab(model) else {
            throw LLMEngineError.modelLoadFailed("llama_model_get_vocab returned null")
        }
        return vocabulary
    }

    private static func canRunDecodeProbe(
        model: OpaquePointer,
        vocabulary: OpaquePointer,
        contextSize: Int,
        batchSizeLimit: Int,
        threads: Int32,
        memoryBudgetBytes: UInt64
    ) -> LlamaContextDecodeProbeOutcome {
        let probeTokens = decodeProbeTokens(vocabulary: vocabulary)
        guard !probeTokens.isEmpty else {
            return LlamaContextDecodeProbeOutcome(
                succeeded: false,
                shouldReloadModelBeforeNextProbe: false
            )
        }

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
            let estimate = LlamaContextMemoryGuardrail.estimate(
                profile: LlamaContextMemoryGuardrail.profile(
                    model: model,
                    contextParams: params
                ),
                contextSize: contextSize,
                batchSize: batchSize,
                budgetBytes: memoryBudgetBytes
            )
            guard LlamaContextMemoryGuardrail.allowsProbe(estimate) else {
                continue
            }

            if let context = llama_init_from_model(model, params) {
                var tokens = probeTokens
                let decodeResult = tokens.withUnsafeMutableBufferPointer { buffer -> Int32 in
                    guard let baseAddress = buffer.baseAddress else { return -1 }
                    let batch = llama_batch_get_one(baseAddress, Int32(buffer.count))
                    return llama_decode(context, batch)
                }
                llama_free(context)
                if decodeResult == 0 {
                    return LlamaContextDecodeProbeOutcome(
                        succeeded: true,
                        shouldReloadModelBeforeNextProbe: false
                    )
                }
                return LlamaContextDecodeProbeOutcome(
                    succeeded: false,
                    shouldReloadModelBeforeNextProbe: true
                )
            }
        }
        return LlamaContextDecodeProbeOutcome(
            succeeded: false,
            shouldReloadModelBeforeNextProbe: false
        )
    }

    private static func decodeProbeTokens(vocabulary: OpaquePointer) -> [llama_token] {
        let vocabularySize = llama_vocab_n_tokens(vocabulary)
        guard vocabularySize > 0 else { return [] }

        let candidates = [
            llama_vocab_bos(vocabulary),
            llama_vocab_eos(vocabulary),
            llama_vocab_eot(vocabulary),
            llama_vocab_nl(vocabulary),
            llama_token(0)
        ]
        for token in candidates where token >= 0 && token < vocabularySize {
            return [token]
        }

        return [0]
    }
}
