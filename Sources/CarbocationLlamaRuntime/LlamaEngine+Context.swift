import CarbocationLocalLLM
import Foundation
import llama

/// Describes an inference-context value that cannot be represented safely by
/// llama.cpp or by the runtime's signed token-position and batch-count APIs.
public enum LlamaContextConfigurationError: Error, LocalizedError, Hashable, Sendable {
    public enum Parameter: String, Hashable, Sendable {
        case requestedContext
        case batchSizeLimit
        case microBatchSizeLimit
    }

    case exceedsBackendLimit(parameter: Parameter, maximum: Int)

    public var errorDescription: String? {
        switch self {
        case .exceedsBackendLimit(let parameter, let maximum):
            return "Llama context option '\(parameter.rawValue)' must not exceed \(maximum)."
        }
    }
}

extension LlamaEngine {
    /// Matches llama.cpp's deployment-safe default physical batch size.
    static let defaultMicroBatchSizeLimit = 512
    /// Matches the runtime's bounded MTP draft-token window. Larger values can
    /// overflow llama.cpp's recurrent-memory snapshot arithmetic.
    static let maximumRecurrentStateSnapshots = 32
    /// Although llama context parameters are unsigned 32-bit values, token
    /// positions and submitted batch counts are signed 32-bit values.
    static let maximumBackendContextParameter = Int(Int32.max)

    static func usesGPUOffload(gpuLayerCount: Int32) -> Bool {
        gpuLayerCount != 0
    }

    static func clampedContextSize(requestedContext: Int, trainingContext: Int) -> Int {
        let normalizedTrainingContext = max(0, trainingContext)
        let requested = requestedContext > 0
            ? requestedContext
            : (normalizedTrainingContext > 0 ? normalizedTrainingContext : LlamaContextPolicy.unknownTrainingFallback)
        let modelUpperBound = normalizedTrainingContext > 0 ? normalizedTrainingContext : requested
        let platformUpperBound = min(platformMaximumContextSize, maximumBackendContextParameter)
        let upperBound = min(modelUpperBound, platformUpperBound)
        return max(LlamaContextPolicy.minimumContext, min(requested, upperBound))
    }

    /// Validates positive public context settings before model loading begins.
    /// Zero and negative values retain their existing automatic/minimum sentinel
    /// behavior and are normalized by the context builder.
    static func validateContextConfiguration(
        requestedContext: Int,
        batchSizeLimit: Int,
        microBatchSizeLimit: Int?
    ) throws {
        try validatePositiveContextParameter(
            requestedContext,
            parameter: .requestedContext
        )
        try validatePositiveContextParameter(
            batchSizeLimit,
            parameter: .batchSizeLimit
        )
        if let microBatchSizeLimit {
            try validatePositiveContextParameter(
                microBatchSizeLimit,
                parameter: .microBatchSizeLimit
            )
        }
    }

    private static func validatePositiveContextParameter(
        _ value: Int,
        parameter: LlamaContextConfigurationError.Parameter
    ) throws {
        guard value <= 0 || value <= maximumBackendContextParameter else {
            throw LlamaContextConfigurationError.exceedsBackendLimit(
                parameter: parameter,
                maximum: maximumBackendContextParameter
            )
        }
    }

    static func contextBatchCandidates(
        contextSize: Int,
        batchSizeLimit: Int,
        microBatchSizeLimit: Int? = nil
    ) -> [Int] {
        let logicalBatchSize = min(max(1, contextSize), max(1, batchSizeLimit))
        let requestedMicroBatchSize = microBatchSizeLimit ?? defaultMicroBatchSizeLimit
        let first = min(logicalBatchSize, max(1, requestedMicroBatchSize))
        var candidates = [first]
        var next = first
        while next > 1 {
            next = max(1, next / 2)
            if candidates.last != next {
                candidates.append(next)
            }
        }
        return candidates
    }

    static func contextParams(
        contextSize: Int,
        batchSize: Int,
        microBatchSize: Int? = nil,
        threads: Int32,
        recurrentStateSnapshots: Int = 0
    ) -> llama_context_params {
        var params = llama_context_default_params()
        let clampedContext = min(maximumBackendContextParameter, max(1, contextSize))
        let clampedBatch = min(clampedContext, max(1, batchSize))
        let requestedMicroBatch = microBatchSize ?? min(clampedBatch, defaultMicroBatchSizeLimit)
        let clampedMicroBatch = min(clampedBatch, max(1, requestedMicroBatch))
        params.n_ctx = UInt32(clampedContext)
        params.n_batch = UInt32(clampedBatch)
        params.n_ubatch = UInt32(clampedMicroBatch)
        params.n_threads = threads
        params.n_threads_batch = threads
        params.n_rs_seq = UInt32(
            min(maximumRecurrentStateSnapshots, max(0, recurrentStateSnapshots))
        )
        return params
    }

}

extension llama_model_params {
    mutating func configureLoadMode(useMemoryMap: Bool) {
        load_mode = useMemoryMap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE
    }

    mutating func configureForCPUOnly() {
        n_gpu_layers = 0
        split_mode = LLAMA_SPLIT_MODE_NONE
        main_gpu = -1
    }
}
