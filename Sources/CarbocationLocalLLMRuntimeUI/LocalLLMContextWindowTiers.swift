import CarbocationLocalLLM
import CarbocationLocalLLMRuntime

enum LocalLLMContextWindowTiers {
    static let defaultMinimumTier = 4_096

    static func candidates(
        trainingContext: Int,
        calibratedMaximum: Int?,
        defaultAutoCap: Int = LlamaContextPolicy.defaultAutoCap,
        minimumTier: Int = defaultMinimumTier
    ) -> [Int] {
        let knownTrainingContext = trainingContext > 0
            ? trainingContext
            : LlamaContextPolicy.unknownTrainingFallback
        let upperBound: Int
        if let calibratedMaximum, calibratedMaximum > 0 {
            upperBound = min(knownTrainingContext, calibratedMaximum)
        } else {
            upperBound = min(knownTrainingContext, defaultAutoCap)
        }
        let resolvedUpperBound = max(LlamaContextPolicy.minimumContext, upperBound)
        let visibleFloor = min(max(LlamaContextPolicy.minimumContext, minimumTier), resolvedUpperBound)
        let tiers = LlamaContextCalibrationAlgorithm
            .powerOfTwoTiers(upTo: resolvedUpperBound)
            .filter { $0 >= visibleFloor }
        return tiers.isEmpty ? [resolvedUpperBound] : tiers
    }

    static func nearestIndex(for value: Int, in candidates: [Int]) -> Int {
        guard let first = candidates.first else { return 0 }
        var bestIndex = 0
        var bestDistance = abs(first - value)
        for (index, candidate) in candidates.enumerated().dropFirst() {
            let distance = abs(candidate - value)
            if distance < bestDistance {
                bestIndex = index
                bestDistance = distance
            }
        }
        return bestIndex
    }
}
