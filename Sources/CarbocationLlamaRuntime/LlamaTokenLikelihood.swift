import Foundation

/// Optional work performed while scoring known text. The default value leaves
/// full-vocabulary statistics disabled and preserves ordinary likelihood
/// scoring behavior.
public struct LlamaTokenLikelihoodOptions: Hashable, Sendable {
    public var vocabularyStatistics: LlamaVocabularyStatisticsOptions?

    public init(vocabularyStatistics: LlamaVocabularyStatisticsOptions? = nil) {
        self.vocabularyStatistics = vocabularyStatistics
    }
}

/// Compact summaries derived from each full next-token distribution. Raw
/// vocabulary logits are never retained or returned.
public struct LlamaVocabularyStatisticsOptions: Hashable, Sendable {
    /// Temperatures for which to calculate the log normalization term
    /// `log(sum_v p(v)^(1 / temperature))`. Values are validated when scoring,
    /// then deduplicated and evaluated in ascending order.
    public var temperatureNormalizationTemperatures: [Double]
    /// Whether to calculate the midpoint cumulative probability mass ahead of
    /// the observed token. This is a probability-aware alternative to raw
    /// vocabulary rank with a model-native uniform baseline.
    public var includesProbabilityMassRank: Bool

    public init(
        temperatureNormalizationTemperatures: [Double] = [],
        includesProbabilityMassRank: Bool = false
    ) {
        self.temperatureNormalizationTemperatures = temperatureNormalizationTemperatures
        self.includesProbabilityMassRank = includesProbabilityMassRank
    }
}

public enum LlamaVocabularyStatisticsError: Error, LocalizedError, Hashable, Sendable {
    case invalidTemperature(Double)

    public var errorDescription: String? {
        switch self {
        case .invalidTemperature(let temperature):
            return "Vocabulary-statistics temperatures must be finite and greater than zero; received \(temperature)."
        }
    }
}

/// One temperature-dependent normalization term from the model's raw
/// next-token distribution.
public struct LlamaTemperatureNormalization: Hashable, Sendable {
    public var temperature: Double
    public var logNormalizer: Double

    public init(temperature: Double, logNormalizer: Double) {
        self.temperature = temperature
        self.logNormalizer = logNormalizer
    }
}

/// Scalar statistics for one full next-token vocabulary distribution.
public struct LlamaTokenVocabularyStatistics: Hashable, Sendable {
    /// Expected log probability of a token drawn from the raw model
    /// distribution. Its negation is the distribution entropy in nats.
    public var expectedLogProbability: Double
    /// Variance of log probability under the raw model distribution.
    public var logProbabilityVariance: Double
    /// Probability mass assigned to tokens more likely than the observed
    /// token, plus half the mass tied with it. Values are in `[0, 1]`.
    public var probabilityMassRank: Double?
    /// Total probability assigned to tokens tied with the observed token,
    /// including the observed token itself. Combine this with the midpoint
    /// rank to reconstruct the tie interval for seeded uniformization.
    public var probabilityMassRankTieProbability: Double?
    /// Requested temperature-normalization terms, sorted by temperature.
    public var temperatureNormalizations: [LlamaTemperatureNormalization]

    public var entropy: Double {
        -expectedLogProbability
    }

    public var logProbabilityStandardDeviation: Double {
        sqrt(max(0, logProbabilityVariance))
    }

    public init(
        expectedLogProbability: Double,
        logProbabilityVariance: Double,
        probabilityMassRank: Double? = nil,
        probabilityMassRankTieProbability: Double? = nil,
        temperatureNormalizations: [LlamaTemperatureNormalization] = []
    ) {
        self.expectedLogProbability = expectedLogProbability
        self.logProbabilityVariance = logProbabilityVariance
        self.probabilityMassRank = probabilityMassRank
        self.probabilityMassRankTieProbability = probabilityMassRankTieProbability
        self.temperatureNormalizations = temperatureNormalizations
    }

    public func temperatureNormalization(
        at temperature: Double
    ) -> LlamaTemperatureNormalization? {
        temperatureNormalizations.first { $0.temperature == temperature }
    }
}

/// The loaded llama model's likelihood for one observed token, conditioned on
/// every preceding token in the scored text.
public struct LlamaTokenLikelihood: Hashable, Sendable, Identifiable {
    public var index: Int
    public var tokenID: Int32
    /// The exact bytes rendered by this token.
    public var tokenBytes: Data
    /// The token's half-open range in the scored text's UTF-8 bytes.
    public var byteRange: Range<Int>
    /// Natural-log probability before any sampling transforms are applied.
    public var logProbability: Double
    /// One-based rank in the model vocabulary, ordered by descending logit.
    public var rank: Int
    /// Compact full-vocabulary summaries when explicitly requested.
    public var vocabularyStatistics: LlamaTokenVocabularyStatistics?

    public var id: Int { index }

    /// Returns `nil` when this token contains only part of a UTF-8 scalar.
    public var tokenText: String? {
        String(data: tokenBytes, encoding: .utf8)
    }

    public var negativeLogProbability: Double {
        -logProbability
    }

    public var probability: Double {
        exp(logProbability)
    }

    public init(
        index: Int,
        tokenID: Int32,
        tokenBytes: Data,
        byteRange: Range<Int>,
        logProbability: Double,
        rank: Int,
        vocabularyStatistics: LlamaTokenVocabularyStatistics? = nil
    ) {
        self.index = index
        self.tokenID = tokenID
        self.tokenBytes = tokenBytes
        self.byteRange = byteRange
        self.logProbability = logProbability
        self.rank = rank
        self.vocabularyStatistics = vocabularyStatistics
    }
}

/// Progress through a bounded token-likelihood scoring operation.
public struct LlamaTokenLikelihoodProgress: Hashable, Sendable {
    /// Tokens whose likelihoods have been computed.
    public var completedTokenCount: Int
    /// Total observed tokens in the tokenized input.
    public var totalTokenCount: Int

    public var fractionCompleted: Double {
        guard totalTokenCount > 0 else { return 1 }
        return min(1, max(0, Double(completedTokenCount) / Double(totalTokenCount)))
    }

    public init(completedTokenCount: Int, totalTokenCount: Int) {
        self.completedTokenCount = completedTokenCount
        self.totalTokenCount = totalTokenCount
    }
}

public enum LlamaTokenLikelihoodError: Error, LocalizedError, Hashable, Sendable {
    case noInitialContext
    case contextBudgetExceeded(contextSize: Int, requiredTokens: Int)
    case logitsUnavailable
    case invalidTokenID(Int32, vocabularySize: Int)
    case tokenRenderingMismatch

    public var errorDescription: String? {
        switch self {
        case .noInitialContext:
            return "The loaded model did not provide a usable beginning-of-sequence context."
        case .contextBudgetExceeded(let contextSize, let requiredTokens):
            return "Likelihood scoring needs \(requiredTokens) context tokens, but the loaded context has \(contextSize)."
        case .logitsUnavailable:
            return "llama.cpp did not return logits for the current token position."
        case .invalidTokenID(let tokenID, let vocabularySize):
            return "Token id \(tokenID) is outside the model vocabulary size \(vocabularySize)."
        case .tokenRenderingMismatch:
            return "The model's token pieces did not reconstruct the scored text exactly."
        }
    }
}
