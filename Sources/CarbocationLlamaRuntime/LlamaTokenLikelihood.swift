import Foundation

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
        rank: Int
    ) {
        self.index = index
        self.tokenID = tokenID
        self.tokenBytes = tokenBytes
        self.byteRange = byteRange
        self.logProbability = logProbability
        self.rank = rank
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
