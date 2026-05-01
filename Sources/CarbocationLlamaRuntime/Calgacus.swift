import Foundation

public struct CalgacusEncodeRequest: Hashable, Sendable {
    public var secretText: String
    public var coverPrompt: String
    public var requestedContext: Int

    public init(secretText: String, coverPrompt: String, requestedContext: Int) {
        self.secretText = secretText
        self.coverPrompt = coverPrompt
        self.requestedContext = requestedContext
    }
}

public struct CalgacusDecodeRequest: Hashable, Sendable {
    public var coverText: String
    public var coverPrompt: String
    public var requestedContext: Int

    public init(coverText: String, coverPrompt: String, requestedContext: Int) {
        self.coverText = coverText
        self.coverPrompt = coverPrompt
        self.requestedContext = requestedContext
    }
}

public struct CalgacusRankStats: Hashable, Sendable {
    public var tokenCount: Int
    public var maxRank: Int
    public var meanRank: Double
    public var medianRank: Double
    public var cumulativeNegativeLogProbability: Double
    public var averageNegativeLogProbability: Double

    public init(
        tokenCount: Int,
        maxRank: Int,
        meanRank: Double,
        medianRank: Double,
        cumulativeNegativeLogProbability: Double,
        averageNegativeLogProbability: Double
    ) {
        self.tokenCount = tokenCount
        self.maxRank = maxRank
        self.meanRank = meanRank
        self.medianRank = medianRank
        self.cumulativeNegativeLogProbability = cumulativeNegativeLogProbability
        self.averageNegativeLogProbability = averageNegativeLogProbability
    }
}

public struct CalgacusTraceEntry: Hashable, Sendable, Identifiable {
    public var index: Int
    public var tokenID: Int32
    public var tokenText: String
    public var rank: Int
    public var negativeLogProbability: Double

    public var id: Int { index }

    public init(
        index: Int,
        tokenID: Int32,
        tokenText: String,
        rank: Int,
        negativeLogProbability: Double
    ) {
        self.index = index
        self.tokenID = tokenID
        self.tokenText = tokenText
        self.rank = rank
        self.negativeLogProbability = negativeLogProbability
    }
}

public struct CalgacusEncodeResult: Hashable, Sendable {
    public var coverText: String
    public var secretTokenCount: Int
    public var coverTokenCount: Int
    public var stats: CalgacusRankStats
    public var trace: [CalgacusTraceEntry]

    public init(
        coverText: String,
        secretTokenCount: Int,
        coverTokenCount: Int,
        stats: CalgacusRankStats,
        trace: [CalgacusTraceEntry]
    ) {
        self.coverText = coverText
        self.secretTokenCount = secretTokenCount
        self.coverTokenCount = coverTokenCount
        self.stats = stats
        self.trace = trace
    }
}

public struct CalgacusDecodeResult: Hashable, Sendable {
    public var secretText: String
    public var coverTokenCount: Int
    public var recoveredTokenCount: Int
    public var stats: CalgacusRankStats
    public var trace: [CalgacusTraceEntry]

    public init(
        secretText: String,
        coverTokenCount: Int,
        recoveredTokenCount: Int,
        stats: CalgacusRankStats,
        trace: [CalgacusTraceEntry]
    ) {
        self.secretText = secretText
        self.coverTokenCount = coverTokenCount
        self.recoveredTokenCount = recoveredTokenCount
        self.stats = stats
        self.trace = trace
    }
}

public enum CalgacusStage: String, Hashable, Sendable {
    case secretRanking = "secret-ranking"
    case coverGeneration = "cover-generation"
    case coverRanking = "cover-ranking"
    case secretRecovery = "secret-recovery"
    case verification = "verification"
}

public enum CalgacusEvent: Sendable {
    case started(operation: String)
    case tokensPrepared(operation: String, count: Int)
    case tokenProcessed(stage: CalgacusStage, index: Int, total: Int, rank: Int)
    case completed(operation: String, duration: TimeInterval)
}

public enum CalgacusError: Error, LocalizedError, Sendable {
    case emptySecretText
    case emptyCoverText
    case noInitialContext
    case contextBudgetExceeded(operation: String, contextSize: Int, requiredTokens: Int)
    case logitsUnavailable
    case invalidRank(Int, vocabularySize: Int)
    case invalidTokenID(Int32, vocabularySize: Int)
    case invalidGeneratedCoverToken(rank: Int, tokenID: Int32, reason: String)
    case textRenderingFailed(operation: String)
    case verificationFailed(expectedTokenCount: Int, recoveredTokenCount: Int)

    public var errorDescription: String? {
        switch self {
        case .emptySecretText:
            return "Secret text is empty after tokenization."
        case .emptyCoverText:
            return "Cover text is empty after tokenization."
        case .noInitialContext:
            return "The loaded model did not provide a usable beginning-of-sequence context."
        case .contextBudgetExceeded(let operation, let contextSize, let requiredTokens):
            return "\(operation) needs \(requiredTokens) tokens, but the loaded context has \(contextSize)."
        case .logitsUnavailable:
            return "llama.cpp did not return logits for the current token position."
        case .invalidRank(let rank, let vocabularySize):
            return "Rank \(rank) is outside the model vocabulary size \(vocabularySize)."
        case .invalidTokenID(let tokenID, let vocabularySize):
            return "Token id \(tokenID) is outside the model vocabulary size \(vocabularySize)."
        case .invalidGeneratedCoverToken(let rank, let tokenID, let reason):
            return "Rank \(rank) selected invalid cover token \(tokenID): \(reason)."
        case .textRenderingFailed(let operation):
            return "\(operation) produced bytes that could not be rendered as stable UTF-8 text."
        case .verificationFailed(let expectedTokenCount, let recoveredTokenCount):
            return "Calgacus verification failed: expected \(expectedTokenCount) secret tokens, recovered \(recoveredTokenCount)."
        }
    }
}
