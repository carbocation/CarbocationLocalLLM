import Foundation

package struct LLMThinkingTerminationRequest: Equatable, Sendable {
    package var generationID: UInt64
    package var requestID: UInt64
    package var message: String
}

public final class LLMGenerationControl: @unchecked Sendable {
    private let lock = NSLock()
    private var nextGenerationID: UInt64 = 0
    private var activeGenerationID: UInt64?
    private var activeThinkingTerminationRequestCount: UInt64 = 0
    private var pendingThinkingTerminationRequest: LLMThinkingTerminationRequest?

    public init() {}

    public var thinkingTerminationRequestCount: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return activeGenerationID == nil ? 0 : activeThinkingTerminationRequestCount
    }

    @discardableResult
    public func requestThinkingTermination(message: String = "") -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard let activeGenerationID else {
            return false
        }

        activeThinkingTerminationRequestCount += 1
        pendingThinkingTerminationRequest = LLMThinkingTerminationRequest(
            generationID: activeGenerationID,
            requestID: activeThinkingTerminationRequestCount,
            message: message
        )
        return true
    }

    package func beginGeneration() -> UInt64 {
        lock.lock()
        defer { lock.unlock() }

        nextGenerationID += 1
        activeGenerationID = nextGenerationID
        activeThinkingTerminationRequestCount = 0
        pendingThinkingTerminationRequest = nil
        return nextGenerationID
    }

    package func finishGeneration(_ generationID: UInt64) {
        lock.lock()
        defer { lock.unlock() }

        guard activeGenerationID == generationID else {
            return
        }
        activeGenerationID = nil
        activeThinkingTerminationRequestCount = 0
        pendingThinkingTerminationRequest = nil
    }

    package func takePendingThinkingTerminationRequest(
        for generationID: UInt64
    ) -> LLMThinkingTerminationRequest? {
        lock.lock()
        defer { lock.unlock() }

        guard pendingThinkingTerminationRequest?.generationID == generationID else {
            return nil
        }

        let request = pendingThinkingTerminationRequest
        pendingThinkingTerminationRequest = nil
        return request
    }
}
