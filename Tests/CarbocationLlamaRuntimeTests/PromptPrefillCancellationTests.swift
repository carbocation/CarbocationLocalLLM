import CarbocationLocalLLM
import XCTest
@testable import CarbocationLlamaRuntime

final class PromptPrefillCancellationTests: XCTestCase {
    private enum TestError: Error {
        case decodeFailed
    }

    func testCancelledTaskDoesNotStartPromptPrefill() async {
        let result = await Task { () -> (decodedChunks: Int, cancelled: Bool) in
            withUnsafeCurrentTask { task in
                task?.cancel()
            }
            var decodedChunks = 0

            do {
                try LlamaEngine.performPromptPrefillChunks(
                    LlamaEngine.promptPrefillChunkPlans(tokenCount: 12, maxBatchSize: 4)
                ) { _ in
                    decodedChunks += 1
                }
                return (decodedChunks, false)
            } catch is CancellationError {
                return (decodedChunks, true)
            } catch {
                return (decodedChunks, false)
            }
        }.value

        XCTAssertTrue(result.cancelled)
        XCTAssertEqual(result.decodedChunks, 0)
    }

    func testPromptPrefillStopsAfterCancellationBetweenChunks() async {
        let result = await Task { () -> (decodedChunks: Int, cancelled: Bool) in
            var decodedChunks = 0

            do {
                try LlamaEngine.performPromptPrefillChunks(
                    LlamaEngine.promptPrefillChunkPlans(tokenCount: 12, maxBatchSize: 4)
                ) { _ in
                    decodedChunks += 1
                    if decodedChunks == 1 {
                        withUnsafeCurrentTask { task in
                            task?.cancel()
                        }
                    }
                }
                return (decodedChunks, false)
            } catch is CancellationError {
                return (decodedChunks, true)
            } catch {
                return (decodedChunks, false)
            }
        }.value

        XCTAssertTrue(result.cancelled)
        XCTAssertEqual(result.decodedChunks, 1)
    }

    func testCachedPrefixCancellationDoesNotReplayFullPrompt() {
        var replayedFullPrompt = false

        XCTAssertThrowsError(
            try LlamaEngine.retryPromptPrefillAfterCachedPrefixFailure(CancellationError()) {
                replayedFullPrompt = true
            }
        ) { error in
            XCTAssertTrue(error is CancellationError)
        }
        XCTAssertFalse(replayedFullPrompt)
    }

    func testCachedPrefixDecodeFailureStillReplaysFullPrompt() throws {
        var replayedFullPrompt = false

        try LlamaEngine.retryPromptPrefillAfterCachedPrefixFailure(TestError.decodeFailed) {
            replayedFullPrompt = true
        }

        XCTAssertTrue(replayedFullPrompt)
    }
}
