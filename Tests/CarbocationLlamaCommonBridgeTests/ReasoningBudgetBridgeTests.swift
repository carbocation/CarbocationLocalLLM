import CarbocationLlamaCommonBridge
import llama
import XCTest

final class ReasoningBudgetBridgeTests: XCTestCase {
    func testZeroBudgetForcesCloseImmediatelyAfterThinkingStarts() {
        let sampler = makeSampler(budget: 0)
        defer { llama_sampler_free(sampler) }

        llama_sampler_accept(sampler, 1)

        XCTAssertEqual(
            carbocation_llama_reasoning_budget_sampler_state(sampler),
            CARBOCATION_LLAMA_REASONING_BUDGET_FORCING
        )

        let candidates = appliedCandidates(sampler: sampler, ids: [2, 3])
        XCTAssertEqual(candidates[0].logit, 0)
        XCTAssertTrue(candidates[1].logit.isInfinite)
        XCTAssertLessThan(candidates[1].logit, 0)
    }

    func testPositiveBudgetForcesCloseAfterBudgetIsSpent() {
        let sampler = makeSampler(budget: 2)
        defer { llama_sampler_free(sampler) }

        llama_sampler_accept(sampler, 1)
        XCTAssertEqual(
            carbocation_llama_reasoning_budget_sampler_state(sampler),
            CARBOCATION_LLAMA_REASONING_BUDGET_COUNTING
        )

        llama_sampler_accept(sampler, 10)
        XCTAssertEqual(
            carbocation_llama_reasoning_budget_sampler_state(sampler),
            CARBOCATION_LLAMA_REASONING_BUDGET_COUNTING
        )

        llama_sampler_accept(sampler, 11)
        XCTAssertEqual(
            carbocation_llama_reasoning_budget_sampler_state(sampler),
            CARBOCATION_LLAMA_REASONING_BUDGET_FORCING
        )

        let candidates = appliedCandidates(sampler: sampler, ids: [2, 12])
        XCTAssertEqual(candidates[0].logit, 0)
        XCTAssertTrue(candidates[1].logit.isInfinite)
        XCTAssertLessThan(candidates[1].logit, 0)
    }

    func testRemainingBudgetReportsCountdown() {
        let sampler = makeSampler(budget: 3)
        defer { llama_sampler_free(sampler) }

        XCTAssertEqual(carbocation_llama_reasoning_budget_sampler_remaining(sampler), 3)

        llama_sampler_accept(sampler, 1)
        XCTAssertEqual(carbocation_llama_reasoning_budget_sampler_remaining(sampler), 3)

        llama_sampler_accept(sampler, 10)
        XCTAssertEqual(carbocation_llama_reasoning_budget_sampler_remaining(sampler), 2)

        llama_sampler_accept(sampler, 11)
        XCTAssertEqual(carbocation_llama_reasoning_budget_sampler_remaining(sampler), 1)
    }

    func testNaturalCloseBeforeBudgetExhaustionLeavesSamplerDone() {
        let sampler = makeSampler(budget: 4)
        defer { llama_sampler_free(sampler) }

        llama_sampler_accept(sampler, 1)
        llama_sampler_accept(sampler, 2)

        XCTAssertEqual(
            carbocation_llama_reasoning_budget_sampler_state(sampler),
            CARBOCATION_LLAMA_REASONING_BUDGET_DONE
        )

        let candidates = appliedCandidates(sampler: sampler, ids: [2, 3])
        XCTAssertEqual(candidates[0].logit, 0)
        XCTAssertEqual(candidates[1].logit, 0)
    }

    func testCountingInitialStateSupportsPrefilledOpenThinking() {
        let sampler = makeSampler(
            budget: 1,
            initialState: CARBOCATION_LLAMA_REASONING_BUDGET_COUNTING
        )
        defer { llama_sampler_free(sampler) }

        XCTAssertEqual(
            carbocation_llama_reasoning_budget_sampler_state(sampler),
            CARBOCATION_LLAMA_REASONING_BUDGET_COUNTING
        )

        llama_sampler_accept(sampler, 10)

        XCTAssertEqual(
            carbocation_llama_reasoning_budget_sampler_state(sampler),
            CARBOCATION_LLAMA_REASONING_BUDGET_FORCING
        )
    }

    private func makeSampler(
        budget: Int32,
        initialState: carbocation_llama_reasoning_budget_state = CARBOCATION_LLAMA_REASONING_BUDGET_IDLE
    ) -> UnsafeMutablePointer<llama_sampler> {
        let start: [llama_token] = [1]
        let end: [llama_token] = [2]
        let forced: [llama_token] = [2]

        let sampler = start.withUnsafeBufferPointer { startBuffer in
            end.withUnsafeBufferPointer { endBuffer in
                forced.withUnsafeBufferPointer { forcedBuffer in
                    carbocation_llama_reasoning_budget_sampler_init(
                        nil,
                        startBuffer.baseAddress,
                        startBuffer.count,
                        endBuffer.baseAddress,
                        endBuffer.count,
                        forcedBuffer.baseAddress,
                        forcedBuffer.count,
                        budget,
                        initialState
                    )
                }
            }
        }

        return try! XCTUnwrap(sampler)
    }

    private func appliedCandidates(
        sampler: UnsafeMutablePointer<llama_sampler>,
        ids: [llama_token]
    ) -> [llama_token_data] {
        var candidates = ids.map { llama_token_data(id: $0, logit: 0, p: 0) }
        candidates.withUnsafeMutableBufferPointer { buffer in
            var candidateArray = llama_token_data_array(
                data: buffer.baseAddress,
                size: buffer.count,
                selected: -1,
                sorted: false
            )
            llama_sampler_apply(sampler, &candidateArray)
        }
        return candidates
    }
}
