import CarbocationLocalLLM
import XCTest
@testable import CarbocationLlamaRuntime

final class CarbocationLlamaRuntimeTests: XCTestCase {
    func testRuntimeTargetImportsAndLinksLlamaSymbols() {
        let summary = LlamaRuntimeSmoke.defaultModelParameterSummary()
        XCTAssertTrue(summary.contains("use_mmap="))
        XCTAssertTrue(summary.contains("n_gpu_layers="))
    }

    func testRuntimeCanReadDefaultContextParams() {
        XCTAssertGreaterThan(LlamaRuntimeSmoke.defaultContextBatchSize(), 0)
    }

    func testPrefillRangesSplitByBatchSize() {
        let ranges = LlamaEngine.prefillRanges(tokenCount: 10, maxBatchSize: 4)
            .map { [$0.lowerBound, $0.upperBound] }

        XCTAssertEqual(ranges, [[0, 4], [4, 8], [8, 10]])
        XCTAssertEqual(LlamaEngine.prefillRanges(tokenCount: 0, maxBatchSize: 4).count, 0)
        XCTAssertEqual(LlamaEngine.prefillRanges(tokenCount: 10, maxBatchSize: 0).count, 0)
    }

    func testContextClampRespectsTrainingLimitAndMinimum() {
        XCTAssertEqual(
            LlamaEngine.clampedContextSize(requestedContext: 32_768, trainingContext: 16_384),
            16_384
        )
        XCTAssertEqual(
            LlamaEngine.clampedContextSize(requestedContext: 128, trainingContext: 32_768),
            LlamaContextPolicy.minimumContext
        )
        XCTAssertEqual(
            LlamaEngine.clampedContextSize(requestedContext: 0, trainingContext: 0),
            LlamaContextPolicy.unknownTrainingFallback
        )
    }

    func testContextBatchCandidatesFallBackByHalving() {
        XCTAssertEqual(
            LlamaEngine.contextBatchCandidates(contextSize: 4_096, batchSizeLimit: 64),
            [64, 32, 16, 8, 4, 2, 1]
        )
        XCTAssertEqual(
            LlamaEngine.contextBatchCandidates(contextSize: 32, batchSizeLimit: 512),
            [32, 16, 8, 4, 2, 1]
        )
    }

    func testContextParamsClampBatchAndMicroBatchTogether() {
        let params = LlamaEngine.contextParams(
            contextSize: 4_096,
            batchSize: 64,
            threads: 2
        )

        XCTAssertEqual(params.n_ctx, 4_096)
        XCTAssertEqual(params.n_batch, 64)
        XCTAssertEqual(params.n_ubatch, 64)
        XCTAssertEqual(params.n_threads, 2)
        XCTAssertEqual(params.n_threads_batch, 2)
    }

    func testGenerationBudgetMath() {
        XCTAssertEqual(
            LlamaEngine.maxGenerationTokens(contextSize: 4_096, promptTokenCount: 3_000, reserve: 1_024),
            72
        )
        XCTAssertEqual(
            LlamaEngine.maxGenerationTokens(contextSize: 4_096, promptTokenCount: 4_000, reserve: 1_024),
            0
        )
    }

    func testStopSequenceTrimmingUsesEarliestMatch() {
        XCTAssertEqual(
            LlamaEngine.trimmingAtFirstStopSequence("alpha STOP beta END", stopSequences: ["END", "STOP"]),
            "alpha "
        )
        XCTAssertNil(LlamaEngine.trimmingAtFirstStopSequence("alpha beta", stopSequences: ["STOP"]))
        XCTAssertNil(LlamaEngine.trimmingAtFirstStopSequence("alpha beta", stopSequences: [""]))
    }

    func testCalgacusRankUsesDescendingLogitsAndTokenIDTieBreak() throws {
        let logits: [Float] = [0.5, 2.0, 2.0, -1.0]

        XCTAssertEqual(try LlamaEngine.calgacusRank(of: 1, in: logits), 1)
        XCTAssertEqual(try LlamaEngine.calgacusRank(of: 2, in: logits), 2)
        XCTAssertEqual(try LlamaEngine.calgacusRank(of: 0, in: logits), 3)
        XCTAssertEqual(try LlamaEngine.calgacusRank(of: 3, in: logits), 4)
    }

    func testCalgacusTokenAtRankUsesSameOrdering() throws {
        let logits: [Float] = [0.5, 2.0, 2.0, -1.0]

        XCTAssertEqual(try LlamaEngine.calgacusToken(atRank: 1, in: logits), 1)
        XCTAssertEqual(try LlamaEngine.calgacusToken(atRank: 2, in: logits), 2)
        XCTAssertEqual(try LlamaEngine.calgacusToken(atRank: 3, in: logits), 0)
        XCTAssertEqual(try LlamaEngine.calgacusToken(atRank: 4, in: logits), 3)
    }

    func testCalgacusRankTreatsNonFiniteLogitsAsLeastLikely() throws {
        let logits: [Float] = [.nan, 1.0, -.infinity, .infinity]

        XCTAssertEqual(try LlamaEngine.calgacusToken(atRank: 1, in: logits), 1)
        XCTAssertEqual(try LlamaEngine.calgacusToken(atRank: 2, in: logits), 0)
        XCTAssertEqual(try LlamaEngine.calgacusToken(atRank: 3, in: logits), 2)
    }

    func testCalgacusNegativeLogProbabilityUsesSoftmax() throws {
        let logits: [Float] = [0, 0]

        XCTAssertEqual(
            try LlamaEngine.calgacusNegativeLogProbability(of: 0, in: logits),
            log(2.0),
            accuracy: 0.000_001
        )
    }

    func testCalgacusStatsSummarizeTrace() {
        let trace = [
            CalgacusTraceEntry(index: 0, tokenID: 10, tokenText: "a", rank: 1, negativeLogProbability: 0.1),
            CalgacusTraceEntry(index: 1, tokenID: 11, tokenText: "b", rank: 5, negativeLogProbability: 0.2),
            CalgacusTraceEntry(index: 2, tokenID: 12, tokenText: "c", rank: 9, negativeLogProbability: 0.3)
        ]

        let stats = LlamaEngine.calgacusStats(for: trace)

        XCTAssertEqual(stats.tokenCount, 3)
        XCTAssertEqual(stats.maxRank, 9)
        XCTAssertEqual(stats.meanRank, 5)
        XCTAssertEqual(stats.medianRank, 5)
        XCTAssertEqual(stats.cumulativeNegativeLogProbability, 0.6, accuracy: 0.000_001)
        XCTAssertEqual(stats.averageNegativeLogProbability, 0.2, accuracy: 0.000_001)
    }

    func testCalgacusContextBudgetRejectsOverflow() {
        XCTAssertNoThrow(try LlamaEngine.calgacusValidateBudget(
            operation: "test",
            contextSize: 8,
            contextTokenCount: 3,
            payloadTokenCount: 5
        ))

        XCTAssertThrowsError(try LlamaEngine.calgacusValidateBudget(
            operation: "test",
            contextSize: 8,
            contextTokenCount: 4,
            payloadTokenCount: 5
        )) { error in
            guard case CalgacusError.contextBudgetExceeded = error else {
                return XCTFail("Expected contextBudgetExceeded, got \(error)")
            }
        }
    }

    func testCalgacusLiveRoundTripWhenModelPathProvided() async throws {
        guard let path = ProcessInfo.processInfo.environment["CALGACUS_TEST_MODEL_PATH"],
              !path.isEmpty
        else {
            throw XCTSkip("Set CALGACUS_TEST_MODEL_PATH to run the live Calgacus round-trip test.")
        }

        let engine = LlamaEngine(configuration: LlamaEngineConfiguration(
            gpuLayerCount: 0,
            useMemoryMap: true,
            batchSizeLimit: 512,
            threadCount: 2,
            promptReserveTokens: 0
        ))
        let loaded = try await engine.load(
            modelAt: URL(fileURLWithPath: path),
            requestedContext: 1_024
        )

        let secret = "The recipe is simple."
        let encoded = try await engine.encodeCalgacus(
            CalgacusEncodeRequest(
                secretText: secret,
                coverPrompt: "Write a friendly note about soup:",
                requestedContext: loaded.contextSize
            ),
            onEvent: { _ in }
        )
        let decoded = try await engine.decodeCalgacus(
            CalgacusDecodeRequest(
                coverText: encoded.coverText,
                coverPrompt: "Write a friendly note about soup:",
                requestedContext: loaded.contextSize
            ),
            onEvent: { _ in }
        )

        XCTAssertEqual(decoded.secretText, secret)
        await engine.unload()
    }

    func testFallbackPromptDetectsGemmaWithoutAppConcepts() throws {
        let descriptor = LlamaModelDescriptor(
            url: URL(fileURLWithPath: "/tmp/gemma-3.gguf"),
            displayName: "Gemma 3",
            filename: "gemma-3.gguf"
        )

        let rendered = try LlamaEngine.fallbackPrompt(
            system: "System",
            user: "User",
            embeddedTemplate: nil,
            descriptor: descriptor
        )

        XCTAssertEqual(rendered.mode, .gemmaFallback)
        XCTAssertTrue(rendered.text.contains("<start_of_turn>user"))
        XCTAssertTrue(rendered.text.contains("<start_of_turn>model"))
    }

    func testFallbackPromptDetectsChatMLWithoutAppConcepts() throws {
        let descriptor = LlamaModelDescriptor(
            url: URL(fileURLWithPath: "/tmp/qwen.gguf"),
            displayName: "Qwen",
            filename: "qwen.gguf"
        )

        let rendered = try LlamaEngine.fallbackPrompt(
            system: "System",
            user: "User",
            embeddedTemplate: nil,
            descriptor: descriptor
        )

        XCTAssertEqual(rendered.mode, .chatMLFallback)
        XCTAssertTrue(rendered.text.contains("<|im_start|>system"))
        XCTAssertTrue(rendered.text.contains("<|im_start|>assistant"))
    }

    func testFallbackPromptRejectsUnknownTemplateFamily() {
        let descriptor = LlamaModelDescriptor(
            url: URL(fileURLWithPath: "/tmp/unknown.gguf"),
            displayName: "Unknown",
            filename: "unknown.gguf"
        )

        XCTAssertThrowsError(try LlamaEngine.fallbackPrompt(
            system: "System",
            user: "User",
            embeddedTemplate: nil,
            descriptor: descriptor
        )) { error in
            guard case LLMEngineError.chatTemplateUnavailable = error else {
                return XCTFail("Expected chatTemplateUnavailable, got \(error)")
            }
        }
    }
}
