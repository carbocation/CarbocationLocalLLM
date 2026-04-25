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
