import CarbocationLocalLLM
import XCTest
@testable import CarbocationLlamaRuntime

final class LlamaContextConfigurationValidationTests: XCTestCase {
    func testValidAndSentinelContextConfigurationPassesValidation() throws {
        try LlamaEngine.validateContextConfiguration(
            requestedContext: 262_144,
            batchSizeLimit: 2_048,
            microBatchSizeLimit: 512
        )
        try LlamaEngine.validateContextConfiguration(
            requestedContext: 0,
            batchSizeLimit: -1,
            microBatchSizeLimit: 0
        )
    }

    func testOversizedPositiveContextConfigurationIsRejected() {
        let maximum = LlamaEngine.maximumBackendContextParameter
        let oversized = maximum + 1

        assertConfigurationError(
            requestedContext: oversized,
            batchSizeLimit: 2_048,
            microBatchSizeLimit: nil,
            equals: .exceedsBackendLimit(parameter: .requestedContext, maximum: maximum)
        )
        assertConfigurationError(
            requestedContext: 4_096,
            batchSizeLimit: oversized,
            microBatchSizeLimit: nil,
            equals: .exceedsBackendLimit(parameter: .batchSizeLimit, maximum: maximum)
        )
        assertConfigurationError(
            requestedContext: 4_096,
            batchSizeLimit: 2_048,
            microBatchSizeLimit: oversized,
            equals: .exceedsBackendLimit(parameter: .microBatchSizeLimit, maximum: maximum)
        )
    }

    func testContextSizeAndParameterBuilderDefensivelyClampNativeBounds() {
        XCTAssertEqual(
            LlamaEngine.clampedContextSize(requestedContext: Int.max, trainingContext: 0),
            LlamaEngine.maximumBackendContextParameter
        )

        let params = LlamaEngine.contextParams(
            contextSize: Int.max,
            batchSize: Int.max,
            microBatchSize: Int.max,
            threads: 2,
            recurrentStateSnapshots: Int.max
        )
        XCTAssertEqual(params.n_ctx, UInt32(LlamaEngine.maximumBackendContextParameter))
        XCTAssertEqual(params.n_batch, UInt32(LlamaEngine.maximumBackendContextParameter))
        XCTAssertEqual(params.n_ubatch, UInt32(LlamaEngine.maximumBackendContextParameter))
        XCTAssertEqual(
            params.n_rs_seq,
            UInt32(LlamaEngine.maximumRecurrentStateSnapshots)
        )
    }

    func testCalibrationRejectsOversizedBackendConfigurationBeforeLoading() async {
        let maximum = LlamaEngine.maximumBackendContextParameter
        let engine = LlamaEngine(configuration: LlamaEngineConfiguration(
            batchSizeLimit: maximum + 1
        ))
        let model = InstalledModel(
            displayName: "Invalid calibration fixture",
            filename: "missing.gguf",
            sizeBytes: 0,
            contextLength: 4_096,
            quantization: nil,
            source: .imported
        )

        do {
            _ = try await engine.calibrateContext(
                model: model,
                from: URL(fileURLWithPath: "/missing-calibration-fixture")
            )
            XCTFail("Calibration should reject an oversized batch before model loading.")
        } catch let error as LlamaContextConfigurationError {
            XCTAssertEqual(
                error,
                .exceedsBackendLimit(parameter: .batchSizeLimit, maximum: maximum)
            )
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testPublicLoadRejectsOversizedValuesBeforeModelIO() async {
        let maximum = LlamaEngine.maximumBackendContextParameter
        let missingURL = URL(fileURLWithPath: "/missing-load-validation-fixture.gguf")
        let engine = LlamaEngine()

        do {
            _ = try await engine.load(
                modelAt: missingURL,
                requestedContext: maximum + 1
            )
            XCTFail("Load should reject context size before reading the model.")
        } catch let error as LlamaContextConfigurationError {
            XCTAssertEqual(
                error,
                .exceedsBackendLimit(parameter: .requestedContext, maximum: maximum)
            )
        } catch {
            XCTFail("Unexpected error: \(error)")
        }

        let invalidConfigurationEngine = LlamaEngine(
            configuration: LlamaEngineConfiguration(batchSizeLimit: maximum + 1)
        )
        do {
            _ = try await invalidConfigurationEngine.load(
                modelAt: missingURL,
                requestedContext: 4_096
            )
            XCTFail("Load should reject batch size before reading the model.")
        } catch let error as LlamaContextConfigurationError {
            XCTAssertEqual(
                error,
                .exceedsBackendLimit(parameter: .batchSizeLimit, maximum: maximum)
            )
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testSamplerDiagnosticsFormatNonFiniteAndHugeValuesWithoutTrapping() {
        let diagnostics = LlamaEngine.resolvedSamplerDiagnostics(options: GenerationOptions(
            temperature: .nan,
            presencePenalty: Double.greatestFiniteMagnitude
        ))

        XCTAssertTrue(diagnostics.requestLine.lowercased().contains("nan"))
        XCTAssertTrue(diagnostics.resolvedLine.lowercased().contains("nan"))
        XCTAssertTrue(diagnostics.requestLine.contains("presencePenalty="))
    }

    private func assertConfigurationError(
        requestedContext: Int,
        batchSizeLimit: Int,
        microBatchSizeLimit: Int?,
        equals expected: LlamaContextConfigurationError,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        do {
            try LlamaEngine.validateContextConfiguration(
                requestedContext: requestedContext,
                batchSizeLimit: batchSizeLimit,
                microBatchSizeLimit: microBatchSizeLimit
            )
            XCTFail("Expected validation to fail.", file: file, line: line)
        } catch let error as LlamaContextConfigurationError {
            XCTAssertEqual(error, expected, file: file, line: line)
        } catch {
            XCTFail("Unexpected error: \(error)", file: file, line: line)
        }
    }
}
