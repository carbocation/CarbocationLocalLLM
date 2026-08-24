import XCTest
@testable import CarbocationLocalLLM

final class GenerationOptionsValidationTests: XCTestCase {
    func testValidSamplingOptionsPassValidation() throws {
        let options = GenerationOptions(
            temperature: 0.7,
            topP: 0.95,
            topK: 20,
            minP: 0.05,
            presencePenalty: -1.5,
            repetitionPenalty: 1.1
        )
        try options.validate()
        try options.validateForFloat32Backend(maximumTopK: Int(Int32.max))
    }

    func testDisabledSamplingSentinelsRemainValid() throws {
        let options = GenerationOptions(
            temperature: -Double.greatestFiniteMagnitude,
            topP: 0,
            topK: Int.min,
            minP: -Double.greatestFiniteMagnitude,
            presencePenalty: 0,
            repetitionPenalty: 0.5
        )
        try options.validate()
        try options.validateForFloat32Backend(maximumTopK: Int(Int32.max))
    }

    func testNonFiniteSamplingOptionsAreRejected() {
        assertValidationError(
            GenerationOptions(temperature: .nan),
            equals: .mustBeFinite(.temperature)
        )
        assertValidationError(
            GenerationOptions(topP: .infinity),
            equals: .mustBeFinite(.topP)
        )
        assertValidationError(
            GenerationOptions(minP: -.infinity),
            equals: .mustBeFinite(.minP)
        )
        assertValidationError(
            GenerationOptions(presencePenalty: .nan),
            equals: .mustBeFinite(.presencePenalty)
        )
        assertValidationError(
            GenerationOptions(repetitionPenalty: .infinity),
            equals: .mustBeFinite(.repetitionPenalty)
        )
    }

    func testInvalidProbabilitiesAreRejected() {
        assertValidationError(
            GenerationOptions(topP: -0.01),
            equals: .mustBeWithinUnitInterval(.topP)
        )
        assertValidationError(
            GenerationOptions(topP: 1.01),
            equals: .mustBeWithinUnitInterval(.topP)
        )
        assertValidationError(
            GenerationOptions(minP: 1.01),
            equals: .mustNotExceedOne(.minP)
        )
    }

    func testInvalidRepetitionPenaltyIsRejected() {
        assertValidationError(
            GenerationOptions(repetitionPenalty: 0),
            equals: .mustBePositive(.repetitionPenalty)
        )
        assertValidationError(
            GenerationOptions(repetitionPenalty: -1),
            equals: .mustBePositive(.repetitionPenalty)
        )
    }

    func testNegativeThinkingBudgetIsRejectedWithoutTrapping() {
        let constructed = GenerationOptions(thinkingBudgetTokens: -1)
        assertValidationError(
            constructed,
            equals: .mustBeNonnegative(.thinkingBudgetTokens)
        )
        assertBackendValidationError(
            constructed,
            equals: .mustBeNonnegative(.thinkingBudgetTokens)
        )

        var mutated = GenerationOptions()
        mutated.thinkingBudgetTokens = -2
        assertValidationError(
            mutated,
            equals: .mustBeNonnegative(.thinkingBudgetTokens)
        )
    }

    func testValuesOutsideNativeNumericRangesAreRejected() {
        assertBackendValidationError(
            GenerationOptions(topK: Int(Int32.max) + 1),
            equals: .exceedsBackendIntegerLimit(.topK, maximum: Int(Int32.max))
        )
        assertBackendValidationError(
            GenerationOptions(temperature: Double.greatestFiniteMagnitude),
            equals: .unsafeForBackendFloatingPoint(.temperature)
        )
        assertBackendValidationError(
            GenerationOptions(temperature: Double(Float.leastNonzeroMagnitude)),
            equals: .unsafeForBackendFloatingPoint(.temperature)
        )
        assertBackendValidationError(
            GenerationOptions(topP: Double.leastNonzeroMagnitude),
            equals: .unsafeForBackendFloatingPoint(.topP)
        )
        assertBackendValidationError(
            GenerationOptions(presencePenalty: Double.greatestFiniteMagnitude),
            equals: .unsafeForBackendFloatingPoint(.presencePenalty)
        )
        assertBackendValidationError(
            GenerationOptions(repetitionPenalty: Double.greatestFiniteMagnitude),
            equals: .unsafeForBackendFloatingPoint(.repetitionPenalty)
        )
        assertBackendValidationError(
            GenerationOptions(repetitionPenalty: Double.leastNonzeroMagnitude),
            equals: .unsafeForBackendFloatingPoint(.repetitionPenalty)
        )
        assertBackendValidationError(
            GenerationOptions(repetitionPenalty: Double(Float.leastNonzeroMagnitude)),
            equals: .unsafeForBackendFloatingPoint(.repetitionPenalty)
        )
    }

    private func assertValidationError(
        _ options: GenerationOptions,
        equals expected: GenerationOptionsValidationError,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        do {
            try options.validate()
            XCTFail("Expected validation to fail.", file: file, line: line)
        } catch let error as GenerationOptionsValidationError {
            XCTAssertEqual(error, expected, file: file, line: line)
        } catch {
            XCTFail("Unexpected error: \(error)", file: file, line: line)
        }
    }

    private func assertBackendValidationError(
        _ options: GenerationOptions,
        equals expected: GenerationOptionsValidationError,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        do {
            try options.validateForFloat32Backend(maximumTopK: Int(Int32.max))
            XCTFail("Expected backend validation to fail.", file: file, line: line)
        } catch let error as GenerationOptionsValidationError {
            XCTAssertEqual(error, expected, file: file, line: line)
        } catch {
            XCTFail("Unexpected error: \(error)", file: file, line: line)
        }
    }
}
