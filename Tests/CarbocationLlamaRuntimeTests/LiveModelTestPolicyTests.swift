import XCTest

final class LiveModelTestPolicyTests: XCTestCase {
    func testProfileRequiresMasterOptInEvenWhenAPathIsPresent() {
        XCTAssertThrowsError(try LiveModelTestPolicy.requiredReadablePath(
            "LLAMA_TEST_MODEL_PATH",
            for: .textPreflight,
            environment: ["LLAMA_TEST_MODEL_PATH": "/tmp/model.gguf"]
        )) { error in
            XCTAssertTrue(error is XCTSkip)
        }
    }

    func testEnabledProfileReportsItsMissingRequiredPath() {
        XCTAssertThrowsError(try LiveModelTestPolicy.requiredReadablePath(
            "LLAMA_TEST_MODEL_PATH",
            for: .textPreflight,
            environment: [LiveModelTestPolicy.optInEnvironmentVariable: "1"]
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("LLAMA_TEST_MODEL_PATH"))
            XCTAssertTrue(error.localizedDescription.contains("text-preflight"))
        }
    }
}
