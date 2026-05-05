import CarbocationLocalLLM
import CarbocationLocalLLMRuntime
import CarbocationLocalLLMUI
import SwiftUI
@testable import CarbocationLocalLLMRuntimeUI
import XCTest

@MainActor
final class CarbocationLocalLLMRuntimeUITests: XCTestCase {
    func testConfigurationViewConstructsWithDefaultRuntimeCalibrationWiring() throws {
        let root = try makeTemporaryDirectory()
        let library = ModelLibrary(root: root, contextLengthProbe: { _ in nil })
        let suiteName = "CarbocationLocalLLMRuntimeUITests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = LlamaContextCalibrationStore(defaults: defaults)

        let view = LocalLLMModelConfigurationView(
            library: library,
            selectedModelID: .constant(""),
            defaults: defaults,
            calibrationStore: store
        )
        _ = view.body

        let adapter = LocalLLMModelConfigurationView.makeCalibrationAdapter(
            library: library,
            store: store,
            configuration: LocalLLMEngineConfiguration()
        )
        XCTAssertEqual(
            adapter.runtimeFingerprint,
            LocalLLMEngine.contextCalibrationRuntimeFingerprint()
        )
    }

    func testUncalibratedContextWindowTiersStayConservative() {
        let candidates = LocalLLMContextWindowTiers.candidates(
            trainingContext: 262_144,
            calibratedMaximum: nil
        )

        let expectedFloor = min(
            LocalLLMContextWindowTiers.defaultMinimumTier,
            LlamaContextPolicy.defaultAutoCap
        )
        XCTAssertEqual(candidates.first, expectedFloor)
        XCTAssertEqual(candidates.last, LlamaContextPolicy.defaultAutoCap)
        XCTAssertFalse(candidates.contains { $0 > LlamaContextPolicy.defaultAutoCap })
    }

    func testCalibratedContextWindowTiersExposeSubMaximumChoices() {
        let candidates = LocalLLMContextWindowTiers.candidates(
            trainingContext: 262_144,
            calibratedMaximum: 262_144
        )

        XCTAssertEqual(candidates.first, LocalLLMContextWindowTiers.defaultMinimumTier)
        XCTAssertEqual(candidates.last, 262_144)
        XCTAssertTrue(candidates.contains(4_096))
        XCTAssertTrue(candidates.contains(8_192))
        XCTAssertTrue(candidates.contains(LlamaContextPolicy.defaultAutoCap))
        XCTAssertTrue(candidates.contains(32_768))
        XCTAssertTrue(candidates.contains(65_536))
        XCTAssertEqual(
            LocalLLMContextWindowTiers.nearestIndex(for: 60_000, in: candidates),
            candidates.firstIndex(of: 65_536)
        )
    }

    private func makeTemporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("CarbocationLocalLLMRuntimeUITests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
