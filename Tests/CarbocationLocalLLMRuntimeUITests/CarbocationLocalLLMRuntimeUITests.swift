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
        let library = ModelLibrary(root: root, searchConfiguration: .managedOnly, contextLengthProbe: { _ in nil })
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

    func testCalibrationAdapterRunsLifecycleCallbacks() async throws {
        let suiteName = "CarbocationLocalLLMRuntimeUICalibrationAdapterTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = LlamaContextCalibrationStore(defaults: defaults)
        let runtime = LlamaContextCalibrationRuntimeFingerprint(
            platform: "macOS",
            gpuLayerCount: 999,
            useMemoryMap: true,
            batchSizeLimit: 2_048,
            threadCount: 4
        )
        let model = InstalledModel(
            displayName: "Calibration Model",
            filename: "model-Q4_K_M.gguf",
            sizeBytes: 2_000_000,
            contextLength: 65_536,
            quantization: "Q4_K_M",
            source: .imported,
            sha256: "abc123"
        )
        let record = LlamaContextCalibrationRecord(
            key: store.key(for: model, runtime: runtime),
            maximumSupportedContext: 65_536,
            probedTiers: [
                LlamaContextCalibrationProbe(context: 65_536, succeeded: true)
            ]
        )
        var events: [String] = []
        let adapter = ModelLibraryPickerCalibrationAdapter(
            store: store,
            runtimeFingerprint: runtime,
            onCalibrationStarted: { model in
                events.append("started:\(model.id.uuidString)")
            },
            onCalibrationCompleted: { _, record in
                events.append("completed:\(record.maximumSupportedContext)")
            },
            calibrate: { calibratedModel, progress in
                XCTAssertEqual(calibratedModel.id, model.id)
                progress(LlamaContextCalibrationProgress(
                    phase: .completed,
                    currentContext: record.maximumSupportedContext
                ))
                events.append("calibrated")
                return record
            }
        )

        let returned = try await adapter.runCalibration(model) { progress in
            events.append("progress:\(progress.currentContext ?? 0)")
        }

        XCTAssertEqual(returned, record)
        XCTAssertEqual(events, [
            "started:\(model.id.uuidString)",
            "progress:65536",
            "calibrated",
            "completed:65536"
        ])
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
