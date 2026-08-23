import CarbocationLlamaRuntime
import CarbocationLocalLLM
import Darwin
import Foundation

private let defaultBenchmarkPrompt = (1...12).map { index in
    "Benchmark paragraph \(index): Local language-model inference processes a stable prompt prefix "
        + "before decoding response tokens. Keep the response continuous, factual, and concise."
}.joined(separator: "\n") + "\n\nWrite short factual sentences about on-device language-model inference until the token limit is reached."

private let defaultWarmSuffix = "\n\nContinue with additional short sentences while preserving the same style and topic."

private struct BenchmarkArguments {
    var modelPath: String = ProcessInfo.processInfo.environment["CLLM_BENCHMARK_MODEL_PATH"] ?? ""
    var systemPrompt = "You are a deterministic benchmark assistant. Follow the continuation instruction without a conclusion."
    var prompt = ProcessInfo.processInfo.environment["CLLM_BENCHMARK_PROMPT"] ?? defaultBenchmarkPrompt
    var warmSuffix = defaultWarmSuffix
    var contextSize = 4_096
    var maxOutputTokens = 128
    var gpuLayerCount = LlamaEngineConfiguration.defaultGPULayerCount
    var useMemoryMap = true
    var batchSizeLimit = LlamaEngineConfiguration.defaultBatchSizeLimit
    var threadCount: Int32?
    var accelerationPolicy = LLMAccelerationPolicy.disabled
    var mtpMaxDraftTokens = LlamaEngineConfiguration.defaultMTPMaxDraftTokens
    var jsonOutput = false
}

private enum BenchmarkError: LocalizedError {
    case missingModel
    case modelNotFound(String)
    case missingValue(String)
    case unknownArgument(String)
    case invalidInteger(name: String, value: String)
    case invalidRange(name: String, value: Int, expected: String)
    case unreadablePromptFile(String, String)

    var errorDescription: String? {
        switch self {
        case .missingModel:
            return "Provide --model PATH or set CLLM_BENCHMARK_MODEL_PATH."
        case .modelNotFound(let path):
            return "Benchmark model is not a readable file: \(path)"
        case .missingValue(let name):
            return "Missing value for \(name)."
        case .unknownArgument(let name):
            return "Unknown argument: \(name)."
        case .invalidInteger(let name, let value):
            return "Invalid integer for \(name): \(value)."
        case .invalidRange(let name, let value, let expected):
            return "Invalid value for \(name): \(value). Expected \(expected)."
        case .unreadablePromptFile(let path, let detail):
            return "Could not read prompt file \(path): \(detail)"
        }
    }
}

private struct BenchmarkReport: Codable {
    var schemaVersion: Int
    var generatedAt: String
    var buildConfiguration: String
    var system: SystemReport
    var configuration: ConfigurationReport
    var model: ModelReport
    var modelLoadSeconds: Double
    var runs: [RunReport]
}

private struct SystemReport: Codable {
    var operatingSystem: String
    var activeProcessorCount: Int
    var physicalMemoryBytes: UInt64
}

private struct ConfigurationReport: Codable {
    var requestedContextSize: Int
    var maxOutputTokens: Int
    var requestedGPULayerCount: Int
    var useMemoryMap: Bool
    var requestedBatchSizeLimit: Int
    var effectiveLogicalBatchSize: Int?
    var effectivePhysicalMicroBatchSize: Int?
    var effectiveThreadCount: Int
    var accelerationPolicy: String
    var mtpMaxDraftTokens: Int
    var promptUTF8Bytes: Int
    var warmSuffixUTF8Bytes: Int
}

private struct ModelReport: Codable {
    var path: String
    var filename: String
    var fileSizeBytes: UInt64?
    var architecture: String?
    var loadedContextSize: Int
    var trainingContextSize: Int
    var hasEmbeddedChatTemplate: Bool
    var supportsMTPAcceleration: Bool
    var nextNPredictLayers: Int?
}

private struct RunReport: Codable {
    var name: String
    var cacheCondition: String
    var preflightSeconds: Double
    var promptTokens: Int
    var effectiveMaxOutputTokens: Int
    var timeToFirstTokenSeconds: Double?
    var totalRequestSeconds: Double
    var runtimeReportedSeconds: Double?
    var generatedTokens: Int
    var stopReason: String
    var templateMode: String
    var outputUTF8Bytes: Int
    var endToEndGeneratedTokensPerSecond: Double?
    var postFirstTokenTokensPerSecond: Double?
    var acceleration: AccelerationReport?
}

private struct AccelerationReport: Codable {
    var status: String
    var accelerator: String
    var maxDraftTokens: Int
    var draftCalls: Int
    var draftTokensGenerated: Int
    var draftTokensAccepted: Int
    var acceptanceRate: Double?

    init(_ stats: LLMGenerationAccelerationStats) {
        status = stats.status.rawValue
        accelerator = stats.accelerator
        maxDraftTokens = stats.maxDraftTokens
        draftCalls = stats.draftCalls
        draftTokensGenerated = stats.draftTokensGenerated
        draftTokensAccepted = stats.draftTokensAccepted
        acceptanceRate = stats.acceptanceRate
    }
}

private struct EventSnapshot {
    var firstTokenInstant: ContinuousClock.Instant?
    var runtimeReportedSeconds: Double?
    var promptTokens: Int?
    var generatedTokens: Int?
    var stopReason: String?
    var templateMode: LLMChatTemplateMode?
    var accelerationStats: LLMGenerationAccelerationStats?
}

private final class EventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var firstTokenInstant: ContinuousClock.Instant?
    private var runtimeReportedSeconds: Double?
    private var promptTokens: Int?
    private var generatedTokens: Int?
    private var stopReason: String?
    private var templateMode: LLMChatTemplateMode?
    private var accelerationStats: LLMGenerationAccelerationStats?

    func record(_ event: LLMPhaseAwareStreamEvent, at instant: ContinuousClock.Instant) {
        lock.withLock {
            switch event {
            case .firstByteReceived:
                if firstTokenInstant == nil {
                    firstTokenInstant = instant
                }
            case .generationStats(let promptTokens, let generatedTokens, let stopReason, let templateMode, _):
                self.promptTokens = promptTokens
                self.generatedTokens = generatedTokens
                self.stopReason = stopReason
                self.templateMode = templateMode
            case .accelerationStats(let stats):
                accelerationStats = stats
            case .done(_, let duration, _):
                runtimeReportedSeconds = duration
            case .requestSent, .phaseChanged, .tokenChunk, .finalAnswerDelta, .finalAnswerSnapshot, .diagnostic:
                break
            }
        }
    }

    func snapshot() -> EventSnapshot {
        lock.withLock {
            EventSnapshot(
                firstTokenInstant: firstTokenInstant,
                runtimeReportedSeconds: runtimeReportedSeconds,
                promptTokens: promptTokens,
                generatedTokens: generatedTokens,
                stopReason: stopReason,
                templateMode: templateMode,
                accelerationStats: accelerationStats
            )
        }
    }
}

@main
private enum CLLMBenchmarkCommand {
    static func main() async {
        do {
            let arguments = try parseArguments(Array(CommandLine.arguments.dropFirst()))
            let report = try await benchmark(arguments)
            if arguments.jsonOutput {
                try printJSON(report)
            } else {
                printHumanReadable(report)
            }
        } catch {
            fputs("error: \(error.localizedDescription)\n", stderr)
            fputs("Run CLLMBenchmarkCommand --help for usage.\n", stderr)
            exit(1)
        }
    }

    private static func parseArguments(_ rawArguments: [String]) throws -> BenchmarkArguments {
        var arguments = BenchmarkArguments()
        var index = 0

        func requireValue(_ name: String) throws -> String {
            guard index + 1 < rawArguments.count else {
                throw BenchmarkError.missingValue(name)
            }
            index += 1
            return rawArguments[index]
        }

        func parseInteger(_ name: String) throws -> Int {
            let value = try requireValue(name)
            guard let parsed = Int(value) else {
                throw BenchmarkError.invalidInteger(name: name, value: value)
            }
            return parsed
        }

        while index < rawArguments.count {
            let argument = rawArguments[index]
            switch argument {
            case "--model":
                arguments.modelPath = try requireValue(argument)
            case "--system":
                arguments.systemPrompt = try requireValue(argument)
            case "--prompt":
                arguments.prompt = try requireValue(argument)
            case "--prompt-file":
                let path = try requireValue(argument)
                do {
                    arguments.prompt = try String(contentsOfFile: path, encoding: .utf8)
                } catch {
                    throw BenchmarkError.unreadablePromptFile(path, error.localizedDescription)
                }
            case "--warm-suffix":
                arguments.warmSuffix = try requireValue(argument)
            case "--context":
                arguments.contextSize = try parseInteger(argument)
            case "--max-output":
                arguments.maxOutputTokens = try parseInteger(argument)
            case "--gpu-layers":
                let value = try parseInteger(argument)
                guard let converted = Int32(exactly: value) else {
                    throw BenchmarkError.invalidRange(name: argument, value: value, expected: "a 32-bit signed integer")
                }
                arguments.gpuLayerCount = converted
            case "--batch-size":
                arguments.batchSizeLimit = try parseInteger(argument)
            case "--threads":
                let value = try parseInteger(argument)
                guard let converted = Int32(exactly: value) else {
                    throw BenchmarkError.invalidRange(name: argument, value: value, expected: "a positive 32-bit integer")
                }
                arguments.threadCount = converted
            case "--no-mmap":
                arguments.useMemoryMap = false
            case "--enable-mtp":
                arguments.accelerationPolicy = .automatic
            case "--disable-mtp":
                arguments.accelerationPolicy = .disabled
            case "--mtp-max-draft":
                arguments.mtpMaxDraftTokens = try parseInteger(argument)
            case "--json":
                arguments.jsonOutput = true
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                throw BenchmarkError.unknownArgument(argument)
            }
            index += 1
        }

        try validate(arguments)
        return arguments
    }

    private static func validate(_ arguments: BenchmarkArguments) throws {
        guard !arguments.modelPath.isEmpty else {
            throw BenchmarkError.missingModel
        }

        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: arguments.modelPath, isDirectory: &isDirectory),
              !isDirectory.boolValue,
              FileManager.default.isReadableFile(atPath: arguments.modelPath) else {
            throw BenchmarkError.modelNotFound(arguments.modelPath)
        }

        guard arguments.contextSize >= LlamaContextPolicy.minimumContext else {
            throw BenchmarkError.invalidRange(
                name: "--context",
                value: arguments.contextSize,
                expected: "at least \(LlamaContextPolicy.minimumContext)"
            )
        }
        guard arguments.maxOutputTokens > 1 else {
            throw BenchmarkError.invalidRange(name: "--max-output", value: arguments.maxOutputTokens, expected: "at least 2")
        }
        guard arguments.batchSizeLimit > 0 else {
            throw BenchmarkError.invalidRange(name: "--batch-size", value: arguments.batchSizeLimit, expected: "a positive integer")
        }
        if let threadCount = arguments.threadCount, threadCount <= 0 {
            throw BenchmarkError.invalidRange(name: "--threads", value: Int(threadCount), expected: "a positive integer")
        }
        guard (1...32).contains(arguments.mtpMaxDraftTokens) else {
            throw BenchmarkError.invalidRange(name: "--mtp-max-draft", value: arguments.mtpMaxDraftTokens, expected: "1 through 32")
        }
    }

    private static func benchmark(_ arguments: BenchmarkArguments) async throws -> BenchmarkReport {
        let modelURL = URL(fileURLWithPath: arguments.modelPath).standardizedFileURL
        let configuration = LlamaEngineConfiguration(
            gpuLayerCount: arguments.gpuLayerCount,
            useMemoryMap: arguments.useMemoryMap,
            batchSizeLimit: arguments.batchSizeLimit,
            threadCount: arguments.threadCount,
            accelerationPolicy: arguments.accelerationPolicy,
            mtpMaxDraftTokens: arguments.mtpMaxDraftTokens
        )
        let runtimeFingerprint = LlamaEngine.contextCalibrationRuntimeFingerprint(configuration: configuration)
        let engine = LlamaEngine(configuration: configuration)
        let clock = ContinuousClock()

        let loadStart = clock.now
        let loaded = try await engine.load(modelAt: modelURL, requestedContext: arguments.contextSize)
        let loadEnd = clock.now

        let options = GenerationOptions(
            temperature: 0,
            maxOutputTokens: arguments.maxOutputTokens,
            enableThinking: false
        )
        let scenarios = [
            (name: "cold-prefix", condition: "empty prompt KV cache", prompt: arguments.prompt),
            (name: "warm-exact-prefix", condition: "same prompt as prior run", prompt: arguments.prompt),
            (
                name: "warm-extended-prefix",
                condition: "prior prompt plus deterministic suffix",
                prompt: arguments.prompt + arguments.warmSuffix
            )
        ]

        var runs: [RunReport] = []
        runs.reserveCapacity(scenarios.count)
        for scenario in scenarios {
            runs.append(try await benchmarkRun(
                name: scenario.name,
                cacheCondition: scenario.condition,
                engine: engine,
                clock: clock,
                systemPrompt: arguments.systemPrompt,
                prompt: scenario.prompt,
                options: options
            ))
        }

        await engine.unload()

        let processInfo = ProcessInfo.processInfo
        #if DEBUG
        let buildConfiguration = "debug"
        #else
        let buildConfiguration = "release"
        #endif
        return BenchmarkReport(
            schemaVersion: 1,
            generatedAt: ISO8601DateFormatter().string(from: Date()),
            buildConfiguration: buildConfiguration,
            system: SystemReport(
                operatingSystem: processInfo.operatingSystemVersionString,
                activeProcessorCount: processInfo.activeProcessorCount,
                physicalMemoryBytes: processInfo.physicalMemory
            ),
            configuration: ConfigurationReport(
                requestedContextSize: arguments.contextSize,
                maxOutputTokens: arguments.maxOutputTokens,
                requestedGPULayerCount: runtimeFingerprint.gpuLayerCount,
                useMemoryMap: runtimeFingerprint.useMemoryMap,
                requestedBatchSizeLimit: runtimeFingerprint.batchSizeLimit,
                effectiveLogicalBatchSize: loaded.logicalBatchSize,
                effectivePhysicalMicroBatchSize: loaded.physicalMicroBatchSize,
                effectiveThreadCount: runtimeFingerprint.threadCount,
                accelerationPolicy: arguments.accelerationPolicy.rawValue,
                mtpMaxDraftTokens: runtimeFingerprint.mtpMaxDraftTokens,
                promptUTF8Bytes: arguments.prompt.utf8.count,
                warmSuffixUTF8Bytes: arguments.warmSuffix.utf8.count
            ),
            model: ModelReport(
                path: modelURL.path,
                filename: loaded.filename,
                fileSizeBytes: fileSize(at: modelURL),
                architecture: loaded.architecture,
                loadedContextSize: loaded.contextSize,
                trainingContextSize: loaded.trainingContextSize,
                hasEmbeddedChatTemplate: loaded.hasEmbeddedChatTemplate,
                supportsMTPAcceleration: loaded.supportsMTPAcceleration,
                nextNPredictLayers: loaded.nextNPredictLayers
            ),
            modelLoadSeconds: loadStart.duration(to: loadEnd).secondsDouble,
            runs: runs
        )
    }

    private static func benchmarkRun(
        name: String,
        cacheCondition: String,
        engine: LlamaEngine,
        clock: ContinuousClock,
        systemPrompt: String,
        prompt: String,
        options: GenerationOptions
    ) async throws -> RunReport {
        let preflightStart = clock.now
        let preflight = try await engine.preflight(system: systemPrompt, prompt: prompt, options: options)
        let preflightEnd = clock.now

        let recorder = EventRecorder()
        let requestStart = clock.now
        let response = try await engine.generate(
            system: systemPrompt,
            prompt: prompt,
            options: options,
            onPhaseAwareEvent: { event in
                recorder.record(event, at: clock.now)
            },
            ()
        )
        let requestEnd = clock.now
        let snapshot = recorder.snapshot()

        let totalSeconds = requestStart.duration(to: requestEnd).secondsDouble
        let timeToFirstToken = snapshot.firstTokenInstant.map {
            requestStart.duration(to: $0).secondsDouble
        }
        let generatedTokens = snapshot.generatedTokens ?? 0
        let endToEndRate = totalSeconds > 0
            ? Double(generatedTokens) / totalSeconds
            : nil
        let postFirstTokenRate: Double?
        if let firstTokenInstant = snapshot.firstTokenInstant, generatedTokens > 1 {
            let postFirstTokenSeconds = firstTokenInstant.duration(to: requestEnd).secondsDouble
            postFirstTokenRate = postFirstTokenSeconds > 0
                ? Double(generatedTokens - 1) / postFirstTokenSeconds
                : nil
        } else {
            postFirstTokenRate = nil
        }

        return RunReport(
            name: name,
            cacheCondition: cacheCondition,
            preflightSeconds: preflightStart.duration(to: preflightEnd).secondsDouble,
            promptTokens: snapshot.promptTokens ?? preflight.promptTokens,
            effectiveMaxOutputTokens: preflight.effectiveMaxOutputTokens,
            timeToFirstTokenSeconds: timeToFirstToken,
            totalRequestSeconds: totalSeconds,
            runtimeReportedSeconds: snapshot.runtimeReportedSeconds,
            generatedTokens: generatedTokens,
            stopReason: snapshot.stopReason ?? "missing-stats",
            templateMode: snapshot.templateMode?.rawValue ?? preflight.templateMode.rawValue,
            outputUTF8Bytes: response.utf8.count,
            endToEndGeneratedTokensPerSecond: endToEndRate,
            postFirstTokenTokensPerSecond: postFirstTokenRate,
            acceleration: snapshot.accelerationStats.map(AccelerationReport.init)
        )
    }

    private static func fileSize(at url: URL) -> UInt64? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let number = attributes[.size] as? NSNumber else {
            return nil
        }
        return number.uint64Value
    }

    private static func printJSON(_ report: BenchmarkReport) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        guard let json = String(data: data, encoding: .utf8) else { return }
        print(json)
    }

    private static func printHumanReadable(_ report: BenchmarkReport) {
        print("CLLM inference benchmark (schema \(report.schemaVersion))")
        print("build: \(report.buildConfiguration)")
        if report.buildConfiguration != "release" {
            print("warning: rerun with `swift run -c release` before comparing performance")
        }
        print("system: \(report.system.operatingSystem), processors=\(report.system.activeProcessorCount), memory=\(report.system.physicalMemoryBytes)")
        print("model: \(report.model.filename)")
        print("model-path: \(report.model.path)")
        print("context: requested=\(report.configuration.requestedContextSize) loaded=\(report.model.loadedContextSize) training=\(report.model.trainingContextSize)")
        print("runtime: gpuLayersRequested=\(report.configuration.requestedGPULayerCount) batchLimit=\(report.configuration.requestedBatchSizeLimit) nBatch=\(formatted(report.configuration.effectiveLogicalBatchSize)) nUbatch=\(formatted(report.configuration.effectivePhysicalMicroBatchSize)) threads=\(report.configuration.effectiveThreadCount) mmap=\(report.configuration.useMemoryMap) mtp=\(report.configuration.accelerationPolicy) mtpDraft=\(report.configuration.mtpMaxDraftTokens)")
        print(String(format: "model-load: %.6fs", report.modelLoadSeconds))

        for run in report.runs {
            print("run: \(run.name) (\(run.cacheCondition))")
            print(String(
                format: "  preflight=%.6fs promptTokens=%d effectiveMaxOutput=%d generatedTokens=%d outputBytes=%d stop=%@",
                run.preflightSeconds,
                run.promptTokens,
                run.effectiveMaxOutputTokens,
                run.generatedTokens,
                run.outputUTF8Bytes,
                run.stopReason
            ))
            print(String(
                format: "  ttft=%@ total=%.6fs runtimeReported=%@ endToEndTokPerSec=%@ postFirstTokenTokPerSec=%@",
                formatted(run.timeToFirstTokenSeconds),
                run.totalRequestSeconds,
                formatted(run.runtimeReportedSeconds),
                formatted(run.endToEndGeneratedTokensPerSecond),
                formatted(run.postFirstTokenTokensPerSecond)
            ))
            if let acceleration = run.acceleration {
                print(String(
                    format: "  acceleration=%@ accelerator=%@ maxDraft=%d calls=%d drafted=%d accepted=%d acceptance=%@",
                    acceleration.status,
                    acceleration.accelerator,
                    acceleration.maxDraftTokens,
                    acceleration.draftCalls,
                    acceleration.draftTokensGenerated,
                    acceleration.draftTokensAccepted,
                    formatted(acceleration.acceptanceRate)
                ))
            }
        }

        print("note: postFirstTokenTokPerSec is derived as (generatedTokens - 1) / (requestEnd - firstToken).")
        print("note: compare Release runs on the same idle hardware, model file, prompt, and runtime settings.")
    }

    private static func formatted(_ value: Double?) -> String {
        value.map { String(format: "%.6f", $0) } ?? "n/a"
    }

    private static func formatted(_ value: Int?) -> String {
        value.map(String.init) ?? "n/a"
    }

    private static func printUsage() {
        print("""
        Usage:
          CLLM_BENCHMARK_MODEL_PATH=/absolute/model.gguf swift run -c release CLLMBenchmarkCommand [options]
          swift run -c release CLLMBenchmarkCommand --model /absolute/model.gguf [options]

        The command performs three measurement scenarios with one loaded context and no pass/fail thresholds:
          cold-prefix          empty prompt KV cache
          warm-exact-prefix    exact repeat of the first prompt
          warm-extended-prefix first prompt plus a deterministic suffix

        Options:
          --model PATH             GGUF model path (or CLLM_BENCHMARK_MODEL_PATH)
          --system TEXT            system prompt
          --prompt TEXT            benchmark prompt (or CLLM_BENCHMARK_PROMPT)
          --prompt-file PATH       read the benchmark prompt from a UTF-8 file
          --warm-suffix TEXT       suffix used by the warm-extended-prefix run
          --context N              requested context size (default: 4096)
          --max-output N           maximum generated tokens (default: 128)
          --gpu-layers N           llama GPU layer count (platform default when omitted)
          --batch-size N           context batch-size limit (platform default when omitted)
          --threads N              decode and batch thread count (runtime default when omitted)
          --no-mmap                disable memory-mapped model loading
          --enable-mtp             enable MTP acceleration for supported models
          --disable-mtp            disable MTP acceleration (default)
          --mtp-max-draft N        MTP maximum draft tokens, 1...32 (runtime default when omitted)
          --json                   emit a stable, machine-readable JSON report
          --help                   show this help

        This is an opt-in measurement harness, not a pass/fail test. It makes no timing assertions.
        """)
    }
}

private extension Duration {
    var secondsDouble: Double {
        let components = self.components
        return Double(components.seconds) + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }
}
