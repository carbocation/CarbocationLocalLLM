import CarbocationLocalLLM
import CarbocationLlamaRuntime
import Darwin
import Foundation

struct ProbeArguments {
    var modelPath = ""
    var system = "You are concise and helpful."
    var prompt = "Write ten sentences about local on-device language models."
    var context = 32_768
    var maxOutput = 64
    var policy = LLMAccelerationPolicy.automatic
    var mtpMaxDraft = 3
    var mtpDiagnostics = false
    var compareDisabled = false
}

final class ReproCommandStore: @unchecked Sendable {
    private let lock = NSLock()
    private var firstValue: String?
    private var lastValue: String?
    private var recordedCount = 0

    func record(_ command: String) {
        lock.withLock {
            if firstValue == nil {
                firstValue = command
            }
            lastValue = command
            recordedCount += 1
        }
    }

    func snapshot() -> (first: String, last: String, count: Int)? {
        lock.withLock {
            guard let firstValue, let lastValue else { return nil }
            return (firstValue, lastValue, recordedCount)
        }
    }
}

struct TokenDiagnosticRecord: Sendable {
    var index: Int
    var position: Int
    var source: String
    var tokenID: Int32
    var tokenDescription: String
}

final class TokenDiagnosticStore: @unchecked Sendable {
    private let lock = NSLock()
    private var records: [TokenDiagnosticRecord] = []

    func record(_ message: String) {
        guard let record = Self.parse(message) else { return }
        lock.withLock {
            records.append(record)
        }
    }

    func snapshot() -> [TokenDiagnosticRecord] {
        lock.withLock { records }
    }

    private static func parse(_ message: String) -> TokenDiagnosticRecord? {
        let prefix = "token-diagnostic generated "
        guard message.hasPrefix(prefix) else { return nil }

        let payload = String(message.dropFirst(prefix.count))
        let tokenMarker = " token="
        guard let tokenRange = payload.range(of: tokenMarker) else { return nil }
        let fieldText = String(payload[..<tokenRange.lowerBound])
        let tokenDescription = String(payload[tokenRange.upperBound...])

        var fields: [String: String] = [:]
        for part in fieldText.split(separator: " ") {
            let pieces = part.split(separator: "=", maxSplits: 1)
            guard pieces.count == 2 else { continue }
            fields[String(pieces[0])] = String(pieces[1])
        }

        guard let indexText = fields["index"],
              let positionText = fields["pos"],
              let source = fields["source"],
              let index = Int(indexText),
              let position = Int(positionText) else {
            return nil
        }

        let tokenIDText = tokenDescription.split(separator: ":", maxSplits: 1).first.map(String.init) ?? ""
        guard let tokenID = Int32(tokenIDText) else { return nil }

        return TokenDiagnosticRecord(
            index: index,
            position: position,
            source: source,
            tokenID: tokenID,
            tokenDescription: tokenDescription
        )
    }
}

struct ProbeRunResult: Sendable {
    var policy: LLMAccelerationPolicy
    var response: String
    var tokens: [TokenDiagnosticRecord]
}

enum RetainedProbeEngines {
    static var engines: [LlamaEngine] = []
}

enum ProbeError: LocalizedError {
    case missingValue(String)
    case unknownArgument(String)
    case missingModel
    case invalidInteger(String, String)
    case invalidPolicy(String)

    var errorDescription: String? {
        switch self {
        case .missingValue(let name):
            return "Missing value for \(name)."
        case .unknownArgument(let name):
            return "Unknown argument: \(name)."
        case .missingModel:
            return "Provide --model PATH."
        case .invalidInteger(let name, let value):
            return "Invalid integer for \(name): \(value)."
        case .invalidPolicy(let value):
            return "Invalid policy for --policy: \(value). Use automatic or disabled."
        }
    }
}

@main
enum CLLMMTPReproCommand {
    static func main() async {
        do {
            let arguments = try parseArguments(Array(CommandLine.arguments.dropFirst()))
            try await run(arguments)
        } catch {
            fputs("error: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }

    private static func parseArguments(_ rawArguments: [String]) throws -> ProbeArguments {
        var arguments = ProbeArguments()
        var index = 0

        func requireValue(_ name: String) throws -> String {
            guard index + 1 < rawArguments.count else {
                throw ProbeError.missingValue(name)
            }
            index += 1
            return rawArguments[index]
        }

        func parseInt(_ name: String) throws -> Int {
            let value = try requireValue(name)
            guard let parsed = Int(value) else {
                throw ProbeError.invalidInteger(name, value)
            }
            return parsed
        }

        func parsePolicy(_ name: String) throws -> LLMAccelerationPolicy {
            let value = try requireValue(name)
            guard let parsed = LLMAccelerationPolicy(rawValue: value) else {
                throw ProbeError.invalidPolicy(value)
            }
            return parsed
        }

        while index < rawArguments.count {
            let argument = rawArguments[index]
            switch argument {
            case "--model":
                arguments.modelPath = try requireValue(argument)
            case "--system":
                arguments.system = try requireValue(argument)
            case "--prompt":
                arguments.prompt = try requireValue(argument)
            case "--context":
                arguments.context = try parseInt(argument)
            case "--max-output":
                arguments.maxOutput = try parseInt(argument)
            case "--policy":
                arguments.policy = try parsePolicy(argument)
            case "--mtp-max-draft":
                arguments.mtpMaxDraft = try parseInt(argument)
            case "--mtp-diagnostics":
                arguments.mtpDiagnostics = true
            case "--compare-disabled":
                arguments.compareDisabled = true
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                throw ProbeError.unknownArgument(argument)
            }
            index += 1
        }

        guard !arguments.modelPath.isEmpty else {
            throw ProbeError.missingModel
        }

        return arguments
    }

    private static func run(_ arguments: ProbeArguments) async throws {
        if arguments.compareDisabled {
            var disabledArguments = arguments
            disabledArguments.policy = .disabled
            disabledArguments.compareDisabled = false
            let disabled = try await runSingle(disabledArguments, label: "disabled", unloadAfterRun: true)

            var automaticArguments = arguments
            automaticArguments.policy = .automatic
            automaticArguments.compareDisabled = false
            let automatic = try await runSingle(automaticArguments, label: "automatic", unloadAfterRun: false)

            printComparison(baseline: disabled, candidate: automatic)
            fflush(stdout)
            Darwin._exit(0)
        }

        _ = try await runSingle(arguments, label: nil, unloadAfterRun: true)
    }

    private static func runSingle(
        _ arguments: ProbeArguments,
        label: String?,
        unloadAfterRun: Bool
    ) async throws -> ProbeRunResult {
        let engine = LlamaEngine(configuration: LlamaEngineConfiguration(
            accelerationPolicy: arguments.policy,
            mtpMaxDraftTokens: arguments.mtpMaxDraft,
            emitsMTPDiagnostics: arguments.mtpDiagnostics
        ))

        let loaded = try await engine.load(
            modelAt: URL(fileURLWithPath: arguments.modelPath),
            requestedContext: arguments.context
        )
        if let label {
            print("run: \(label)")
        }
        print("model: \(loaded.displayName ?? loaded.filename)")
        print("context: \(loaded.contextSize)")
        print("mtp-support: \(loaded.supportsMTPAcceleration ? "yes" : "no")")
        print("mtp-policy: \(arguments.policy.rawValue)")
        print("mtp-max-draft: \(arguments.mtpMaxDraft)")
        print("mtp-diagnostics: \(arguments.mtpDiagnostics ? "yes" : "no")")

        let options = LLMSamplingDefaults.extractionSafe.applying(to: GenerationOptions(
            maxOutputTokens: arguments.maxOutput,
            enableThinking: false,
            thinkingBudgetMessage: "Thinking budget reached."
        ))
        let samplerDiagnostics = LlamaEngine.resolvedSamplerDiagnostics(options: options)
        print("sampler-request: \(samplerDiagnostics.requestLine)")
        print("sampler-resolved: \(samplerDiagnostics.resolvedLine)")

        let reproCommandStore = ReproCommandStore()
        let tokenDiagnosticStore = TokenDiagnosticStore()
        let response = try await engine.generate(
            system: arguments.system,
            prompt: arguments.prompt,
            options: options,
            onPhaseAwareEvent: { event in
                let line = format(event)
                if case .diagnostic(let message) = event {
                    if let command = reproCommand(fromDiagnostic: message) {
                        reproCommandStore.record(command)
                    }
                    tokenDiagnosticStore.record(message)
                }
                print(line)
            }
        )

        print("response:")
        print(response)
        if let capturedReproCommands = reproCommandStore.snapshot() {
            print("repro-command-count: \(capturedReproCommands.count)")
            print("first-repro-command:")
            print(capturedReproCommands.first)
            print("last-repro-command:")
            print(capturedReproCommands.last)
        } else {
            print("repro-command-count: 0")
            print("first-repro-command: none")
            print("last-repro-command: none")
        }
        let tokens = tokenDiagnosticStore.snapshot()
        if !tokens.isEmpty {
            print("token-diagnostic-count: \(tokens.count)")
        }
        if unloadAfterRun {
            await engine.unload()
        } else {
            RetainedProbeEngines.engines.append(engine)
        }
        return ProbeRunResult(
            policy: arguments.policy,
            response: response,
            tokens: tokens
        )
    }

    private static func printComparison(baseline: ProbeRunResult, candidate: ProbeRunResult) {
        print("comparison:")
        print("baseline-policy: \(baseline.policy.rawValue)")
        print("candidate-policy: \(candidate.policy.rawValue)")
        print("baseline-token-count: \(baseline.tokens.count)")
        print("candidate-token-count: \(candidate.tokens.count)")

        let sharedCount = min(baseline.tokens.count, candidate.tokens.count)
        for index in 0..<sharedCount {
            let baselineToken = baseline.tokens[index]
            let candidateToken = candidate.tokens[index]
            if baselineToken.tokenID != candidateToken.tokenID {
                print("first-divergence-index: \(index + 1)")
                print("baseline-token: pos=\(baselineToken.position) source=\(baselineToken.source) token=\(baselineToken.tokenDescription)")
                print("candidate-token: pos=\(candidateToken.position) source=\(candidateToken.source) token=\(candidateToken.tokenDescription)")
                return
            }
        }

        if baseline.tokens.count == candidate.tokens.count {
            print("first-divergence-index: none")
            print("result: committed token IDs match")
        } else {
            print("first-divergence-index: \(sharedCount + 1)")
            print("result: one run ended before the other")
        }
    }

    private static func reproCommand(fromDiagnostic message: String) -> String? {
        let marker = "mtp-diagnostic repro-command "
        guard message.hasPrefix(marker) else { return nil }
        return String(message.dropFirst(marker.count))
    }

    private static func format(_ event: LLMPhaseAwareStreamEvent) -> String {
        switch event {
        case .requestSent(let phase):
            return "event: request-sent phase=\(phase.rawValue)"
        case .firstByteReceived(let seconds, let phase):
            return String(format: "event: first-byte %.3fs phase=%@", seconds, phase.rawValue)
        case .phaseChanged(let from, let to):
            return "event: phase \(from.rawValue) -> \(to.rawValue)"
        case .tokenChunk(let preview, let bytesSoFar, let phase):
            return "event: token-chunk phase=\(phase.rawValue) bytes=\(bytesSoFar) preview=\(preview)"
        case .finalAnswerDelta(let text, let bytesSoFar):
            return "event: final-answer-delta bytes=\(bytesSoFar) text=\(text)"
        case .finalAnswerSnapshot(let text, let bytesSoFar, let reason):
            return "event: final-answer-snapshot reason=\(reason.rawValue) bytes=\(bytesSoFar) text=\(text)"
        case .generationStats(let promptTokens, let generatedTokens, let stopReason, let templateMode, let phase):
            return "event: stats phase=\(phase.rawValue) promptTokens=\(promptTokens) generatedTokens=\(generatedTokens) stopReason=\(stopReason) templateMode=\(templateMode.rawValue)"
        case .accelerationStats(let stats):
            let rate = stats.acceptanceRate
                .map { String(format: "%.1f%%", $0 * 100) }
                ?? "n/a"
            return "event: acceleration accelerator=\(stats.accelerator) status=\(stats.status.rawValue) maxDraftTokens=\(stats.maxDraftTokens) draftCalls=\(stats.draftCalls) draftGenerated=\(stats.draftTokensGenerated) draftAccepted=\(stats.draftTokensAccepted) acceptance=\(rate)"
        case .diagnostic(let message):
            return "event: \(message)"
        case .done(let totalBytes, let duration, let phase):
            return String(format: "event: done phase=%@ bytes=%d duration=%.3fs", phase.rawValue, totalBytes, duration)
        }
    }

    private static func printUsage() {
        print("""
        usage:
          swift run CLLMMTPReproCommand --model PATH [options]

        options:
          --system TEXT
          --prompt TEXT
          --context N
          --max-output N
          --policy automatic|disabled
          --mtp-max-draft N
          --mtp-diagnostics
          --compare-disabled
        """)
    }
}
