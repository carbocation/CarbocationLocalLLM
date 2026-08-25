import Foundation
import XCTest

enum LiveModelTestProfile {
    case textPreflight
    case commonChatThinkingAndTools
    case visionKVReset
    case audioTranscription
    case calgacusRoundTrip
    case likelihoodStatistics

    var name: String {
        switch self {
        case .textPreflight:
            return "text-preflight"
        case .commonChatThinkingAndTools:
            return "common-chat-thinking-tools"
        case .visionKVReset:
            return "vision-kv-reset"
        case .audioTranscription:
            return "audio-transcription"
        case .calgacusRoundTrip:
            return "calgacus-round-trip"
        case .likelihoodStatistics:
            return "likelihood-statistics"
        }
    }

    var capabilityContract: String {
        switch self {
        case .textPreflight:
            return "text tokenization, context creation, and grammar validation"
        case .commonChatThinkingAndTools:
            return "an embedded llama.cpp common/chat template with low-effort thinking and named native tool calls"
        case .visionKVReset:
            return "image input through a compatible multimodal projector"
        case .audioTranscription:
            return "audio input and transcription through a compatible multimodal projector"
        case .calgacusRoundTrip:
            return "text generation with the logits needed for a Calgacus encode/decode round trip"
        case .likelihoodStatistics:
            return "token likelihood scoring with opt-in compact full-vocabulary statistics"
        }
    }
}

enum LiveModelTestPolicy {
    static let optInEnvironmentVariable = "CARBOCATION_RUN_LIVE_TESTS"

    static func requireEnabled(
        _ profile: LiveModelTestProfile,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws {
        guard environment[optInEnvironmentVariable] == "1" else {
            throw XCTSkip(
                "Live-model profile '\(profile.name)' is disabled. "
                    + "Run Scripts/run-live-tests.sh \(profile.name) with its required files. "
                    + "Capability contract: \(profile.capabilityContract)."
            )
        }
    }

    static func requiredReadablePath(
        _ variable: String,
        for profile: LiveModelTestProfile,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) throws -> String {
        try requireEnabled(profile, environment: environment)
        guard let value = environment[variable]?.trimmingCharacters(in: .whitespacesAndNewlines),
              !value.isEmpty else {
            throw LiveModelTestConfigurationError(
                "Live-model profile '\(profile.name)' requires \(variable). "
                    + "Capability contract: \(profile.capabilityContract)."
            )
        }
        guard fileManager.isReadableFile(atPath: value) else {
            throw LiveModelTestConfigurationError(
                "Live-model profile '\(profile.name)' cannot read \(variable) at \(value)."
            )
        }
        return value
    }
}

private struct LiveModelTestConfigurationError: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? {
        message
    }
}
