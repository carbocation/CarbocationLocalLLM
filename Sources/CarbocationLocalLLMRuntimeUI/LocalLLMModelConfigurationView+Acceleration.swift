import CarbocationLocalLLM
import CarbocationLocalLLMRuntime
import SwiftUI

extension LocalLLMModelConfigurationView {
    static let mtpDraftTokenRange = 1...32

    var effectiveConfiguration: LocalLLMEngineConfiguration {
        var effective = configuration
        if let accelerationPolicyBinding {
            effective.accelerationPolicy = accelerationPolicyBinding.wrappedValue
        }
        if let mtpMaxDraftTokensBinding {
            effective.mtpMaxDraftTokens = Self.sanitizedMTPDraftTokens(
                mtpMaxDraftTokensBinding.wrappedValue
            )
        }
        return effective
    }

    var mtpAccelerationEnabled: Bool {
        accelerationPolicyBinding?.wrappedValue == .automatic
    }

    var accelerationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Toggle(
                isOn: Binding(
                    get: { mtpAccelerationEnabled },
                    set: { setMTPAccelerationEnabled($0) }
                )
            ) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("MTP Speculative Decoding")
                        .font(.headline)
                    Text("Use MTP automatically when a compatible GGUF advertises support.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .disabled(activeCalibrationModelID != nil)

            if mtpAccelerationEnabled, mtpMaxDraftTokensBinding != nil {
                Stepper(
                    value: Binding(
                        get: { sanitizedBoundMTPDraftTokens },
                        set: { setMTPMaxDraftTokens($0) }
                    ),
                    in: Self.mtpDraftTokenRange
                ) {
                    HStack {
                        Text("Maximum draft tokens")
                        Spacer()
                        Text(sanitizedBoundMTPDraftTokens.formatted())
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
                .disabled(activeCalibrationModelID != nil)
            }

            Text("Changes take effect on the next model load. MTP can increase or decrease throughput depending on the model, quantization, prompt, and hardware.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    var sanitizedBoundMTPDraftTokens: Int {
        Self.sanitizedMTPDraftTokens(
            mtpMaxDraftTokensBinding?.wrappedValue ?? configuration.mtpMaxDraftTokens
        )
    }

    static func sanitizedMTPDraftTokens(_ value: Int) -> Int {
        min(max(value, mtpDraftTokenRange.lowerBound), mtpDraftTokenRange.upperBound)
    }

    func setMTPAccelerationEnabled(_ enabled: Bool) {
        guard activeCalibrationModelID == nil, let accelerationPolicyBinding else { return }
        let policy: LLMAccelerationPolicy = enabled ? .automatic : .disabled
        guard accelerationPolicyBinding.wrappedValue != policy else { return }
        accelerationPolicyBinding.wrappedValue = policy
        refreshCalibrationForRuntimeConfigurationChange()
    }

    func setMTPMaxDraftTokens(_ value: Int) {
        guard activeCalibrationModelID == nil, let mtpMaxDraftTokensBinding else { return }
        let sanitized = Self.sanitizedMTPDraftTokens(value)
        guard mtpMaxDraftTokensBinding.wrappedValue != sanitized else { return }
        mtpMaxDraftTokensBinding.wrappedValue = sanitized
        refreshCalibrationForRuntimeConfigurationChange()
    }

    func refreshCalibrationForRuntimeConfigurationChange() {
        calibrationErrorMessage = nil
        refreshToken = UUID()
    }
}
