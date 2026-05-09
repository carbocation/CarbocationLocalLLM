import CarbocationLocalLLM
import Foundation
import Observation
import SwiftUI

@MainActor
@Observable
final class ModelLibraryDownload {
    var progress = DownloadProgress(bytesReceived: 0, totalBytes: 0, bytesPerSecond: 0)
    var isRunning = false
    var errorMessage: String?

    let reference: HuggingFaceModelReference
    let resolved: HuggingFaceResolution?
    let displayName: String
    let bearerToken: String?

    @ObservationIgnored
    private var task: Task<Void, Never>?

    init(
        reference: HuggingFaceModelReference,
        resolved: HuggingFaceResolution? = nil,
        displayName: String,
        bearerToken: String? = nil
    ) {
        self.reference = reference
        self.resolved = resolved
        self.displayName = displayName
        self.bearerToken = bearerToken
    }

    func start(
        expectedSHA256: String?,
        expectedContextLength: Int,
        source: ModelSource,
        into library: ModelLibrary,
        completion: @escaping @MainActor (Result<InstalledModel, Error>) -> Void
    ) {
        guard !isRunning else { return }
        isRunning = true
        errorMessage = nil

        task = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let resolution: HuggingFaceResolution
                if let resolved {
                    resolution = resolved
                } else {
                    resolution = try await HuggingFaceModelResolver(endpoint: reference.endpoint)
                        .resolve(reference, token: bearerToken)
                }
                let resolvedDisplayName = displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    ? resolution.displayName
                    : displayName
                let result = try await ModelDownloader.download(
                    resolution: resolution,
                    modelsRoot: library.root,
                    expectedPrimarySHA256: expectedSHA256,
                    bearerToken: bearerToken,
                    onProgress: { [weak self] progress in
                        Task { @MainActor [weak self] in
                            self?.progress = progress
                        }
                    }
                )
                let model = try await library.add(
                    artifacts: result.artifacts,
                    displayName: resolvedDisplayName,
                    source: source,
                    hfRepo: resolution.reference.repo,
                    hfFilename: resolution.primaryArtifact.path,
                    sha256: result.primarySHA256,
                    contextLength: expectedContextLength,
                    quantization: resolution.quantization
                )
                isRunning = false
                completion(.success(model))
            } catch is CancellationError {
                isRunning = false
                errorMessage = ModelDownloaderError.cancelled.localizedDescription
                completion(.failure(ModelDownloaderError.cancelled))
            } catch {
                isRunning = false
                errorMessage = error.localizedDescription
                completion(.failure(error))
            }
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
        isRunning = false
    }
}

struct CustomHFSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var input = ""
    @State private var displayName = ""
    @State private var token = ""
    @State private var rememberToken = false
    @State private var resolution: HuggingFaceResolution?
    @State private var resolvedInput = ""
    @State private var isResolving = false
    @State private var parseError: String?

    let onSubmit: (_ resolution: HuggingFaceResolution, _ displayName: String, _ token: String?) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Download from Hugging Face")
                .font(.title3.bold())

            TextField(
                "owner/repo[:quant] or https://huggingface.co/owner/repo/resolve/main/file.gguf",
                text: $input,
                axis: .vertical
            )
            .textFieldStyle(.roundedBorder)
            .lineLimit(2...4)
            .onChange(of: input) { _, _ in
                if input != resolvedInput {
                    resolution = nil
                }
            }

            TextField("Display name", text: $displayName)
                .textFieldStyle(.roundedBorder)

            SecureField("Hugging Face token", text: $token)
                .textFieldStyle(.roundedBorder)

            Toggle("Remember token in Keychain", isOn: $rememberToken)
                .font(.callout)

            if let resolution {
                resolvedSummary(resolution)
            }

            if let parseError {
                Label(parseError, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                    .font(.callout)
            }

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Resolve") {
                    Task { await resolveInput() }
                }
                .disabled(isResolving || input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Button("Download") {
                    Task { await submit() }
                }
                    .buttonStyle(.borderedProminent)
                    .disabled(isResolving || input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 560)
    }

    private func resolvedSummary(_ resolution: HuggingFaceResolution) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(resolution.primaryArtifact.path)
                    .font(.caption.monospaced())
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let quantization = resolution.quantization {
                    Text(quantization)
                        .font(.caption.monospaced())
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(.quaternary, in: .rect(cornerRadius: 3))
                }
            }
            HStack(spacing: 8) {
                Text(formatBytes(resolution.totalSizeBytes))
                if resolution.splitCount > 1 {
                    Text("\(resolution.splitCount) split files")
                }
                if resolution.mmprojArtifact != nil {
                    Text("mmproj included")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(10)
        .background(.quaternary.opacity(0.5), in: .rect(cornerRadius: 6))
    }

    private func resolveInput() async {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let reference = HuggingFaceModelReference.parse(trimmed) else {
            parseError = "Expected owner/repo[:quant], a Hugging Face URL, or owner/repo/file.gguf."
            return
        }

        isResolving = true
        parseError = nil
        defer { isResolving = false }

        do {
            if token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
               let stored = try? HuggingFaceTokenStore.shared.token(for: reference.endpoint) {
                token = stored
            }
            let resolved = try await HuggingFaceModelResolver(endpoint: reference.endpoint)
                .resolve(reference, token: token.nilIfEmpty)
            resolution = resolved
            resolvedInput = trimmed
            if displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                displayName = resolved.displayName
            }
        } catch {
            parseError = error.localizedDescription
        }
    }

    private func submit() async {
        if resolution == nil || resolvedInput != input.trimmingCharacters(in: .whitespacesAndNewlines) {
            await resolveInput()
        }
        guard let resolution else { return }

        if rememberToken, let token = token.nilIfEmpty {
            do {
                try HuggingFaceTokenStore.shared.save(token, for: resolution.reference.endpoint)
            } catch {
                parseError = error.localizedDescription
                return
            }
        }

        let name = displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? resolution.displayName
            : displayName
        onSubmit(resolution, name, token.nilIfEmpty)
        dismiss()
    }

    private func formatBytes(_ bytes: Int64) -> String {
        guard bytes > 0 else { return "Unknown size" }
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: bytes)
    }
}

private extension String {
    var nilIfEmpty: String? {
        let trimmed = trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
