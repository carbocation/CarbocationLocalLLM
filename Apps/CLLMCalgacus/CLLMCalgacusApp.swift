import CarbocationLlamaRuntime
import CarbocationLocalLLM
import CarbocationLocalLLMUI
import Foundation
import Observation
import SwiftUI

@main
struct CLLMCalgacusApp: App {
    @State private var state = CalgacusState()

    var body: some Scene {
        WindowGroup {
            CalgacusRootView(state: state)
        }
    }
}

private enum CalgacusMetadata {
#if os(macOS)
    static let displayName = "CLLMCalgacusMac"
#else
    static let displayName = "CLLMCalgacusIOS"
#endif
    static let appSupportFolderName = "CLLMCalgacus"
    static let selectedModelDefaultsKey = "CLLMCalgacus.selectedModelID"
}

private struct CalgacusRootView: View {
    @Bindable var state: CalgacusState

    var body: some View {
        TabView {
            NavigationStack {
                ModelLibraryPickerView(
                    library: state.library,
                    selectedModelID: $state.selectedModelID,
                    title: CalgacusMetadata.displayName,
                    confirmTitle: "Use Model",
                    confirmDisabled: state.isRunning,
                    systemModels: [],
                    onConfirmSelection: { selection in
                        state.select(selection)
                    }
                )
                .navigationTitle("Models")
                .calgacusNavigationTitleDisplayModeInline()
            }
            .tabItem {
                Label("Models", systemImage: "cpu")
            }

            NavigationStack {
                CalgacusEncodePane(state: state)
                    .navigationTitle("Encode")
                    .calgacusNavigationTitleDisplayModeInline()
            }
            .tabItem {
                Label("Encode", systemImage: "lock.doc")
            }

            NavigationStack {
                CalgacusDecodePane(state: state)
                    .navigationTitle("Decode")
                    .calgacusNavigationTitleDisplayModeInline()
            }
            .tabItem {
                Label("Decode", systemImage: "lock.open")
            }
        }
        .onChange(of: state.selectedModelID) { _, newValue in
            state.persistSelection(newValue)
        }
        .task {
            await state.refreshLibrary()
        }
    }
}

private struct CalgacusEncodePane: View {
    @Bindable var state: CalgacusState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                CalgacusModelStatusView(state: state)
                editor(title: "Cover Prompt", text: $state.encodeCoverPrompt, minHeight: 92)
                editor(title: "Secret Text", text: $state.secretText, minHeight: 130)
                controls
                output(title: "Cover Text", text: state.encodedCoverText)
                CalgacusStatsView(title: "Encode Stats", stats: state.encodeStats)
                output(title: "Trace", text: state.encodeTraceText, monospaced: true)
                output(title: "Events", text: state.encodeEvents, monospaced: true)
            }
            .padding()
        }
        .calgacusScrollDismissesKeyboardInteractively()
    }

    private var controls: some View {
        HStack {
            Button {
                state.runEncode()
            } label: {
                Label(state.isRunningEncode ? "Encoding" : "Encode", systemImage: "play.fill")
            }
            .buttonStyle(.borderedProminent)
            .disabled(!state.canRunEncode)

            Button(role: .destructive) {
                state.cancel()
            } label: {
                Label("Cancel", systemImage: "stop.fill")
            }
            .buttonStyle(.bordered)
            .disabled(!state.isRunningEncode)

            Spacer()

            Button {
                state.clearEncode()
            } label: {
                Label("Clear", systemImage: "xmark.circle")
            }
            .buttonStyle(.bordered)
            .disabled(state.isRunning || state.encodeIsEmpty)
        }
    }

    private func editor(title: String, text: Binding<String>, minHeight: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            TextEditor(text: text)
                .frame(minHeight: minHeight)
                .calgacusTextEditorInput()
                .padding(8)
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))
        }
    }

    private func output(title: String, text: String, monospaced: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            Text(text.isEmpty ? "No output yet." : text)
                .font(monospaced ? .system(.caption, design: .monospaced) : .body)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, minHeight: monospaced ? 120 : 110, alignment: .topLeading)
                .padding(12)
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))
        }
    }
}

private struct CalgacusDecodePane: View {
    @Bindable var state: CalgacusState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                CalgacusModelStatusView(state: state)
                editor(title: "Cover Prompt", text: $state.decodeCoverPrompt, minHeight: 92)
                editor(title: "Cover Text", text: $state.decodeCoverText, minHeight: 130)
                controls
                output(title: "Recovered Secret", text: state.decodedSecretText)
                CalgacusStatsView(title: "Decode Stats", stats: state.decodeStats)
                output(title: "Trace", text: state.decodeTraceText, monospaced: true)
                output(title: "Events", text: state.decodeEvents, monospaced: true)
            }
            .padding()
        }
        .calgacusScrollDismissesKeyboardInteractively()
    }

    private var controls: some View {
        HStack {
            Button {
                state.runDecode()
            } label: {
                Label(state.isRunningDecode ? "Decoding" : "Decode", systemImage: "play.fill")
            }
            .buttonStyle(.borderedProminent)
            .disabled(!state.canRunDecode)

            Button(role: .destructive) {
                state.cancel()
            } label: {
                Label("Cancel", systemImage: "stop.fill")
            }
            .buttonStyle(.bordered)
            .disabled(!state.isRunningDecode)

            Spacer()

            Button {
                state.clearDecode()
            } label: {
                Label("Clear", systemImage: "xmark.circle")
            }
            .buttonStyle(.bordered)
            .disabled(state.isRunning || state.decodeIsEmpty)
        }
    }

    private func editor(title: String, text: Binding<String>, minHeight: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            TextEditor(text: text)
                .frame(minHeight: minHeight)
                .calgacusTextEditorInput()
                .padding(8)
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))
        }
    }

    private func output(title: String, text: String, monospaced: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            Text(text.isEmpty ? "No output yet." : text)
                .font(monospaced ? .system(.caption, design: .monospaced) : .body)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, minHeight: monospaced ? 120 : 110, alignment: .topLeading)
                .padding(12)
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))
        }
    }
}

private struct CalgacusModelStatusView: View {
    let state: CalgacusState

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let loadedInfo = state.loadedInfo {
                Label(loadedInfo.displayName ?? loadedInfo.filename, systemImage: "checkmark.circle")
                    .font(.headline)
                    .foregroundStyle(.green)
            } else if let selectedLabel = state.selectedModelLabel {
                Label(selectedLabel, systemImage: "circle")
                    .font(.headline)
                    .foregroundStyle(.secondary)
            } else {
                Label("No GGUF model selected", systemImage: "exclamationmark.circle")
                    .font(.headline)
                    .foregroundStyle(.orange)
            }

            if let errorMessage = state.errorMessage {
                Text(errorMessage)
                    .font(.callout)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        }
    }
}

private struct CalgacusStatsView: View {
    let title: String
    let stats: CalgacusRankStats?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            Text(stats.map(Self.format) ?? "No stats yet.")
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, minHeight: 96, alignment: .topLeading)
                .padding(12)
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))
        }
    }

    private static func format(_ stats: CalgacusRankStats) -> String {
        String(
            format: "tokens: %d\nmax rank: %d\nmean rank: %.2f\nmedian rank: %.2f\ncumulative NLL: %.3f\naverage NLL: %.3f",
            stats.tokenCount,
            stats.maxRank,
            stats.meanRank,
            stats.medianRank,
            stats.cumulativeNegativeLogProbability,
            stats.averageNegativeLogProbability
        )
    }
}

@MainActor
@Observable
private final class CalgacusState {
    enum Operation {
        case encode
        case decode
    }

    let library: ModelLibrary

    var selectedModelID: String
    var encodeCoverPrompt = "Here it is: the infamous British roasted boar with mint sauce. How to make it perfect."
    var secretText = "The current government has repeatedly failed."
    var encodedCoverText = ""
    var encodeEvents = ""
    var encodeStats: CalgacusRankStats?
    var encodeTraceText = ""

    var decodeCoverPrompt = "Here it is: the infamous British roasted boar with mint sauce. How to make it perfect."
    var decodeCoverText = ""
    var decodedSecretText = ""
    var decodeEvents = ""
    var decodeStats: CalgacusRankStats?
    var decodeTraceText = ""

    var errorMessage: String?
    var loadedInfo: LlamaLoadedModelInfo?
    var activeOperation: Operation?

    private var operationTask: Task<Void, Never>?
    private var activeEngine: LlamaEngine?

    init() {
        let root = ModelStorage.modelsDirectory(appSupportFolderName: CalgacusMetadata.appSupportFolderName)
        library = ModelLibrary(
            root: root,
            contextLengthProbe: { url in
                LlamaEngine.probeTrainingContext(at: url)
            }
        )
        selectedModelID = UserDefaults.standard.string(forKey: CalgacusMetadata.selectedModelDefaultsKey) ?? ""
        normalizeSelection()
    }

    var isRunning: Bool {
        activeOperation != nil
    }

    var isRunningEncode: Bool {
        activeOperation == .encode
    }

    var isRunningDecode: Bool {
        activeOperation == .decode
    }

    var selectedModelLabel: String? {
        library.model(id: selectedModelID)?.displayName
    }

    var canRunEncode: Bool {
        !isRunning
            && selectedInstalledModelID != nil
            && !encodeCoverPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !secretText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var canRunDecode: Bool {
        !isRunning
            && selectedInstalledModelID != nil
            && !decodeCoverPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !decodeCoverText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var encodeIsEmpty: Bool {
        encodedCoverText.isEmpty && encodeEvents.isEmpty && encodeTraceText.isEmpty && encodeStats == nil
    }

    var decodeIsEmpty: Bool {
        decodedSecretText.isEmpty && decodeEvents.isEmpty && decodeTraceText.isEmpty && decodeStats == nil
    }

    func select(_ selection: LLMModelSelection) {
        guard case .installed = selection else { return }
        selectedModelID = selection.storageValue
        loadedInfo = nil
        errorMessage = nil
        persistSelection(selectedModelID)
    }

    func persistSelection(_ value: String) {
        UserDefaults.standard.set(value, forKey: CalgacusMetadata.selectedModelDefaultsKey)
    }

    func refreshLibrary() async {
        await library.refresh()
        normalizeSelection()
    }

    func runEncode() {
        guard canRunEncode else { return }
        activeOperation = .encode
        errorMessage = nil
        encodedCoverText = ""
        encodeEvents = ""
        encodeStats = nil
        encodeTraceText = ""

        let engine = LlamaEngine(configuration: LlamaEngineConfiguration(
            promptReserveTokens: 0,
            heartbeatInterval: 0.5
        ))
        activeEngine = engine

        operationTask = Task { @MainActor [weak self] in
            await self?.encode(using: engine)
        }
    }

    func runDecode() {
        guard canRunDecode else { return }
        activeOperation = .decode
        errorMessage = nil
        decodedSecretText = ""
        decodeEvents = ""
        decodeStats = nil
        decodeTraceText = ""

        let engine = LlamaEngine(configuration: LlamaEngineConfiguration(
            promptReserveTokens: 0,
            heartbeatInterval: 0.5
        ))
        activeEngine = engine

        operationTask = Task { @MainActor [weak self] in
            await self?.decode(using: engine)
        }
    }

    func cancel() {
        operationTask?.cancel()
        operationTask = nil

        switch activeOperation {
        case .encode:
            appendEncodeEvent("cancelled")
        case .decode:
            appendDecodeEvent("cancelled")
        case nil:
            break
        }

        activeOperation = nil
        let engine = activeEngine
        activeEngine = nil
        Task {
            await engine?.unload()
        }
    }

    func clearEncode() {
        encodedCoverText = ""
        encodeEvents = ""
        encodeStats = nil
        encodeTraceText = ""
        if activeOperation != .encode {
            errorMessage = nil
        }
    }

    func clearDecode() {
        decodedSecretText = ""
        decodeEvents = ""
        decodeStats = nil
        decodeTraceText = ""
        if activeOperation != .decode {
            errorMessage = nil
        }
    }

    private func encode(using engine: LlamaEngine) async {
        defer {
            activeOperation = nil
            operationTask = nil
            activeEngine = nil
        }

        do {
            let loaded = try await loadSelectedModel(using: engine)
            let result = try await engine.encodeCalgacus(
                CalgacusEncodeRequest(
                    secretText: secretText,
                    coverPrompt: encodeCoverPrompt,
                    requestedContext: loaded.contextSize
                )
            ) { [weak self] event in
                Task { @MainActor [weak self] in
                    self?.appendEncodeEvent(Self.format(event: event))
                }
            }

            encodedCoverText = result.coverText
            encodeStats = result.stats
            encodeTraceText = Self.format(trace: result.trace)
            decodeCoverPrompt = encodeCoverPrompt
            decodeCoverText = result.coverText
            appendEncodeEvent("done")
            await engine.unload()
        } catch is CancellationError {
            appendEncodeEvent("cancelled")
            await engine.unload()
        } catch {
            errorMessage = error.localizedDescription
            appendEncodeEvent("failed: \(error.localizedDescription)")
            await engine.unload()
        }
    }

    private func decode(using engine: LlamaEngine) async {
        defer {
            activeOperation = nil
            operationTask = nil
            activeEngine = nil
        }

        do {
            let loaded = try await loadSelectedModel(using: engine)
            let result = try await engine.decodeCalgacus(
                CalgacusDecodeRequest(
                    coverText: decodeCoverText,
                    coverPrompt: decodeCoverPrompt,
                    requestedContext: loaded.contextSize
                )
            ) { [weak self] event in
                Task { @MainActor [weak self] in
                    self?.appendDecodeEvent(Self.format(event: event))
                }
            }

            decodedSecretText = result.secretText
            decodeStats = result.stats
            decodeTraceText = Self.format(trace: result.trace)
            appendDecodeEvent("done")
            await engine.unload()
        } catch is CancellationError {
            appendDecodeEvent("cancelled")
            await engine.unload()
        } catch {
            errorMessage = error.localizedDescription
            appendDecodeEvent("failed: \(error.localizedDescription)")
            await engine.unload()
        }
    }

    private func loadSelectedModel(using engine: LlamaEngine) async throws -> LlamaLoadedModelInfo {
        await library.refresh()
        guard let modelID = selectedInstalledModelID else {
            throw CalgacusAppError.noInstalledModelSelected
        }
        guard let model = library.model(id: modelID) else {
            normalizeSelection()
            throw CalgacusAppError.installedModelNotFound(modelID)
        }

        let requestedContext = LlamaContextPolicy.resolvedRequestedContext(for: model)
        let loaded = try await engine.load(
            model: model,
            from: library.root,
            requestedContext: requestedContext
        )
        loadedInfo = loaded

        let line = "model: \(loaded.displayName ?? loaded.filename), context: \(loaded.contextSize)"
        switch activeOperation {
        case .encode:
            appendEncodeEvent(line)
        case .decode:
            appendDecodeEvent(line)
        case nil:
            break
        }
        return loaded
    }

    private var selectedInstalledModelID: UUID? {
        guard let selection = LLMModelSelection(storageValue: selectedModelID),
              case .installed(let id) = selection
        else {
            return nil
        }
        return id
    }

    private func normalizeSelection() {
        if selectedModelID.isEmpty {
            selectedModelID = library.models.first?.id.uuidString ?? ""
            return
        }

        if selectedInstalledModelID == nil {
            selectedModelID = library.models.first?.id.uuidString ?? ""
        }
    }

    private func appendEncodeEvent(_ line: String) {
        if encodeEvents.isEmpty {
            encodeEvents = line
        } else {
            encodeEvents += "\n\(line)"
        }
    }

    private func appendDecodeEvent(_ line: String) {
        if decodeEvents.isEmpty {
            decodeEvents = line
        } else {
            decodeEvents += "\n\(line)"
        }
    }

    private static func format(event: CalgacusEvent) -> String {
        switch event {
        case .started(let operation):
            return "event: \(operation)-started"
        case .tokensPrepared(let operation, let count):
            return "event: \(operation)-tokens count=\(count)"
        case .tokenProcessed(let stage, let index, let total, let rank):
            return "event: \(stage.rawValue) \(index)/\(total) rank=\(rank)"
        case .completed(let operation, let duration):
            return String(format: "event: %@-completed %.3fs", operation, duration)
        }
    }

    private static func format(trace: [CalgacusTraceEntry]) -> String {
        guard !trace.isEmpty else { return "" }

        let visible = trace.prefix(80).map { entry in
            let text = entry.tokenText
                .replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\t", with: "\\t")
            return String(
                format: "#%03d token=%d rank=%d nll=%.3f text=%@",
                entry.index + 1,
                entry.tokenID,
                entry.rank,
                entry.negativeLogProbability,
                text
            )
        }

        if trace.count > visible.count {
            return (visible + ["... \(trace.count - visible.count) more"]).joined(separator: "\n")
        }
        return visible.joined(separator: "\n")
    }
}

private enum CalgacusAppError: Error, LocalizedError {
    case noInstalledModelSelected
    case installedModelNotFound(UUID)

    var errorDescription: String? {
        switch self {
        case .noInstalledModelSelected:
            return "Pick an installed GGUF model before running Calgacus."
        case .installedModelNotFound(let id):
            return "Installed model was not found: \(id.uuidString)"
        }
    }
}

private extension View {
    @ViewBuilder
    func calgacusNavigationTitleDisplayModeInline() -> some View {
#if os(iOS)
        navigationBarTitleDisplayMode(.inline)
#else
        self
#endif
    }

    @ViewBuilder
    func calgacusScrollDismissesKeyboardInteractively() -> some View {
#if os(iOS)
        scrollDismissesKeyboard(.interactively)
#else
        self
#endif
    }

    @ViewBuilder
    func calgacusTextEditorInput() -> some View {
#if os(iOS)
        textInputAutocapitalization(.sentences)
            .autocorrectionDisabled()
#else
        self
#endif
    }
}
