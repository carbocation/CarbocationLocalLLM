import CarbocationLocalLLM
import CarbocationLocalLLMRuntime
import CarbocationLocalLLMUI
import Observation
import SwiftUI

@main
struct CLLMDemoIOSApp: App {
    @State private var state = SmokeDemoState()

    var body: some Scene {
        WindowGroup {
            SmokeRootView(state: state)
        }
    }
}

private struct SmokeRootView: View {
    @Bindable var state: SmokeDemoState

    var body: some View {
        TabView {
            NavigationStack {
                ModelLibraryPickerView(
                    library: state.library,
                    selectedModelID: $state.selectedModelID,
                    title: "CLLM Demo iOS",
                    confirmTitle: "Use Model",
                    confirmDisabled: state.isRunning,
                    systemModels: state.systemModels,
                    onConfirmSelection: { selection in
                        state.select(selection)
                    }
                )
                .navigationTitle("Models")
                .navigationBarTitleDisplayMode(.inline)
            }
            .tabItem {
                Label("Models", systemImage: "cpu")
            }

            NavigationStack {
                PromptPane(state: state)
                    .navigationTitle("Prompt")
                    .navigationBarTitleDisplayMode(.inline)
            }
            .tabItem {
                Label("Prompt", systemImage: "text.bubble")
            }
        }
        .onChange(of: state.selectedModelID) { _, newValue in
            state.persistSelection(newValue)
        }
    }
}

private struct PromptPane: View {
    @Bindable var state: SmokeDemoState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                selectedModelStatus
                promptEditor
                controls
                outputSection
                eventsSection
            }
            .padding()
        }
        .scrollDismissesKeyboard(.interactively)
    }

    @ViewBuilder
    private var selectedModelStatus: some View {
        if let loadedInfo = state.loadedInfo {
            Label(loadedInfo.displayName, systemImage: "checkmark.circle")
                .font(.headline)
                .foregroundStyle(.green)
        } else if let selectedLabel = state.selectedModelLabel {
            Label(selectedLabel, systemImage: "circle")
                .font(.headline)
                .foregroundStyle(.secondary)
        } else {
            Label("No model selected", systemImage: "exclamationmark.circle")
                .font(.headline)
                .foregroundStyle(.orange)
        }

        if let errorMessage = state.errorMessage {
            Text(errorMessage)
                .font(.callout)
                .foregroundStyle(.red)
        }
    }

    private var promptEditor: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("System")
                .font(.headline)
            TextEditor(text: $state.systemPrompt)
                .frame(minHeight: 80)
                .textInputAutocapitalization(.sentences)
                .autocorrectionDisabled()
                .padding(8)
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))

            Text("Prompt")
                .font(.headline)
                .padding(.top, 8)
            TextEditor(text: $state.prompt)
                .frame(minHeight: 150)
                .textInputAutocapitalization(.sentences)
                .padding(8)
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))
        }
    }

    private var controls: some View {
        HStack {
            Button {
                state.run()
            } label: {
                Label(state.isRunning ? "Running" : "Run", systemImage: "play.fill")
            }
            .buttonStyle(.borderedProminent)
            .disabled(state.isRunning || !state.canRun)

            Button(role: .destructive) {
                state.cancel()
            } label: {
                Label("Cancel", systemImage: "stop.fill")
            }
            .buttonStyle(.bordered)
            .disabled(!state.isRunning)

            Spacer()

            Button {
                state.clear()
            } label: {
                Label("Clear", systemImage: "xmark.circle")
            }
            .buttonStyle(.bordered)
            .disabled(state.isRunning || (state.output.isEmpty && state.events.isEmpty))
        }
    }

    private var outputSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Output")
                .font(.headline)
            Text(state.output.isEmpty ? "No output yet." : state.output)
                .font(.body)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, minHeight: 140, alignment: .topLeading)
                .padding(12)
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))
        }
    }

    private var eventsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Events")
                .font(.headline)
            Text(state.events.isEmpty ? "No events yet." : state.events)
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, minHeight: 120, alignment: .topLeading)
                .padding(12)
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))
        }
    }
}

@MainActor
@Observable
private final class SmokeDemoState {
    private static let requestedContext = 2_048

    let library: ModelLibrary

    var selectedModelID: String
    var systemPrompt = "You are concise and helpful."
    var prompt = "Write one sentence about local on-device language models."
    var output = ""
    var events = ""
    var errorMessage: String?
    var loadedInfo: LocalLLMLoadedModelInfo?
    var isRunning = false

    private var generationTask: Task<Void, Never>?
    private var activeEngine: LocalLLMEngine?

    init() {
        let root = ModelStorage.modelsDirectory(appSupportFolderName: "CLLMDemoIOS")
        library = ModelLibrary(
            root: root,
            contextLengthProbe: { url in
                LocalLLMEngine.probeTrainingContext(at: url)
            }
        )
        selectedModelID = UserDefaults.standard.string(forKey: "CLLMDemoIOS.selectedModelID") ?? ""
        normalizeSelection()
    }

    var systemModels: [LLMSystemModelOption] {
        LocalLLMEngine.availableSystemModels()
    }

    var selectedModelLabel: String? {
        if let systemModel = systemModels.first(where: { $0.id == selectedModelID }) {
            return systemModel.displayName
        }
        return library.model(id: selectedModelID)?.displayName
    }

    var canRun: Bool {
        LLMModelSelection(storageValue: selectedModelID) != nil
            && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func select(_ selection: LLMModelSelection) {
        selectedModelID = selection.storageValue
        loadedInfo = nil
        errorMessage = nil
        persistSelection(selectedModelID)
    }

    func persistSelection(_ value: String) {
        UserDefaults.standard.set(value, forKey: "CLLMDemoIOS.selectedModelID")
    }

    func run() {
        guard !isRunning,
              let selection = LLMModelSelection(storageValue: selectedModelID)
        else { return }

        isRunning = true
        output = ""
        events = ""
        errorMessage = nil
        loadedInfo = nil

        let engine = LocalLLMEngine(configuration: LocalLLMEngineConfiguration(
            heartbeatInterval: 0.5
        ))
        activeEngine = engine

        generationTask = Task { @MainActor [weak self] in
            await self?.run(selection: selection, engine: engine)
        }
    }

    func cancel() {
        generationTask?.cancel()
        generationTask = nil
        isRunning = false
        appendEvent("cancelled")

        let engine = activeEngine
        activeEngine = nil
        Task {
            await engine?.unload()
        }
    }

    func clear() {
        output = ""
        events = ""
        errorMessage = nil
    }

    private func run(selection: LLMModelSelection, engine: LocalLLMEngine) async {
        defer {
            isRunning = false
            generationTask = nil
            activeEngine = nil
        }

        do {
            let loaded = try await engine.load(
                selection: selection,
                from: library,
                requestedContext: Self.requestedContext
            )
            loadedInfo = loaded

            appendEvent("model: \(loaded.displayName)")
            appendEvent("context: \(loaded.contextSize)")
            appendEvent("grammar: \(loaded.supportsGrammar ? "yes" : "no")")

            let response = try await engine.generate(
                system: systemPrompt,
                prompt: prompt,
                options: GenerationOptions(
                    temperature: loaded.supportsGrammar ? 0 : nil,
                    maxOutputTokens: 256,
                    stopAtBalancedJSON: false
                )
            ) { [weak self] event in
                Task { @MainActor [weak self] in
                    self?.appendEvent(Self.format(event: event))
                }
            }

            output = response
            await engine.unload()
            appendEvent("done")
        } catch is CancellationError {
            appendEvent("cancelled")
            await engine.unload()
        } catch {
            errorMessage = error.localizedDescription
            appendEvent("failed: \(error.localizedDescription)")
            await engine.unload()
        }
    }

    private func normalizeSelection() {
        if selectedModelID.isEmpty {
            selectedModelID = systemModels.first?.id ?? library.models.first?.id.uuidString ?? ""
            return
        }

        if LLMModelSelection(storageValue: selectedModelID) == nil {
            selectedModelID = systemModels.first?.id ?? library.models.first?.id.uuidString ?? ""
        }
    }

    private func appendEvent(_ line: String) {
        if events.isEmpty {
            events = line
        } else {
            events += "\n\(line)"
        }
    }

    private static func format(event: LLMStreamEvent) -> String {
        switch event {
        case .requestSent:
            return "event: request-sent"
        case .firstByteReceived(let seconds):
            return String(format: "event: first-byte %.3fs", seconds)
        case .tokenChunk(let preview, let bytesSoFar):
            return "event: token-chunk bytes=\(bytesSoFar) preview=\(preview)"
        case .generationStats(let promptTokens, let generatedTokens, let stopReason, let templateMode):
            return "event: stats promptTokens=\(promptTokens) generatedTokens=\(generatedTokens) stopReason=\(stopReason) templateMode=\(templateMode.rawValue)"
        case .done(let totalBytes, let duration):
            return String(format: "event: done bytes=%d duration=%.3fs", totalBytes, duration)
        }
    }
}
