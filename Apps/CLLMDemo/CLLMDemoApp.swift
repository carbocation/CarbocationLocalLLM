import CarbocationLocalLLM
import CarbocationLocalLLMRuntime
import CarbocationLocalLLMUI
import Observation
import SwiftUI

@main
struct CLLMDemoApp: App {
    @State private var state = DemoState()

    var body: some Scene {
        WindowGroup {
            DemoRootView(state: state)
        }
    }
}

private enum CLLMDemoMetadata {
#if os(macOS)
    static let displayName = "CLLMDemoMac"
#else
    static let displayName = "CLLMDemoIOS"
#endif
    static let appSupportFolderName = "CLLMDemo"
    static let selectedModelDefaultsKey = "CLLMDemo.selectedModelID"
}

private struct DemoRootView: View {
    @Bindable var state: DemoState

    var body: some View {
        TabView {
            NavigationStack {
                ModelLibraryPickerView(
                    library: state.library,
                    selectedModelID: $state.selectedModelID,
                    title: CLLMDemoMetadata.displayName,
                    confirmTitle: "Use Model",
                    confirmDisabled: state.isRunning,
                    systemModels: state.systemModels,
                    onConfirmSelection: { selection in
                        state.select(selection)
                    }
                )
                .navigationTitle("Models")
                .demoNavigationTitleDisplayModeInline()
            }
            .tabItem {
                Label("Models", systemImage: "cpu")
            }

            NavigationStack {
                PromptPane(state: state)
                    .navigationTitle("Prompt")
                    .demoNavigationTitleDisplayModeInline()
            }
            .tabItem {
                Label("Prompt", systemImage: "text.bubble")
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

private struct PromptPane: View {
    @Bindable var state: DemoState

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
        .demoScrollDismissesKeyboardInteractively()
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
                .demoTextEditorInput()
                .padding(8)
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))

            Text("Prompt")
                .font(.headline)
                .padding(.top, 8)
            TextEditor(text: $state.prompt)
                .frame(minHeight: 150)
                .demoTextEditorInput()
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
private final class DemoState {
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
    private let engine = LocalLLMEngine(configuration: LocalLLMEngineConfiguration(
        heartbeatInterval: 0.5
    ))

    init() {
        let root = ModelStorage.modelsDirectory(appSupportFolderName: CLLMDemoMetadata.appSupportFolderName)
        library = ModelLibrary(
            root: root,
            contextLengthProbe: { url in
                LocalLLMEngine.probeTrainingContext(at: url)
            }
        )
        selectedModelID = UserDefaults.standard.string(forKey: CLLMDemoMetadata.selectedModelDefaultsKey) ?? ""
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
        Task { [engine] in
            await engine.unload()
        }
    }

    func persistSelection(_ value: String) {
        UserDefaults.standard.set(value, forKey: CLLMDemoMetadata.selectedModelDefaultsKey)
    }

    func refreshLibrary() async {
        await library.refresh()
        normalizeSelection()
    }

    func run() {
        guard !isRunning, !selectedModelID.isEmpty else { return }

        isRunning = true
        output = ""
        events = ""
        errorMessage = nil

        let storedSelection = selectedModelID
        generationTask = Task { @MainActor [weak self] in
            await self?.run(storedSelection: storedSelection)
        }
    }

    func cancel() {
        generationTask?.cancel()
        generationTask = nil
        isRunning = false
        appendEvent("cancelled")

        loadedInfo = nil
        Task { [engine] in
            await engine.unload()
        }
    }

    func clear() {
        output = ""
        events = ""
        errorMessage = nil
    }

    private func run(storedSelection: String) async {
        defer {
            isRunning = false
            generationTask = nil
        }

        do {
            guard let plan = await LocalLLMEngine.loadPlan(from: storedSelection, in: library) else {
                normalizeSelection()
                errorMessage = "Selected model is unavailable. Pick a model in Settings."
                appendEvent("failed: selected model unavailable")
                await engine.unload()
                return
            }

            let loaded = try await engine.load(
                selection: plan.selection,
                from: library,
                requestedContext: plan.requestedContext
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
            appendEvent("done")
        } catch is CancellationError {
            appendEvent("cancelled")
            loadedInfo = nil
            await engine.unload()
        } catch {
            errorMessage = error.localizedDescription
            appendEvent("failed: \(error.localizedDescription)")
            loadedInfo = nil
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

private extension View {
    @ViewBuilder
    func demoNavigationTitleDisplayModeInline() -> some View {
#if os(iOS)
        navigationBarTitleDisplayMode(.inline)
#else
        self
#endif
    }

    @ViewBuilder
    func demoScrollDismissesKeyboardInteractively() -> some View {
#if os(iOS)
        scrollDismissesKeyboard(.interactively)
#else
        self
#endif
    }

    @ViewBuilder
    func demoTextEditorInput() -> some View {
#if os(iOS)
        textInputAutocapitalization(.sentences)
            .autocorrectionDisabled()
#else
        self
#endif
    }
}
