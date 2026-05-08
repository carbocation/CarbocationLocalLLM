import CarbocationLocalLLM
import CarbocationLocalLLMRuntime
import CarbocationLocalLLMRuntimeUI
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
                LocalLLMModelConfigurationView(
                    library: state.library,
                    selectedModelID: $state.selectedModelID,
                    title: CLLMDemoMetadata.displayName,
                    confirmTitle: "Use Model",
                    confirmDisabled: state.isRunning,
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
#if os(iOS)
    @FocusState private var focusedEditor: PromptEditorField?
#endif

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                selectedModelStatus
                promptEditor
                generationOptionsEditor
                controls
                streamSection
                outputSection
                eventsSection
            }
            .padding()
        }
        .demoScrollDismissesKeyboardInteractively()
#if os(iOS)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button("Done") {
                    focusedEditor = nil
                }
                .disabled(focusedEditor == nil)
            }
        }
#endif
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
#if os(iOS)
            TextField("System prompt", text: $state.systemPrompt, axis: .vertical)
                .lineLimit(3...6)
                .textFieldStyle(.plain)
                .frame(maxWidth: .infinity, minHeight: 80, alignment: .topLeading)
                .demoTextEditorInput()
                .focused($focusedEditor, equals: .system)
                .padding(8)
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))
#else
            TextEditor(text: $state.systemPrompt)
                .frame(minHeight: 80)
                .demoTextEditorInput()
                .padding(8)
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))
#endif

            Text("Prompt")
                .font(.headline)
                .padding(.top, 8)
#if os(iOS)
            TextField("Prompt", text: $state.prompt, axis: .vertical)
                .lineLimit(6...10)
                .textFieldStyle(.plain)
                .frame(maxWidth: .infinity, minHeight: 150, alignment: .topLeading)
                .demoTextEditorInput()
                .focused($focusedEditor, equals: .prompt)
                .padding(8)
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))
#else
            TextEditor(text: $state.prompt)
                .frame(minHeight: 150)
                .demoTextEditorInput()
                .padding(8)
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))
#endif
        }
    }

    private var generationOptionsEditor: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Generation")
                .font(.headline)

            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text("Max output")
                    .frame(width: 84, alignment: .leading)

                TextField("Context cap", text: $state.maxOutputTokensText)
                    .textFieldStyle(.roundedBorder)
                    .demoNumericInput()
                    .disabled(state.isRunning)
                    .frame(maxWidth: 180)

                Text("tokens")
                    .foregroundStyle(.secondary)
            }

            Toggle("Thinking", isOn: $state.enableThinking)
                .disabled(state.isRunning)

            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text("Budget")
                    .foregroundStyle(state.enableThinking ? .primary : .secondary)
                    .frame(width: 84, alignment: .leading)

                TextField("No cap", text: $state.thinkingBudgetText)
                    .textFieldStyle(.roundedBorder)
                    .demoNumericInput()
                    .disabled(!state.enableThinking || state.isRunning)
                    .frame(maxWidth: 180)

                Text("tokens")
                    .foregroundStyle(.secondary)
            }

            if let message = state.thinkingBudgetValidationMessage {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            if let message = state.maxOutputTokensValidationMessage {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
        .padding(12)
        .background(.quaternary.opacity(0.25), in: .rect(cornerRadius: 8))
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

    private var streamSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Stream")
                .font(.headline)

            HStack(spacing: 12) {
                Label(
                    state.streamPhaseTitle,
                    systemImage: state.streamPhaseSystemImage
                )
                .font(.headline)
                .foregroundStyle(state.streamPhaseColor)

                Spacer()

                if state.streamBytesSoFar > 0 {
                    Text("\(state.streamBytesSoFar) bytes")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(12)
            .background(.quaternary.opacity(0.25), in: .rect(cornerRadius: 8))
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

#if os(iOS)
private enum PromptEditorField: Hashable {
    case system
    case prompt
}
#endif

@MainActor
@Observable
private final class DemoState {
    let library: ModelLibrary

    var selectedModelID: String
    var systemPrompt = "You are concise and helpful."
    var prompt = "Write one sentence about local on-device language models."
    var maxOutputTokensText = "256"
    var enableThinking = false
    var thinkingBudgetText = ""
    var output = ""
    var events = ""
    var streamPhase: LLMStreamContentPhase = .unknown
    var streamBytesSoFar = 0
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
            && generationOptionsValidationMessage == nil
    }

    var generationOptionsValidationMessage: String? {
        maxOutputTokensValidationMessage ?? thinkingBudgetValidationMessage
    }

    var streamPhaseTitle: String {
        switch streamPhase {
        case .unknown:
            return isRunning ? "Detecting phase..." : "Idle"
        case .thinking:
            return isRunning ? "Thinking..." : "Thinking"
        case .final:
            return isRunning ? "Final answer" : "Final"
        }
    }

    var streamPhaseSystemImage: String {
        switch streamPhase {
        case .unknown:
            return isRunning ? "questionmark.circle" : "circle"
        case .thinking:
            return "brain.head.profile"
        case .final:
            return "text.bubble"
        }
    }

    var streamPhaseColor: Color {
        switch streamPhase {
        case .unknown:
            return .secondary
        case .thinking:
            return .orange
        case .final:
            return .green
        }
    }

    var maxOutputTokensValidationMessage: String? {
        let trimmed = maxOutputTokensText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        guard let value = Int(trimmed), value > 0 else {
            return "Max output must be blank or a positive integer."
        }
        return nil
    }

    var thinkingBudgetValidationMessage: String? {
        guard enableThinking else { return nil }

        let trimmed = thinkingBudgetText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        guard let value = Int(trimmed), value >= 0 else {
            return "Thinking budget must be blank, zero, or a positive integer."
        }
        guard value <= Int(Int32.max) else {
            return "Thinking budget is too large."
        }
        return nil
    }

    private var parsedMaxOutputTokens: Int? {
        let trimmed = maxOutputTokensText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return Int(trimmed)
    }

    private var parsedThinkingBudgetTokens: Int? {
        guard enableThinking else { return nil }

        let trimmed = thinkingBudgetText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return Int(trimmed)
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
        guard canRun else {
            errorMessage = generationOptionsValidationMessage ?? "Select a model and enter a prompt before running."
            return
        }

        isRunning = true
        output = ""
        events = ""
        resetStreamState()
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
        resetStreamState()
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

            let options = generationOptions(for: loaded)
            appendGenerationOptionsEvent(options: options, loaded: loaded)

            let response = try await engine.generate(
                system: systemPrompt,
                prompt: prompt,
                options: options,
                onPhaseAwareEvent: { [weak self] event in
                    Task { @MainActor [weak self] in
                        self?.handle(event: event)
                    }
                }
            )

            if output != response {
                output = response
            }
            appendEvent("done")
            await releaseLoadedModel()
        } catch is CancellationError {
            appendEvent("cancelled")
            await releaseLoadedModel()
        } catch {
            errorMessage = error.localizedDescription
            appendEvent("failed: \(error.localizedDescription)")
            await releaseLoadedModel()
        }
    }

    private func releaseLoadedModel() async {
        let hadLoadedModel = loadedInfo != nil
        loadedInfo = nil
        await engine.unload()
        if hadLoadedModel {
            appendEvent("model released")
        }
    }

    private func generationOptions(for loaded: LocalLLMLoadedModelInfo) -> GenerationOptions {
        GenerationOptions(
            temperature: loaded.supportsGrammar ? 0 : nil,
            maxOutputTokens: parsedMaxOutputTokens,
            stopAtBalancedJSON: false,
            enableThinking: enableThinking,
            thinkingBudgetTokens: parsedThinkingBudgetTokens,
            thinkingBudgetMessage: "Thinking budget reached."
        )
    }

    private func appendGenerationOptionsEvent(
        options: GenerationOptions,
        loaded: LocalLLMLoadedModelInfo
    ) {
        let maxOutput = options.maxOutputTokens.map(String.init) ?? "context"
        appendEvent("max-output: \(maxOutput)")

        var line = "thinking: \(options.enableThinking ? "enabled" : "disabled")"
        if options.enableThinking {
            let budget = options.thinkingBudgetTokens.map(String.init) ?? "none"
            line += " budget=\(budget)"
            if case .system = loaded.selection {
                line += " ignored-by-provider"
            }
        }
        appendEvent(line)
    }

    private func handle(event: LLMPhaseAwareStreamEvent) {
        switch event {
        case .requestSent(let phase):
            streamPhase = phase
        case .firstByteReceived(_, let phase):
            streamPhase = phase
        case .phaseChanged(_, let phase):
            streamPhase = phase
        case .tokenChunk(_, let bytesSoFar, let phase):
            streamPhase = phase
            streamBytesSoFar = bytesSoFar
        case .finalAnswerDelta(let text, let bytesSoFar):
            streamPhase = .final
            streamBytesSoFar = bytesSoFar
            output += text
        case .finalAnswerSnapshot(let text, let bytesSoFar, _):
            streamPhase = .final
            streamBytesSoFar = bytesSoFar
            if output != text {
                output = text
            }
        case .generationStats(_, _, _, _, let phase):
            streamPhase = phase
        case .done(_, _, let phase):
            streamPhase = phase
        }

        appendEvent(Self.format(event: event))
    }

    private func resetStreamState() {
        streamPhase = .unknown
        streamBytesSoFar = 0
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

    private static func format(event: LLMPhaseAwareStreamEvent) -> String {
        switch event {
        case .requestSent(let phase):
            return "event: request-sent phase=\(phase.rawValue)"
        case .firstByteReceived(let seconds, let phase):
            return String(format: "event: first-byte %.3fs phase=%@", seconds, phase.rawValue)
        case .phaseChanged(let previousPhase, let phase):
            return "event: phase \(previousPhase.rawValue) -> \(phase.rawValue)"
        case .tokenChunk(let preview, let bytesSoFar, let phase):
            return "event: token-chunk phase=\(phase.rawValue) bytes=\(bytesSoFar) preview=\(preview)"
        case .finalAnswerDelta(let text, let bytesSoFar):
            return "event: final-answer-delta bytes=\(bytesSoFar) text=\(text)"
        case .finalAnswerSnapshot(let text, let bytesSoFar, let reason):
            return "event: final-answer-snapshot reason=\(reason.rawValue) bytes=\(bytesSoFar) text=\(text)"
        case .generationStats(let promptTokens, let generatedTokens, let stopReason, let templateMode, let phase):
            return "event: stats phase=\(phase.rawValue) promptTokens=\(promptTokens) generatedTokens=\(generatedTokens) stopReason=\(stopReason) templateMode=\(templateMode.rawValue)"
        case .done(let totalBytes, let duration, let phase):
            return String(format: "event: done phase=%@ bytes=%d duration=%.3fs", phase.rawValue, totalBytes, duration)
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
            .keyboardType(.asciiCapable)
            .textContentType(nil)
            .submitLabel(.done)
#else
        self
#endif
    }

    @ViewBuilder
    func demoNumericInput() -> some View {
#if os(iOS)
        keyboardType(.numberPad)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
#else
        self
#endif
    }
}
