import CarbocationLocalLLM
import CarbocationLocalLLMRuntime
import CarbocationLocalLLMRuntimeUI
import CarbocationLocalLLMTools
import Foundation
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

private enum DemoRunMode: String, CaseIterable, Identifiable {
    case plain
    case tools

    var id: Self { self }

    var title: String {
        switch self {
        case .plain:
            return "Plain"
        case .tools:
            return "Tools"
        }
    }
}

private enum DemoToolSamplePrompt: String, CaseIterable, Identifiable {
    case calculate
    case convertUnits
    case loadWebpage
    case unsupportedCurrency
    case blockedURL

    var id: Self { self }

    var title: String {
        switch self {
        case .calculate:
            return "Math"
        case .convertUnits:
            return "Units"
        case .loadWebpage:
            return "Webpage"
        case .unsupportedCurrency:
            return "Currency Error"
        case .blockedURL:
            return "Bad URL"
        }
    }

    var systemImage: String {
        switch self {
        case .calculate:
            return "function"
        case .convertUnits:
            return "ruler"
        case .loadWebpage:
            return "globe"
        case .unsupportedCurrency:
            return "dollarsign.arrow.circlepath"
        case .blockedURL:
            return "lock.slash"
        }
    }

    var prompt: String {
        switch self {
        case .calculate:
            return "Use calculate to compute 17.5 * 23 + 8, then give the result in one sentence."
        case .convertUnits:
            return "Use convert_units to convert 12 miles to kilometers and 72 Fahrenheit to Celsius."
        case .loadWebpage:
            return "Use load_webpage to load https://example.com, then summarize the page in one sentence."
        case .unsupportedCurrency:
            return "Try to use convert_units to convert 10 USD to EUR, then explain the tool result."
        case .blockedURL:
            return "Try to use load_webpage to load file:///etc/passwd, then explain the tool result."
        }
    }
}

private struct DemoFixtureWebpageFetcher: LLMWebpageFetching {
    func fetch(_ request: URLRequest) async throws -> LLMWebpageFetchResponse {
        guard let url = request.url else {
            throw URLError(.badURL)
        }

        let html = """
        <!doctype html>
        <html>
          <head>
            <title>Fixture Tool Page</title>
            <style>body { font-family: system-ui; }</style>
          </head>
          <body>
            <main>
              <h1>Fixture Tool Page</h1>
              <p>This deterministic page exists for manual tool-calling tests.</p>
              <p>It lets the demo exercise webpage loading without relying on live network access.</p>
            </main>
          </body>
        </html>
        """

        return LLMWebpageFetchResponse(
            data: Data(html.utf8),
            finalURL: url,
            statusCode: 200,
            mimeType: "text/html"
        )
    }
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
            state.resetSamplingControlsToDefaults()
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
                toolLabSection
                controls
                streamSection
                outputSection
                toolTranscriptSection
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
            HStack {
                Text("Generation")
                    .font(.headline)

                Spacer()

                Button {
                    state.resetSamplingControlsToDefaults()
                } label: {
                    Label("Reset", systemImage: "arrow.counterclockwise")
                }
                .buttonStyle(.bordered)
                .disabled(state.isRunning)
                .help("Reset sampling controls")
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

            Divider()

            Text("Sampling")
                .font(.subheadline.weight(.semibold))

            samplingRow("Temperature", text: $state.temperatureText)
            samplingRow("Top P", text: $state.topPText)
            samplingRow("Top K", text: $state.topKText, integerOnly: true)
            samplingRow("Min P", text: $state.minPText)
            samplingRow("Presence penalty", text: $state.presencePenaltyText)
            samplingRow("Repetition penalty", text: $state.repetitionPenaltyText)

            if let message = state.generationOptionsValidationMessage {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
        .padding(12)
        .background(.quaternary.opacity(0.25), in: .rect(cornerRadius: 8))
    }

    private func samplingRow(
        _ title: String,
        text: Binding<String>,
        integerOnly: Bool = false
    ) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(title)
                .frame(width: 128, alignment: .leading)

            TextField("Default", text: text)
                .textFieldStyle(.roundedBorder)
                .demoSamplingInput(integerOnly: integerOnly)
                .disabled(state.isRunning)
                .frame(maxWidth: 180)
        }
    }

    private var toolLabSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Tool Lab")
                .font(.headline)

            Picker("Run mode", selection: $state.runMode) {
                ForEach(DemoRunMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .disabled(state.isRunning)

            if state.runMode == .tools {
                VStack(alignment: .leading, spacing: 8) {
                    Toggle("load_webpage", isOn: $state.enableLoadWebpageTool)
                    Toggle("calculate", isOn: $state.enableCalculateTool)
                    Toggle("convert_units", isOn: $state.enableConvertUnitsTool)
                    Toggle("Fixture webpage", isOn: $state.useFixtureWebpageTool)
                        .disabled(!state.enableLoadWebpageTool)
                }
                .disabled(state.isRunning)

                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 132), spacing: 8)],
                    alignment: .leading,
                    spacing: 8
                ) {
                    ForEach(DemoToolSamplePrompt.allCases) { sample in
                        Button {
                            state.applyToolSample(sample)
                        } label: {
                            Label(sample.title, systemImage: sample.systemImage)
                        }
                        .buttonStyle(.bordered)
                        .disabled(state.isRunning)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }

                if let message = state.toolValidationMessage {
                    Text(message)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
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
            .disabled(state.isRunning || (state.output.isEmpty && state.events.isEmpty && state.toolTranscript.isEmpty))
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

    @ViewBuilder
    private var toolTranscriptSection: some View {
        if state.runMode == .tools || !state.toolTranscript.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text("Tool Transcript")
                    .font(.headline)
                Text(state.toolTranscript.isEmpty ? "No tool calls yet." : state.toolTranscript)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, minHeight: 120, alignment: .topLeading)
                    .padding(12)
                    .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 8))
            }
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
    var temperatureText = ""
    var topPText = ""
    var topKText = ""
    var minPText = ""
    var presencePenaltyText = ""
    var repetitionPenaltyText = ""
    var enableThinking = false
    var thinkingBudgetText = ""
    var runMode: DemoRunMode = .plain
    var enableLoadWebpageTool = true
    var enableCalculateTool = true
    var enableConvertUnitsTool = true
    var useFixtureWebpageTool = false
    var output = ""
    var events = ""
    var toolTranscript = ""
    var streamPhase: LLMStreamContentPhase = .unknown
    var streamBytesSoFar = 0
    var errorMessage: String?
    var loadedInfo: LocalLLMLoadedModelInfo?
    var isRunning = false

    private var generationTask: Task<Void, Never>?
    private let appSamplingOverrides: [CuratedModelReference: LLMSamplingDefaults] = [:]
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
        resetSamplingControlsToDefaults()
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
            && toolValidationMessage == nil
    }

    var generationOptionsValidationMessage: String? {
        thinkingBudgetValidationMessage ?? samplingOptionsValidationMessage
    }

    var toolValidationMessage: String? {
        guard runMode == .tools else { return nil }
        guard selectedToolCount > 0 else {
            return "Enable at least one tool before running in Tools mode."
        }
        return nil
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

    var samplingOptionsValidationMessage: String? {
        validateDouble(
            temperatureText,
            name: "Temperature",
            lowerBound: 0,
            upperBound: nil
        )
            ?? validateDouble(topPText, name: "Top P", lowerBound: 0, upperBound: 1)
            ?? validateInt(topKText, name: "Top K", lowerBound: 0)
            ?? validateDouble(minPText, name: "Min P", lowerBound: 0, upperBound: 1)
            ?? validateDouble(
                presencePenaltyText,
                name: "Presence penalty",
                lowerBound: nil,
                upperBound: nil
            )
            ?? validatePositiveDouble(repetitionPenaltyText, name: "Repetition penalty")
    }

    private var parsedTemperature: Double? {
        parsedDouble(temperatureText)
    }

    private var parsedTopP: Double? {
        parsedDouble(topPText)
    }

    private var parsedTopK: Int? {
        parsedInt(topKText)
    }

    private var parsedMinP: Double? {
        parsedDouble(minPText)
    }

    private var parsedPresencePenalty: Double? {
        parsedDouble(presencePenaltyText)
    }

    private var parsedRepetitionPenalty: Double? {
        parsedDouble(repetitionPenaltyText)
    }

    private var parsedThinkingBudgetTokens: Int? {
        guard enableThinking else { return nil }

        let trimmed = thinkingBudgetText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return Int(trimmed)
    }

    private var selectedToolCount: Int {
        [
            enableLoadWebpageTool,
            enableCalculateTool,
            enableConvertUnitsTool
        ].filter { $0 }.count
    }

    private func parsedDouble(_ text: String) -> Double? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return Double(trimmed)
    }

    private func parsedInt(_ text: String) -> Int? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return Int(trimmed)
    }

    private func validateDouble(
        _ text: String,
        name: String,
        lowerBound: Double?,
        upperBound: Double?
    ) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard let value = Double(trimmed), value.isFinite else {
            return "\(name) must be blank or a number."
        }
        if let lowerBound, value < lowerBound {
            return "\(name) must be at least \(Self.formatSamplingValue(lowerBound))."
        }
        if let upperBound, value > upperBound {
            return "\(name) must be at most \(Self.formatSamplingValue(upperBound))."
        }
        return nil
    }

    private func validateInt(_ text: String, name: String, lowerBound: Int) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard let value = Int(trimmed), value >= lowerBound else {
            return "\(name) must be blank or at least \(lowerBound)."
        }
        return nil
    }

    private func validatePositiveDouble(_ text: String, name: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard let value = Double(trimmed), value.isFinite else {
            return "\(name) must be blank or a number."
        }
        guard value > 0 else {
            return "\(name) must be greater than 0."
        }
        return nil
    }

    private func applySamplingDefaults(_ defaults: LLMSamplingDefaults) {
        temperatureText = Self.formatSamplingValue(defaults.temperature)
        topPText = Self.formatSamplingValue(defaults.topP)
        topKText = defaults.topK.map(String.init) ?? ""
        minPText = Self.formatSamplingValue(defaults.minP)
        presencePenaltyText = Self.formatSamplingValue(defaults.presencePenalty)
        repetitionPenaltyText = Self.formatSamplingValue(defaults.repetitionPenalty)
    }

    private func resolvedSamplingDefaultsForSelectedModel() -> LLMSamplingDefaults {
        let selection = LLMModelSelection(storageValue: selectedModelID)
        return resolvedSamplingDefaults(for: selection)
    }

    private func resolvedSamplingDefaults(
        for selection: LLMModelSelection?,
        supportsGrammar: Bool? = nil
    ) -> LLMSamplingDefaults {
        LLMSamplingDefaultsResolver.resolvedDefaults(
            globalDefaults: globalSamplingDefaults(for: selection, supportsGrammar: supportsGrammar),
            installedModel: installedModel(for: selection),
            appOverrides: appSamplingOverrides
        )
    }

    private func installedModel(for selection: LLMModelSelection?) -> InstalledModel? {
        guard case .installed(let id) = selection else { return nil }
        return library.model(id: id)
    }

    private func globalSamplingDefaults(
        for selection: LLMModelSelection?,
        supportsGrammar: Bool?
    ) -> LLMSamplingDefaults {
        if let supportsGrammar {
            return supportsGrammar ? .extractionSafe : .providerDefault
        }

        switch selection {
        case .system:
            return .providerDefault
        case .installed, nil:
            return .extractionSafe
        }
    }

    private static func formatSamplingValue(_ value: Double?) -> String {
        guard let value else { return "" }
        return String(format: "%.6g", value)
    }

    func select(_ selection: LLMModelSelection) {
        selectedModelID = selection.storageValue
        loadedInfo = nil
        errorMessage = nil
        resetSamplingControlsToDefaults()
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
        resetSamplingControlsToDefaults()
    }

    func resetSamplingControlsToDefaults() {
        applySamplingDefaults(resolvedSamplingDefaultsForSelectedModel())
    }

    func applyToolSample(_ sample: DemoToolSamplePrompt) {
        runMode = .tools
        prompt = sample.prompt

        switch sample {
        case .calculate:
            enableCalculateTool = true
        case .convertUnits, .unsupportedCurrency:
            enableConvertUnitsTool = true
        case .loadWebpage, .blockedURL:
            enableLoadWebpageTool = true
            useFixtureWebpageTool = false
        }
    }

    func run() {
        guard !isRunning, !selectedModelID.isEmpty else { return }
        guard canRun else {
            errorMessage = generationOptionsValidationMessage
                ?? toolValidationMessage
                ?? "Select a model and enter a prompt before running."
            return
        }

        isRunning = true
        output = ""
        events = ""
        toolTranscript = ""
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
        toolTranscript = ""
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

            switch runMode {
            case .plain:
                appendEvent("mode: plain")
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

            case .tools:
                let tools = selectedTools()
                appendEvent("mode: tools")
                appendEvent("tools: \(tools.map(\.definition.name).joined(separator: ", "))")
                appendEvent("max-tool-rounds: 4")
                appendToolTranscript("tools: \(tools.map(\.definition.name).joined(separator: ", "))")

                let request = LLMToolGenerationRequest(
                    system: systemPrompt,
                    prompt: prompt,
                    options: options,
                    tools: tools,
                    toolChoice: .auto,
                    maxToolRounds: 4
                )
                let result = try await engine.generateWithTools(
                    request,
                    onEvent: { [weak self] event in
                        Task { @MainActor [weak self] in
                            self?.handle(event: event)
                        }
                    }
                )

                output = result.finalText
                appendEvent(
                    "tool-result: stop=\(result.stopReason) rounds=\(result.roundsCompleted) calls=\(result.toolCalls.count)"
                )
                appendToolTranscript(
                    "final: stop=\(result.stopReason) rounds=\(result.roundsCompleted) calls=\(result.toolCalls.count)"
                )
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
        let requestOptions = GenerationOptions(
            temperature: parsedTemperature,
            topP: parsedTopP,
            topK: parsedTopK,
            minP: parsedMinP,
            presencePenalty: parsedPresencePenalty,
            repetitionPenalty: parsedRepetitionPenalty,
            stopAtBalancedJSON: false,
            enableThinking: enableThinking,
            thinkingBudgetTokens: parsedThinkingBudgetTokens,
            thinkingBudgetMessage: "Thinking budget reached."
        )

        return LLMSamplingDefaultsResolver.resolvedOptions(
            globalDefaults: loaded.supportsGrammar ? .extractionSafe : .providerDefault,
            installedModel: installedModel(for: loaded.selection),
            appOverrides: appSamplingOverrides,
            requestOptions: requestOptions
        )
    }

    private func selectedTools() -> [LLMTool] {
        var tools: [LLMTool] = []

        if enableLoadWebpageTool {
            let fetcher: any LLMWebpageFetching
            if useFixtureWebpageTool {
                fetcher = DemoFixtureWebpageFetcher()
            } else {
                fetcher = URLSessionWebpageFetcher()
            }

            tools.append(LLMStandardTools.loadWebpage(
                configuration: LLMLoadWebpageTool.Configuration(
                    timeout: 10,
                    maximumBytes: 500_000,
                    maximumOutputCharacters: 8_000
                ),
                fetcher: fetcher
            ))
        }

        if enableCalculateTool {
            tools.append(LLMStandardTools.calculate())
        }

        if enableConvertUnitsTool {
            tools.append(LLMStandardTools.convertUnits())
        }

        return tools
    }

    private func appendGenerationOptionsEvent(
        options: GenerationOptions,
        loaded: LocalLLMLoadedModelInfo
    ) {
        let temperature = options.temperature.map { String(describing: $0) } ?? "provider"
        let topP = options.topP.map { String(describing: $0) } ?? "provider"
        let topK = options.topK.map(String.init) ?? "provider"
        let minP = options.minP.map { String(describing: $0) } ?? "provider"
        let presencePenalty = options.presencePenalty.map { String(describing: $0) } ?? "provider"
        let repetitionPenalty = options.repetitionPenalty.map { String(describing: $0) } ?? "provider"
        appendEvent(
            "sampling: temp=\(temperature) top-p=\(topP) top-k=\(topK) "
                + "min-p=\(minP) presence=\(presencePenalty) repetition=\(repetitionPenalty)"
        )

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

    private func handle(event: LLMToolStreamEvent) {
        switch event {
        case .modelEvent(let event):
            handle(event: event)
        case .toolRoundStarted(let round):
            appendEvent("tool-round: \(round)")
            appendToolTranscript("round \(round)")
        case .toolCallStarted(let call):
            appendEvent("tool-call: \(call.name) id=\(call.id)")
            appendToolTranscript(
                "call \(call.id) \(call.name) arguments=\(Self.compactJSON(call.arguments))"
            )
        case .toolCallCompleted(let output):
            appendEvent("tool-output: \(output.name) id=\(output.callID) error=false")
            appendToolTranscript(
                "result \(output.callID) \(output.name) content=\(Self.compactJSON(output.content))"
            )
        case .toolCallFailed(let output):
            appendEvent("tool-output: \(output.name) id=\(output.callID) error=true")
            appendToolTranscript(
                "error \(output.callID) \(output.name) content=\(Self.compactJSON(output.content))"
            )
        }
    }

    private func handle(event: LLMStreamEvent) {
        switch event {
        case .requestSent:
            streamPhase = .unknown
        case .firstByteReceived:
            streamPhase = .final
        case .tokenChunk(_, let bytesSoFar):
            streamPhase = .final
            streamBytesSoFar = bytesSoFar
        case .generationStats:
            streamPhase = .final
        case .done:
            streamPhase = .final
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

    private func appendToolTranscript(_ line: String) {
        if toolTranscript.isEmpty {
            toolTranscript = line
        } else {
            toolTranscript += "\n\(line)"
        }
    }

    private static func compactJSON(_ value: LLMJSONValue) -> String {
        (try? value.jsonString(prettyPrinted: false)) ?? "\(value)"
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

    @ViewBuilder
    func demoSamplingInput(integerOnly: Bool) -> some View {
#if os(iOS)
        if integerOnly {
            keyboardType(.numberPad)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
        } else {
            keyboardType(.decimalPad)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
        }
#else
        self
#endif
    }
}
