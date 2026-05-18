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

            HStack {
                Toggle(
                    isOn: Binding(
                        get: { state.mtpAccelerationEnabled },
                        set: { state.setMTPAccelerationEnabled($0) }
                    )
                ) {
                    Label(
                        "MTP",
                        systemImage: state.mtpAccelerationEnabled ? "bolt.fill" : "bolt.slash"
                    )
                }
                .disabled(state.isRunning)
                .help("Enable or disable MTP acceleration for the next model load")

                Spacer()

                Text(state.accelerationPolicyStatusLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text("Max output")
                    .frame(width: 84, alignment: .leading)

                TextField("Context", text: $state.maxOutputTokensText)
                    .textFieldStyle(.roundedBorder)
                    .demoNumericInput()
                    .disabled(state.isRunning)
                    .frame(maxWidth: 180)

                Text("tokens")
                    .foregroundStyle(.secondary)
            }

            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text("MTP draft")
                    .frame(width: 84, alignment: .leading)

                TextField(
                    "3",
                    text: Binding(
                        get: { state.mtpMaxDraftTokensText },
                        set: { state.setMTPMaxDraftTokensText($0) }
                    )
                )
                .textFieldStyle(.roundedBorder)
                .demoNumericInput()
                .disabled(state.isRunning || !state.mtpAccelerationEnabled)
                .frame(maxWidth: 180)

                Text("tokens")
                    .foregroundStyle(.secondary)
            }

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

            Button {
                state.stopThinking()
            } label: {
                Label("Stop Thinking", systemImage: "brain.head.profile")
            }
            .buttonStyle(.bordered)
            .disabled(!state.canStopThinking)

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
                VStack(alignment: .leading, spacing: 6) {
                    Label(
                        state.streamActivityTitle,
                        systemImage: state.streamActivitySystemImage
                    )
                    .font(.headline)
                    .foregroundStyle(state.streamActivityColor)

                    Label(
                        state.streamPhaseDetailTitle,
                        systemImage: state.streamPhaseDetailSystemImage
                    )
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(state.streamPhaseDetailColor)
                }

                Spacer()

                if let byteCount = state.streamByteCountLabel {
                    Text(byteCount)
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
