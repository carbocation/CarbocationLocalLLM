import CarbocationLocalLLM
import CarbocationLocalLLMRuntime
import CarbocationLocalLLMUI
import Observation
import SwiftUI

#if os(macOS)
import AppKit

@main
struct CLLMSmokeApp {
    @MainActor private static var delegate: SmokeAppDelegate?

    @MainActor
    static func main() {
        let delegate = SmokeAppDelegate()
        Self.delegate = delegate
        let application = NSApplication.shared
        application.setActivationPolicy(.regular)
        application.delegate = delegate
        application.run()
    }
}

@MainActor
private final class SmokeAppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow?
    private var library: ModelLibrary?
    private let smoke = SmokeState()

    func applicationDidFinishLaunching(_ notification: Notification) {
        let root = ModelStorage.modelsDirectory(appSupportFolderName: "CarbocationLocalLLM")
        let contextProbe: ModelContextLengthProbe = { url in
            LocalLLMEngine.probeTrainingContext(at: url)
        }
        let library = ModelLibrary(
            root: root,
            contextLengthProbe: contextProbe
        )
        self.library = library

        let hostingView = NSHostingView(
            rootView:
            SmokeRootView(
                library: library,
                initialSelectedModelID: UserDefaults.standard.string(forKey: "CLLMSmoke.selectedModelID") ?? "",
                smoke: smoke
            )
        )

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1_080, height: 680),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "CLLMSmoke"
        window.contentView = hostingView
        window.center()
        window.makeKeyAndOrderFront(nil)
        self.window = window

        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

private struct SmokeRootView: View {
    let library: ModelLibrary
    let smoke: SmokeState
    @State private var selectedModelID: String

    init(
        library: ModelLibrary,
        initialSelectedModelID: String,
        smoke: SmokeState
    ) {
        self.library = library
        self.smoke = smoke
        _selectedModelID = State(initialValue: initialSelectedModelID)
    }

    var body: some View {
        HStack(spacing: 0) {
            ModelLibraryPickerView(
                library: library,
                selectedModelID: $selectedModelID,
                title: "CLLMSmoke",
                confirmTitle: smoke.isRunning ? "Running" : "Run Smoke Test",
                confirmDisabled: smoke.isRunning,
                systemModels: Self.systemModels,
                onConfirmSelection: { selection in
                    smoke.run(selection: selection, library: library)
                }
            )
            .frame(width: 620)
            .frame(minHeight: 680)

            Divider()

            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Smoke Output")
                        .font(.headline)
                    Spacer()
                    Button {
                        smoke.clear()
                    } label: {
                        Label("Clear", systemImage: "xmark.circle")
                    }
                    .disabled(smoke.output.isEmpty || smoke.isRunning)
                }

                ScrollView {
                    Text(smoke.output.isEmpty ? "Select a model and run the smoke test." : smoke.output)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 6))
            }
            .padding(20)
            .frame(width: 460)
            .frame(minHeight: 680)
        }
        .frame(minWidth: 1_080, minHeight: 680)
        .onChange(of: selectedModelID) { _, newValue in
            UserDefaults.standard.set(newValue, forKey: "CLLMSmoke.selectedModelID")
        }
    }

    private static var systemModels: [LLMSystemModelOption] {
        LocalLLMEngine.availableSystemModels()
    }
}

@MainActor
@Observable
private final class SmokeState {
    private static let jsonObjectGrammar = #"""
    root ::= object
    object ::= "{" ws "\"ok\"" ws ":" ws boolean ws "," ws "\"message\"" ws ":" ws string ws "}" ws
    boolean ::= "true" | "false"
    string ::= "\"" chars "\""
    chars ::= ([^"\\] | "\\" (["\\/bfnrt] | "u" hex hex hex hex))*
    hex ::= [0-9a-fA-F]
    ws ::= [ \t\n\r]*
    """#

    var output = ""
    var isRunning = false

    func clear() {
        output = ""
    }

    func run(selection: LLMModelSelection, library: ModelLibrary) {
        guard !isRunning else { return }
        isRunning = true
        output = ""

        Task { @MainActor in
            await runSmoke(selection: selection, library: library)
            isRunning = false
        }
    }

    private func runSmoke(selection: LLMModelSelection, library: ModelLibrary) async {
        do {
            let engine = LocalLLMEngine(configuration: LocalLLMEngineConfiguration(
                heartbeatInterval: 0.5
            ))
            let loaded = try await engine.load(
                selection: selection,
                from: library,
                requestedContext: 4_096
            )

            append("model: \(loaded.displayName)")
            append("provider: \(providerLabel(for: selection))")
            if case .installed(let id) = selection,
               let model = library.model(id: id) {
                append("path: \(model.weightsURL(in: library.root).path)")
            }
            append("requestedContext: 4096")
            append("loadedContext: \(loaded.contextSize)")
            append("trainingContext: \(loaded.trainingContextSize)")
            append("supportsGrammar: \(loaded.supportsGrammar)")

            let options = generationOptions(for: loaded)

            let response = try await engine.generate(
                system: systemPrompt(for: loaded),
                prompt: #"Return {"ok": true, "message": "hello"}."#,
                options: options
            ) { [weak self] event in
                Task { @MainActor [weak self] in
                    self?.append(Self.format(event: event))
                }
            }

            let normalized = try Self.validatedNormalizedJSON(from: response)
            append("rawResponse:")
            append(response)
            append("normalizedResponse:")
            append(normalized)

            await engine.unload()
            append("smoke: ok")
        } catch {
            append("smoke: failed: \(error.localizedDescription)")
        }
    }

    private func append(_ line: String) {
        if output.isEmpty {
            output = line
        } else {
            output += "\n\(line)"
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

    private static func validatedNormalizedJSON(from response: String) throws -> String {
        let normalized = JSONSalvage.unwrapResponse(response)
        if isValidSmokePayload(normalized) {
            return normalized
        }

        let payload = try JSONSalvage.decode(SmokeJSONPayload.self, from: response)
        guard payload.ok != nil, payload.message != nil else {
            throw SmokeError.invalidJSON(normalized)
        }
        return try LocalLLMJSON.prettyPrintedString(from: payload)
    }

    private static func isValidSmokePayload(_ text: String) -> Bool {
        guard let data = text.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        return object["ok"] != nil && object["message"] != nil
    }

    private func generationOptions(for loaded: LocalLLMLoadedModelInfo) -> GenerationOptions {
        GenerationOptions(
            temperature: loaded.supportsGrammar ? 0 : nil,
            maxOutputTokens: 96,
            stopAtBalancedJSON: true,
            grammar: loaded.supportsGrammar ? Self.jsonObjectGrammar : nil
        )
    }

    private func systemPrompt(for loaded: LocalLLMLoadedModelInfo) -> String {
        if loaded.supportsGrammar {
            return "Return only JSON matching the requested schema."
        }
        return "Return only JSON matching the requested schema. Do not include prose."
    }

    private func providerLabel(for selection: LLMModelSelection) -> String {
        switch selection {
        case .installed:
            return "GGUF"
        case .system(.appleIntelligence):
            return "Apple Intelligence"
        }
    }
}

private struct SmokeJSONPayload: Codable {
    var ok: Bool?
    var message: String?
}

private enum SmokeError: LocalizedError {
    case invalidJSON(String)

    var errorDescription: String? {
        switch self {
        case .invalidJSON(let response):
            return "Response was not valid JSON for the smoke schema: \(response)"
        }
    }
}
#else
@main
struct CLLMSmokeApp {
    static func main() {
        fatalError("CLLMSmoke is only available on macOS.")
    }
}
#endif
