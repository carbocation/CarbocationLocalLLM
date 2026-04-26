import AppKit
import CarbocationAppleIntelligenceRuntime
import CarbocationLlamaRuntime
import CarbocationLocalLLM
import CarbocationLocalLLMUI
import Observation
import SwiftUI

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
            LlamaRuntimeModelProbe.probeTrainingContext(at: url)
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
                onConfirmSystemModel: { model in
                    smoke.run(systemModel: model)
                }
            ) { model in
                smoke.run(model: model, root: library.root)
            }
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
        AppleIntelligenceEngine.systemModelOption().map { [$0] } ?? []
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

    func run(model: InstalledModel, root: URL) {
        guard !isRunning else { return }
        isRunning = true
        output = ""

        Task { @MainActor in
            await runSmoke(model: model, root: root)
            isRunning = false
        }
    }

    func run(systemModel: LLMSystemModelOption) {
        guard !isRunning else { return }
        isRunning = true
        output = ""

        Task { @MainActor in
            switch systemModel.id {
            case AppleIntelligenceEngine.systemModelID:
                await runAppleIntelligenceSmoke(systemModel: systemModel)
            default:
                append("smoke: failed: unsupported system model \(systemModel.displayName)")
            }
            isRunning = false
        }
    }

    private func runSmoke(model: InstalledModel, root: URL) async {
        do {
            append("model: \(model.displayName)")
            append("path: \(model.weightsURL(in: root).path)")
            append("requestedContext: 4096")

            let engine = LlamaEngine(configuration: LlamaEngineConfiguration(heartbeatInterval: 0.5))
            let loaded = try await engine.load(model: model, from: root, requestedContext: 4_096)
            append("loadedContext: \(loaded.contextSize)")
            append("trainingContext: \(loaded.trainingContextSize)")
            append("embeddedTemplate: \(loaded.hasEmbeddedChatTemplate)")

            let options = GenerationOptions(
                temperature: 0,
                maxOutputTokens: 96,
                stopAtBalancedJSON: true,
                grammar: Self.jsonObjectGrammar
            )

            let response = try await engine.generate(
                system: "Return only JSON matching the requested schema.",
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

    private func runAppleIntelligenceSmoke(systemModel: LLMSystemModelOption) async {
        do {
            append("model: \(systemModel.displayName)")
            append("provider: Apple Intelligence")
            append("context: \(systemModel.contextLength)")

            let availability = AppleIntelligenceEngine.availability()
            guard availability.isAvailable else {
                throw AppleIntelligenceEngineError.unavailable(availability)
            }

            let engine = AppleIntelligenceEngine(configuration: AppleIntelligenceEngineConfiguration())
            let options = GenerationOptions(
                maxOutputTokens: 96,
                stopAtBalancedJSON: true
            )
            let response = try await engine.generate(
                system: "Return only JSON matching the requested schema. Do not include prose.",
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
