import CarbocationLocalLLM
import CarbocationLocalLLMRuntime
import CarbocationLocalLLMRuntimeUI
import Foundation
import Observation
import SwiftUI
import UniformTypeIdentifiers

#if os(macOS)
import AppKit

@main
private enum CLLMSmokeApp {
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

    func applicationDidFinishLaunching(_ notification: Notification) {
        let hostingView = NSHostingView(rootView: SmokeRootView())

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1_080, height: 680),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = CLLMSmokeMetadata.displayName
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
#else
@main
private struct CLLMSmokeApp: App {
    var body: some Scene {
        WindowGroup {
            SmokeRootView()
        }
    }
}
#endif

private enum CLLMSmokeMetadata {
#if os(macOS)
    static let displayName = "CLLMSmokeMac"
#else
    static let displayName = "CLLMSmokeIOS"
#endif
    static let appSupportFolderName = "CarbocationLocalLLM"
    static let selectedModelDefaultsKey = "CLLMSmoke.selectedModelID"
}

@MainActor
private struct SmokeRootView: View {
    private let library: ModelLibrary
    @State private var selectedModelID: String
    @State private var showImageImporter = false
    @State private var smoke = SmokeState()

    init() {
        library = Self.makeLibrary()
        _selectedModelID = State(
            initialValue: UserDefaults.standard.string(forKey: CLLMSmokeMetadata.selectedModelDefaultsKey) ?? ""
        )
    }

    var body: some View {
        rootContent
            .onChange(of: selectedModelID) { _, newValue in
                UserDefaults.standard.set(newValue, forKey: CLLMSmokeMetadata.selectedModelDefaultsKey)
            }
            .fileImporter(
                isPresented: $showImageImporter,
                allowedContentTypes: [.png, .jpeg, .heic, .heif],
                allowsMultipleSelection: false
            ) { result in
                smoke.handleImageImport(result)
            }
    }

    @ViewBuilder
    private var rootContent: some View {
#if os(macOS)
        HStack(spacing: 0) {
            modelPicker
                .frame(width: 620)
                .frame(minHeight: 680)

            Divider()

            outputPane
                .padding(20)
                .frame(width: 460)
                .frame(minHeight: 680)
        }
        .frame(minWidth: 1_080, minHeight: 680)
#else
        TabView {
            NavigationStack {
                modelPicker
                    .navigationTitle("Models")
                    .navigationBarTitleDisplayMode(.inline)
            }
            .tabItem {
                Label("Models", systemImage: "cpu")
            }

            NavigationStack {
                ScrollView {
                    outputPane
                        .padding()
                }
                .navigationTitle("Smoke Output")
                .navigationBarTitleDisplayMode(.inline)
            }
            .tabItem {
                Label("Output", systemImage: "text.bubble")
            }
        }
#endif
    }

    private var modelPicker: some View {
        LocalLLMModelConfigurationView(
            library: library,
            selectedModelID: $selectedModelID,
            title: CLLMSmokeMetadata.displayName,
            confirmTitle: smoke.isRunning ? "Running" : "Run Smoke Test",
            confirmDisabled: smoke.isRunning,
            onConfirmSelection: { selection in
                smoke.run(selection: selection, library: library)
            }
        )
    }

    private var outputPane: some View {
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

            imageControls

            ScrollView {
                Text(smoke.output.isEmpty ? "Select a model and run the smoke test." : smoke.output)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, minHeight: 240, alignment: .topLeading)
                    .textSelection(.enabled)
            }
            .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 6))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var imageControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Vision Input")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Button {
                    showImageImporter = true
                } label: {
                    Label(smoke.selectedImage == nil ? "Attach Image" : "Replace Image", systemImage: "photo.badge.plus")
                }
                .disabled(smoke.isRunning)

                Button {
                    smoke.clearImage()
                } label: {
                    Label("Remove", systemImage: "xmark.circle")
                }
                .disabled(smoke.selectedImage == nil || smoke.isRunning)
            }

            if let selectedImage = smoke.selectedImage {
                HStack(spacing: 10) {
#if os(macOS)
                    if let image = NSImage(data: selectedImage.data) {
                        Image(nsImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 64, height: 64)
                            .background(.quaternary.opacity(0.35), in: .rect(cornerRadius: 4))
                    }
#endif
                    VStack(alignment: .leading, spacing: 2) {
                        Text(selectedImage.filename)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Text(selectedImage.detailText)
                            .foregroundStyle(.secondary)
                    }
                    .font(.caption)
                }
            } else {
                Text("No image attached")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.25), in: .rect(cornerRadius: 6))
    }

    private static func makeLibrary() -> ModelLibrary {
        ModelLibrary(
            root: ModelStorage.modelsDirectory(appSupportFolderName: CLLMSmokeMetadata.appSupportFolderName),
            contextLengthProbe: { url in
                LocalLLMEngine.probeTrainingContext(at: url)
            }
        )
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
    var selectedImage: SmokeImageSelection?

    func clear() {
        output = ""
    }

    func clearImage() {
        selectedImage = nil
    }

    func handleImageImport(_ result: Result<[URL], Error>) {
        do {
            guard let url = try result.get().first else { return }
            selectedImage = try SmokeImageSelection(url: url)
            append("image: selected \(selectedImage?.filename ?? url.lastPathComponent)")
        } catch {
            append("image: failed: \(error.localizedDescription)")
        }
    }

    func run(selection: LLMModelSelection, library: ModelLibrary) {
        guard !isRunning else { return }
        isRunning = true
        output = ""
        let image = selectedImage

        Task { @MainActor in
            await runSmoke(selection: selection, library: library, image: image)
            isRunning = false
        }
    }

    private func runSmoke(
        selection: LLMModelSelection,
        library: ModelLibrary,
        image: SmokeImageSelection?
    ) async {
        do {
            let engine = LocalLLMEngine(configuration: LocalLLMEngineConfiguration(
                heartbeatInterval: 0.5
            ))
            let plan = await LocalLLMEngine.loadPlan(
                from: selection.storageValue,
                in: library,
                refreshingLibrary: false
            )
            let requestedContext = plan?.requestedContext ?? 4_096
            let loaded = try await engine.load(
                selection: selection,
                from: library,
                requestedContext: requestedContext
            )

            append("model: \(loaded.displayName)")
            append("provider: \(providerLabel(for: selection))")
            if case .installed(let id) = selection,
               let model = library.model(id: id) {
                append("path: \(model.weightsURL(in: library.root).path)")
            }
            append("requestedContext: \(requestedContext)")
            append("loadedContext: \(loaded.contextSize)")
            append("trainingContext: \(loaded.trainingContextSize)")
            append("supportsGrammar: \(loaded.supportsGrammar)")
            append("supportsMTPAcceleration: \(loaded.supportsMTPAcceleration)")
            append("supportsVision: \(loaded.supportsVision)")
            if let image {
                append("image: \(image.filename) \(image.detailText)")
            }

            let options = generationOptions(for: loaded)

            let response: String
            if let image {
                response = try await engine.generate(
                    messages: messages(for: loaded, image: image),
                    options: options,
                    onPhaseAwareEvent: { [weak self] event in
                        Task { @MainActor [weak self] in
                            self?.append(Self.format(event: event))
                        }
                    }
                )
            } else {
                response = try await engine.generate(
                    system: systemPrompt(for: loaded),
                    prompt: #"Return {"ok": true, "message": "hello"}."#,
                    options: options,
                    onPhaseAwareEvent: { [weak self] event in
                        Task { @MainActor [weak self] in
                            self?.append(Self.format(event: event))
                        }
                    },
                    ()
                )
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

    private static func format(event: LLMPhaseAwareStreamEvent) -> String {
        switch event {
        case .requestSent(let phase):
            return "event: request-sent phase=\(phase.rawValue)"
        case .firstByteReceived(let seconds, let phase):
            return String(format: "event: first-byte phase=%@ %.3fs", phase.rawValue, seconds)
        case .phaseChanged(let from, let to):
            return "event: phase-changed from=\(from.rawValue) to=\(to.rawValue)"
        case .tokenChunk(let preview, let bytesSoFar, let phase):
            return "event: token-chunk phase=\(phase.rawValue) bytes=\(bytesSoFar) preview=\(preview)"
        case .finalAnswerDelta(let text, let bytesSoFar):
            return "event: final-answer-delta bytes=\(bytesSoFar) text=\(text)"
        case .finalAnswerSnapshot(let text, let bytesSoFar, let reason):
            return "event: final-answer-snapshot reason=\(reason.rawValue) bytes=\(bytesSoFar) text=\(text)"
        case .generationStats(let promptTokens, let generatedTokens, let stopReason, let templateMode, let phase):
            return "event: stats phase=\(phase.rawValue) promptTokens=\(promptTokens) generatedTokens=\(generatedTokens) stopReason=\(stopReason) templateMode=\(templateMode.rawValue)"
        case .accelerationStats(let stats):
            let rate = stats.acceptanceRate
                .map { String(format: "%.1f%%", $0 * 100) }
                ?? "n/a"
            return "event: acceleration accelerator=\(stats.accelerator) status=\(stats.status.rawValue) maxDraftTokens=\(stats.maxDraftTokens) draftCalls=\(stats.draftCalls) draftGenerated=\(stats.draftTokensGenerated) draftAccepted=\(stats.draftTokensAccepted) acceptance=\(rate)"
        case .diagnostic(let message):
            return "event: \(message)"
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
        let requestOptions = GenerationOptions(
            maxOutputTokens: 96,
            stopAtBalancedJSON: true,
            grammar: loaded.supportsGrammar ? Self.jsonObjectGrammar : nil
        )
        return (loaded.supportsGrammar ? LLMSamplingDefaults.extractionSafe : .providerDefault)
            .applying(to: requestOptions)
    }

    private func systemPrompt(for loaded: LocalLLMLoadedModelInfo) -> String {
        if loaded.supportsGrammar {
            return "Return only JSON matching the requested schema."
        }
        return "Return only JSON matching the requested schema. Do not include prose."
    }

    private func messages(
        for loaded: LocalLLMLoadedModelInfo,
        image: SmokeImageSelection
    ) -> [LLMChatMessage] {
        [
            LLMChatMessage(role: .system, text: systemPrompt(for: loaded)),
            LLMChatMessage(role: .user, content: [
                .text(#"Inspect the attached image. Return {"ok": true, "message": "<short image description>"}."#),
                .image(image.input)
            ])
        ]
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

private struct SmokeImageSelection: Hashable, Sendable {
    var filename: String
    var data: Data
    var mimeType: String?

    init(url: URL) throws {
        let didAccess = url.startAccessingSecurityScopedResource()
        defer {
            if didAccess {
                url.stopAccessingSecurityScopedResource()
            }
        }

        let data = try Data(contentsOf: url)
        self.filename = url.lastPathComponent
        self.data = data
        self.mimeType = Self.mimeType(for: url, data: data)
    }

    var input: LLMImageInput {
        .encoded(data: data, mimeType: mimeType)
    }

    var detailText: String {
        let byteCount = ByteCountFormatter.string(fromByteCount: Int64(data.count), countStyle: .file)
        if let mimeType {
            return "\(byteCount), \(mimeType)"
        }
        return byteCount
    }

    private static func mimeType(for url: URL, data: Data) -> String? {
        if let format = LLMImageInput.sniffEncodedFormat(data) {
            return format.mimeType
        }
        return UTType(filenameExtension: url.pathExtension)?.preferredMIMEType
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
