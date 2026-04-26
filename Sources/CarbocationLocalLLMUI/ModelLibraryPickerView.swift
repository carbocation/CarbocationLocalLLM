import AppKit
import CarbocationLocalLLM
import Observation
import SwiftUI
import UniformTypeIdentifiers

@MainActor
public struct ModelLibraryPickerView: View {
    private let library: ModelLibrary
    @Binding private var selectedModelID: String
    private let title: String
    private let confirmTitle: String
    private let confirmDisabled: Bool
    private let systemModels: [LLMSystemModelOption]
    private let curatedModels: [CuratedModel]
    private let labelPolicy: ModelLibraryPickerLabelPolicy
    private let onConfirmSelection: (@MainActor (LLMModelSelection) -> Void)?
    private let onModelDeleted: @MainActor (InstalledModel) -> Void
    private let onConfirmInstalledModel: (@MainActor (InstalledModel) -> Void)?

    @State private var activeDownload: ModelLibraryDownload?
    @State private var downloadError: String?
    @State private var showCustomSheet = false
    @State private var showDeleteConfirm: InstalledModel?
    @State private var showDeletePartialConfirm: PartialDownload?
    @State private var refreshToken = UUID()

    public init(
        library: ModelLibrary,
        selectedModelID: Binding<String>,
        title: String = "Choose a Local Model",
        confirmTitle: String = "Use Selected Model",
        confirmDisabled: Bool = false,
        curatedModels: [CuratedModel] = CuratedModelCatalog.all,
        labelPolicy: ModelLibraryPickerLabelPolicy = .default,
        onModelDeleted: @escaping @MainActor (InstalledModel) -> Void = { _ in },
        onConfirm: @escaping @MainActor (InstalledModel) -> Void
    ) {
        self.library = library
        self._selectedModelID = selectedModelID
        self.title = title
        self.confirmTitle = confirmTitle
        self.confirmDisabled = confirmDisabled
        self.systemModels = []
        self.curatedModels = curatedModels
        self.labelPolicy = labelPolicy
        self.onConfirmSelection = nil
        self.onModelDeleted = onModelDeleted
        self.onConfirmInstalledModel = onConfirm
    }

    public init(
        library: ModelLibrary,
        selectedModelID: Binding<String>,
        title: String = "Choose a Local Model",
        confirmTitle: String = "Use Selected Model",
        confirmDisabled: Bool = false,
        systemModels: [LLMSystemModelOption],
        curatedModels: [CuratedModel] = CuratedModelCatalog.all,
        labelPolicy: ModelLibraryPickerLabelPolicy = .default,
        onModelDeleted: @escaping @MainActor (InstalledModel) -> Void = { _ in },
        onConfirmSelection: @escaping @MainActor (LLMModelSelection) -> Void
    ) {
        self.library = library
        self._selectedModelID = selectedModelID
        self.title = title
        self.confirmTitle = confirmTitle
        self.confirmDisabled = confirmDisabled
        self.systemModels = systemModels
        self.curatedModels = curatedModels
        self.labelPolicy = labelPolicy
        self.onConfirmSelection = onConfirmSelection
        self.onModelDeleted = onModelDeleted
        self.onConfirmInstalledModel = nil
    }

    private var selectedSystemModel: LLMSystemModelOption? {
        systemModels.first { $0.id == selectedModelID }
    }

    private var selectedModel: InstalledModel? {
        library.model(id: selectedModelID)
    }

    private var hasSelection: Bool {
        selectedSystemModel != nil || selectedModel != nil
    }

    private var canConfirmSelection: Bool {
        if selectedSystemModel != nil {
            return onConfirmSelection != nil
        }
        if selectedModel != nil {
            return onConfirmSelection != nil || onConfirmInstalledModel != nil
        }
        return false
    }

    private var recommendedCuratedModel: CuratedModel? {
        CuratedModelCatalog.recommendedModel(
            forPhysicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            among: curatedModels
        )
    }

    private var bestInstalledCuratedModel: CuratedModel? {
        ModelLibraryPickerLabelPolicy.bestInstalledCuratedModel(
            forPhysicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            installedModels: library.models,
            curatedModels: curatedModels
        )
    }

    public var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if !systemModels.isEmpty {
                        systemModelSection
                        Divider()
                    }
                    installedSection
                    if !library.partials.isEmpty {
                        Divider()
                        interruptedSection
                    }
                    Divider()
                    downloadSection
                    if let downloadError {
                        Label(downloadError, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.orange)
                            .font(.callout)
                    }
                }
                .padding(20)
                .id(refreshToken)
            }
            Divider()
            footer
        }
        .task {
            library.refresh()
            normalizeSelection()
            refreshToken = UUID()
        }
        .sheet(isPresented: $showCustomSheet) {
            CustomHFSheet { repo, filename, displayName in
                startDownload(
                    hfRepo: repo,
                    hfFilename: filename,
                    displayName: displayName,
                    expectedSHA256: nil,
                    contextLength: 0,
                    source: .customHF
                )
            }
        }
        .alert(
            "Delete \(showDeleteConfirm?.displayName ?? "model")?",
            isPresented: Binding(
                get: { showDeleteConfirm != nil },
                set: { if !$0 { showDeleteConfirm = nil } }
            ),
            presenting: showDeleteConfirm
        ) { model in
            Button("Delete", role: .destructive) { delete(model) }
            Button("Cancel", role: .cancel) {}
        } message: { model in
            Text("This will remove the \(formatBytes(model.sizeBytes)) file from disk.")
        }
        .alert(
            "Delete interrupted download?",
            isPresented: Binding(
                get: { showDeletePartialConfirm != nil },
                set: { if !$0 { showDeletePartialConfirm = nil } }
            ),
            presenting: showDeletePartialConfirm
        ) { partial in
            Button("Delete", role: .destructive) {
                library.deletePartial(partial)
                refresh()
            }
            Button("Cancel", role: .cancel) {}
        } message: { partial in
            Text("This will delete the partial download for \(partial.displayName) and reclaim \(formatBytes(partial.bytesOnDisk)).")
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: "cpu")
                .font(.system(size: 28, weight: .regular))
                .foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.title3.bold())
                Text(headerSubtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 16)
    }

    private var headerSubtitle: String {
        if systemModels.isEmpty {
            return "Installed models, curated downloads, Hugging Face URLs, and local .gguf imports."
        }
        return "System models, installed models, curated downloads, Hugging Face URLs, and local .gguf imports."
    }

    @ViewBuilder
    private var systemModelSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("System Models")
                .font(.headline)

            ForEach(systemModels) { model in
                systemModelRow(model)
            }
        }
    }

    private func systemModelRow(_ model: LLMSystemModelOption) -> some View {
        let isSelected = model.id == selectedModelID
        let statusLabel = labelPolicy.systemModelLabel(for: model)
        return Button {
            selectedModelID = model.id
        } label: {
            HStack(spacing: 10) {
                Image(systemName: isSelected ? "largecircle.fill.circle" : "circle")
                    .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)

                Image(systemName: model.systemImageName)
                    .foregroundStyle(.tint)

                VStack(alignment: .leading, spacing: 3) {
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(model.displayName)
                            .font(.body)
                            .foregroundStyle(.primary)
                        if let statusLabel {
                            statusBadge(statusLabel)
                        }
                    }
                    HStack(spacing: 6) {
                        Text(model.subtitle)
                        if model.contextLength > 0 {
                            Text("context \(model.contextLength.formatted())")
                        }
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }

                Spacer()
            }
            .padding(.vertical, 5)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var installedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Installed Models")
                    .font(.headline)
                Spacer()
                Text("Total: \(formatBytes(library.totalDiskUsageBytes()))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if library.models.isEmpty {
                Text("No GGUF models installed. Download one below, or import an existing .gguf file.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 6)
            } else {
                ForEach(library.models) { model in
                    installedRow(model)
                }
            }

            HStack {
                Button {
                    importLocalGGUF()
                } label: {
                    Label("Import .gguf", systemImage: "square.and.arrow.down")
                }

                Button {
                    revealModelsFolder()
                } label: {
                    Label("Reveal Folder", systemImage: "folder")
                }

                Spacer()

                Button {
                    refresh()
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
            }
        }
    }

    private func installedRow(_ model: InstalledModel) -> some View {
        let isSelected = model.id.uuidString == selectedModelID
        let statusLabel = labelPolicy.installedModelLabel(
            for: model,
            recommendedCuratedModel: recommendedCuratedModel,
            bestInstalledCuratedModel: bestInstalledCuratedModel
        )
        return Button {
            selectedModelID = model.id.uuidString
        } label: {
            HStack(spacing: 10) {
                Image(systemName: isSelected ? "largecircle.fill.circle" : "circle")
                    .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)

                VStack(alignment: .leading, spacing: 3) {
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(model.displayName)
                            .font(.body)
                            .foregroundStyle(.primary)
                        if let statusLabel {
                            statusBadge(statusLabel)
                        }
                    }
                    HStack(spacing: 6) {
                        if let quantization = model.quantization {
                            Text(quantization)
                                .font(.caption.monospaced())
                                .padding(.horizontal, 4)
                                .padding(.vertical, 1)
                                .background(.quaternary, in: .rect(cornerRadius: 3))
                        }
                        Text(formatBytes(model.sizeBytes))
                        if model.contextLength > 0 {
                            Text("context \(model.contextLength.formatted())")
                        }
                        if let repo = model.hfRepo {
                            Text(repo)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }

                Spacer()

                Button {
                    showDeleteConfirm = model
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless)
                .help("Delete this model")
            }
            .padding(.vertical, 5)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var interruptedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Interrupted Downloads")
                .font(.headline)
            ForEach(library.partials) { partial in
                interruptedRow(partial)
            }
        }
    }

    private func interruptedRow(_ partial: PartialDownload) -> some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(partial.displayName)
                    .font(.body)
                Text("\(formatBytes(partial.bytesOnDisk)) of \(formatBytes(partial.totalBytes)) · \(Int(partial.fractionComplete * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Resume") {
                resume(partial)
            }
            .disabled(activeDownload != nil)
            Button {
                showDeletePartialConfirm = partial
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.borderless)
            .help("Delete partial download")
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var downloadSection: some View {
        let recommendedID = recommendedCuratedModel?.id
        VStack(alignment: .leading, spacing: 10) {
            Text("Download a Model")
                .font(.headline)
            Text(recommendationSummary())
                .font(.caption)
                .foregroundStyle(.secondary)

            if let activeDownload {
                activeDownloadRow(activeDownload)
            } else {
                ForEach(curatedModels) { entry in
                    curatedRow(entry, isRecommended: recommendedID == entry.id)
                }
                Button {
                    showCustomSheet = true
                } label: {
                    Label("Paste a Hugging Face URL", systemImage: "link")
                }
                .padding(.top, 4)
            }
        }
    }

    private func curatedRow(_ entry: CuratedModel, isRecommended: Bool) -> some View {
        let alreadyInstalled = library.models.contains {
            $0.hfRepo == entry.hfRepo && $0.hfFilename == entry.hfFilename
        }

        return HStack(alignment: .top, spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(entry.displayName)
                        .font(.body)
                    if isRecommended, let recommendedLabel = labelPolicy.recommendedLabel {
                        statusBadge(recommendedLabel)
                    }
                }
                Text(entry.subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("\(formatBytes(entry.approxSizeBytes)) · context \(entry.contextLength.formatted()) · ~\(entry.recommendedRAMGB) GB RAM")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if alreadyInstalled {
                Label("Installed", systemImage: "checkmark.circle")
                    .foregroundStyle(.green)
                    .font(.caption)
            } else {
                Button("Download") {
                    startDownload(
                        hfRepo: entry.hfRepo,
                        hfFilename: entry.hfFilename,
                        displayName: entry.displayName,
                        expectedSHA256: entry.sha256,
                        contextLength: entry.contextLength,
                        source: .curated
                    )
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func statusBadge(_ label: ModelLibraryPickerStatusLabel) -> some View {
        HStack(spacing: 3) {
            if let systemImageName = label.systemImageName {
                Image(systemName: systemImageName)
            }
            Text(label.title)
        }
        .font(.caption2)
        .fontWeight(.semibold)
        .padding(.horizontal, 6)
        .padding(.vertical, 3)
        .background(statusBadgeColor(for: label.tone).opacity(0.15), in: Capsule())
        .foregroundStyle(statusBadgeColor(for: label.tone))
    }

    private func statusBadgeColor(for tone: ModelLibraryPickerStatusLabel.Tone) -> Color {
        switch tone {
        case .accent:
            return .accentColor
        case .positive:
            return .green
        case .warning:
            return .orange
        case .secondary:
            return .secondary
        }
    }

    private func activeDownloadRow(_ download: ModelLibraryDownload) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Downloading \(download.displayName)")
                .font(.body)
            ProgressView(value: download.progress.fractionComplete)
            HStack {
                Text("\(formatBytes(download.progress.bytesReceived)) of \(formatBytes(download.progress.totalBytes))")
                if download.progress.bytesPerSecond > 0 {
                    Text("\(formatBytes(Int64(download.progress.bytesPerSecond)))/s")
                }
                Spacer()
                Button("Cancel", role: .destructive) {
                    download.cancel()
                    activeDownload = nil
                    refresh()
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            if let errorMessage = download.errorMessage {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
        .padding(.vertical, 4)
    }

    private var footer: some View {
        HStack {
            if let selectedSystemModel {
                Label(selectedSystemModel.displayName, systemImage: "checkmark.circle")
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            } else if let selectedModel {
                Label(selectedModel.displayName, systemImage: "checkmark.circle")
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            } else {
                Label("Select a model", systemImage: "circle")
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button(confirmTitle) {
                if let selectedSystemModel {
                    onConfirmSelection?(selectedSystemModel.selection)
                } else if let selectedModel {
                    if let onConfirmSelection {
                        onConfirmSelection(.installed(selectedModel.id))
                    } else {
                        onConfirmInstalledModel?(selectedModel)
                    }
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(!hasSelection || !canConfirmSelection || confirmDisabled)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private func startDownload(
        hfRepo: String,
        hfFilename: String,
        displayName: String,
        expectedSHA256: String?,
        contextLength: Int,
        source: ModelSource
    ) {
        downloadError = nil
        let download = ModelLibraryDownload(hfRepo: hfRepo, hfFilename: hfFilename, displayName: displayName)
        activeDownload = download
        download.start(
            expectedSHA256: expectedSHA256,
            expectedContextLength: contextLength,
            source: source,
            into: library
        ) { result in
            switch result {
            case .success(let model):
                if selectedModelID.isEmpty {
                    selectedModelID = model.id.uuidString
                }
            case .failure(let error):
                downloadError = error.localizedDescription
            }
            activeDownload = nil
            refresh()
        }
    }

    private func resume(_ partial: PartialDownload) {
        guard let repo = partial.hfRepo, let filename = partial.hfFilename else { return }
        let curated = curatedModels.first {
            $0.hfRepo == repo && $0.hfFilename == filename
        }
        startDownload(
            hfRepo: repo,
            hfFilename: filename,
            displayName: partial.displayName,
            expectedSHA256: curated?.sha256,
            contextLength: curated?.contextLength ?? 0,
            source: curated == nil ? .customHF : .curated
        )
    }

    private func importLocalGGUF() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "gguf") ?? .data]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.directoryURL = library.root

        guard panel.runModal() == .OK, let url = panel.url else { return }
        let didStart = url.startAccessingSecurityScopedResource()
        defer { if didStart { url.stopAccessingSecurityScopedResource() } }

        do {
            let model = try library.importFile(at: url)
            if selectedModelID.isEmpty {
                selectedModelID = model.id.uuidString
            }
            refresh()
        } catch {
            downloadError = error.localizedDescription
        }
    }

    private func delete(_ model: InstalledModel) {
        do {
            try library.delete(id: model.id)
            if selectedModelID == model.id.uuidString {
                selectedModelID = systemModels.first?.id ?? library.models.first?.id.uuidString ?? ""
            }
            refresh()
            onModelDeleted(model)
        } catch {
            downloadError = error.localizedDescription
        }
    }

    private func revealModelsFolder() {
        NSWorkspace.shared.activateFileViewerSelecting([library.root])
    }

    private func refresh() {
        library.refresh()
        normalizeSelection()
        refreshToken = UUID()
    }

    private func normalizeSelection() {
        if selectedModelID.isEmpty {
            selectedModelID = systemModels.first?.id ?? library.models.first?.id.uuidString ?? ""
        } else if selectedSystemModel == nil && selectedModel == nil {
            selectedModelID = systemModels.first?.id ?? library.models.first?.id.uuidString ?? ""
        }
    }

    private func recommendationSummary() -> String {
        let memory = formatBytes(Int64(min(ProcessInfo.processInfo.physicalMemory, UInt64(Int64.max))))
        if let recommendedCuratedModel {
            return "\(memory) RAM detected. Recommended for this Mac: \(recommendedCuratedModel.displayName)."
        }
        return "\(memory) RAM detected."
    }

    private func formatBytes(_ bytes: Int64) -> String {
        guard bytes > 0 else { return "0 B" }
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: bytes)
    }
}

@MainActor
@Observable
private final class ModelLibraryDownload {
    var progress = DownloadProgress(bytesReceived: 0, totalBytes: 0, bytesPerSecond: 0)
    var isRunning = false
    var errorMessage: String?

    let hfRepo: String
    let hfFilename: String
    let displayName: String

    @ObservationIgnored
    private var task: Task<Void, Never>?

    init(hfRepo: String, hfFilename: String, displayName: String) {
        self.hfRepo = hfRepo
        self.hfFilename = hfFilename
        self.displayName = displayName
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
                let result = try await ModelDownloader.download(
                    hfRepo: hfRepo,
                    hfFilename: hfFilename,
                    modelsRoot: library.root,
                    displayName: displayName,
                    expectedSHA256: expectedSHA256,
                    onProgress: { [weak self] progress in
                        Task { @MainActor [weak self] in
                            self?.progress = progress
                        }
                    }
                )
                let model = try library.add(
                    weightsAt: result.tempURL,
                    displayName: displayName,
                    filename: hfFilename,
                    sizeBytes: result.sizeBytes,
                    source: source,
                    hfRepo: hfRepo,
                    hfFilename: hfFilename,
                    sha256: result.sha256,
                    contextLength: expectedContextLength
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

private struct CustomHFSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var input = ""
    @State private var displayName = ""
    @State private var parseError: String?

    let onSubmit: (_ repo: String, _ filename: String, _ displayName: String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Download from Hugging Face")
                .font(.title3.bold())

            TextField(
                "https://huggingface.co/<repo>/resolve/main/<file>.gguf",
                text: $input,
                axis: .vertical
            )
            .textFieldStyle(.roundedBorder)
            .lineLimit(2...4)

            TextField("Display name", text: $displayName)
                .textFieldStyle(.roundedBorder)

            if let parseError {
                Label(parseError, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                    .font(.callout)
            }

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Download") { submit() }
                    .buttonStyle(.borderedProminent)
                    .disabled(input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 480)
    }

    private func submit() {
        guard let (repo, filename) = HuggingFaceURL.parse(input) else {
            parseError = "Expected a Hugging Face URL or repo/file.gguf path."
            return
        }

        let name = displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? filename.replacingOccurrences(of: ".gguf", with: "")
            : displayName
        onSubmit(repo, filename, name)
        dismiss()
    }
}
