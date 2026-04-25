import Foundation

public enum ModelLibraryError: Error, LocalizedError, Sendable {
    case sourceFileMissing(URL)
    case destinationExists(URL)
    case metadataWriteFailed(String)
    case notAGGUF(String)

    public var errorDescription: String? {
        switch self {
        case .sourceFileMissing(let url):
            return "Source file not found: \(url.lastPathComponent)"
        case .destinationExists(let url):
            return "A model already exists at \(url.path)"
        case .metadataWriteFailed(let detail):
            return "Failed to save model metadata: \(detail)"
        case .notAGGUF(let filename):
            return "\(filename) is not a .gguf file."
        }
    }
}

public typealias ModelContextLengthProbe = @Sendable (URL) -> Int?

@MainActor
public final class ModelLibrary {
    public private(set) var models: [InstalledModel] = []
    public private(set) var partials: [PartialDownload] = []

    public let root: URL
    private let fileManager: FileManager
    private let contextLengthProbe: ModelContextLengthProbe?

    public init(
        root: URL,
        fileManager: FileManager = .default,
        contextLengthProbe: ModelContextLengthProbe? = nil
    ) {
        self.root = root
        self.fileManager = fileManager
        self.contextLengthProbe = contextLengthProbe
        try? fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        refresh()
    }

    public func refresh() {
        var found: [InstalledModel] = []
        let decoder = LocalLLMJSON.makeDecoder()

        guard let entries = try? fileManager.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            models = []
            partials = []
            return
        }

        for entry in entries {
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: entry.path, isDirectory: &isDirectory),
                  isDirectory.boolValue
            else { continue }

            let metadataURL = entry.appendingPathComponent("metadata.json")
            if let data = try? Data(contentsOf: metadataURL),
               let metadata = try? decoder.decode(InstalledModel.self, from: data) {
                found.append(metadata)
            } else if let synthesized = synthesizeMetadata(for: entry) {
                found.append(synthesized)
            }
        }

        models = found.sorted {
            $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending
        }
        partials = ModelDownloader.listPartials(in: root)
    }

    public func model(id: UUID) -> InstalledModel? {
        models.first { $0.id == id }
    }

    public func model(id: String) -> InstalledModel? {
        guard let uuid = UUID(uuidString: id) else { return nil }
        return model(id: uuid)
    }

    public func add(
        weightsAt tempURL: URL,
        displayName: String,
        filename: String,
        sizeBytes: Int64,
        source: ModelSource,
        hfRepo: String? = nil,
        hfFilename: String? = nil,
        sha256: String? = nil,
        contextLength: Int = 0,
        quantization: String? = nil
    ) throws -> InstalledModel {
        guard fileManager.fileExists(atPath: tempURL.path) else {
            throw ModelLibraryError.sourceFileMissing(tempURL)
        }
        guard filename.lowercased().hasSuffix(".gguf") else {
            throw ModelLibraryError.notAGGUF(filename)
        }

        let id = UUID()
        let directory = root.appendingPathComponent(id.uuidString, isDirectory: true)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let destination = directory.appendingPathComponent(filename)
        guard !fileManager.fileExists(atPath: destination.path) else {
            throw ModelLibraryError.destinationExists(destination)
        }

        try fileManager.moveItem(at: tempURL, to: destination)

        let resolvedContextLength = contextLengthProbe?(destination) ?? max(0, contextLength)
        let metadata = InstalledModel(
            id: id,
            displayName: displayName,
            filename: filename,
            sizeBytes: sizeBytes,
            contextLength: resolvedContextLength,
            quantization: quantization ?? InstalledModel.inferQuantization(from: filename),
            source: source,
            hfRepo: hfRepo,
            hfFilename: hfFilename,
            sha256: sha256,
            installedAt: Date()
        )

        do {
            try writeMetadata(metadata)
        } catch {
            try? fileManager.removeItem(at: directory)
            throw ModelLibraryError.metadataWriteFailed(error.localizedDescription)
        }

        refresh()
        return metadata
    }

    public func importFile(at sourceURL: URL, displayName: String? = nil) throws -> InstalledModel {
        let filename = sourceURL.lastPathComponent
        guard filename.lowercased().hasSuffix(".gguf") else {
            throw ModelLibraryError.notAGGUF(filename)
        }
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            throw ModelLibraryError.sourceFileMissing(sourceURL)
        }

        let temporaryURL = fileManager.temporaryDirectory
            .appendingPathComponent("cllm-import-\(UUID().uuidString).gguf")
        try fileManager.copyItem(at: sourceURL, to: temporaryURL)

        let attributes = try fileManager.attributesOfItem(atPath: temporaryURL.path)
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        let resolvedName = displayName?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
            ?? filename.replacingOccurrences(of: ".gguf", with: "")

        return try add(
            weightsAt: temporaryURL,
            displayName: resolvedName,
            filename: filename,
            sizeBytes: size,
            source: .imported
        )
    }

    public func delete(id: UUID) throws {
        let directory = root.appendingPathComponent(id.uuidString, isDirectory: true)
        if fileManager.fileExists(atPath: directory.path) {
            try fileManager.removeItem(at: directory)
        }
        refresh()
    }

    public func deletePartial(_ partial: PartialDownload) {
        ModelDownloader.deletePartial(partial)
        refresh()
    }

    public func totalDiskUsageBytes() -> Int64 {
        models.reduce(Int64(0)) { $0 + $1.sizeBytes }
            + partials.reduce(Int64(0)) { $0 + $1.bytesOnDisk }
    }

    public func syncContextLength(_ contextLength: Int, for id: UUID) throws {
        guard contextLength > 0,
              let index = models.firstIndex(where: { $0.id == id }),
              models[index].contextLength != contextLength
        else { return }

        models[index].contextLength = contextLength
        try writeMetadata(models[index])
    }

    public func writeMetadata(_ model: InstalledModel) throws {
        let encoder = LocalLLMJSON.makePrettyEncoder()
        let data = try encoder.encode(model)
        let url = model.metadataURL(in: root)
        try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }

    private func synthesizeMetadata(for directory: URL) -> InstalledModel? {
        guard let uuid = UUID(uuidString: directory.lastPathComponent),
              let files = try? fileManager.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.fileSizeKey]
              ),
              let gguf = files.first(where: { $0.pathExtension.lowercased() == "gguf" })
        else { return nil }

        let attributes = (try? fileManager.attributesOfItem(atPath: gguf.path)) ?? [:]
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        return InstalledModel(
            id: uuid,
            displayName: gguf.deletingPathExtension().lastPathComponent,
            filename: gguf.lastPathComponent,
            sizeBytes: size,
            contextLength: 0,
            quantization: InstalledModel.inferQuantization(from: gguf.lastPathComponent),
            source: .imported,
            installedAt: Date()
        )
    }
}

private extension String {
    var nilIfEmpty: String? {
        trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : self
    }
}
