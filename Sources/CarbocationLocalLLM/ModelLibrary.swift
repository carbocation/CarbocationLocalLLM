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
    private let fileWorker: ModelLibraryFileWorker

    public init(
        root: URL,
        fileManager: FileManager = .default,
        contextLengthProbe: ModelContextLengthProbe? = nil
    ) {
        self.root = root
        self.fileWorker = ModelLibraryFileWorker(
            root: root,
            fileManager: fileManager,
            contextLengthProbe: contextLengthProbe
        )
    }

    public func refresh() async {
        apply(await fileWorker.snapshot())
    }

    public func resolveInstalledModel(id: UUID, refreshing: Bool = true) async -> InstalledModel? {
        if refreshing {
            await refresh()
        }
        return model(id: id)
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
    ) async throws -> InstalledModel {
        let result = try await fileWorker.add(
            weightsAt: tempURL,
            displayName: displayName,
            filename: filename,
            sizeBytes: sizeBytes,
            source: source,
            hfRepo: hfRepo,
            hfFilename: hfFilename,
            sha256: sha256,
            contextLength: contextLength,
            quantization: quantization
        )
        apply(result.snapshot)
        return result.model
    }

    public func importFile(at sourceURL: URL, displayName: String? = nil) async throws -> InstalledModel {
        let result = try await fileWorker.importFile(at: sourceURL, displayName: displayName)
        apply(result.snapshot)
        return result.model
    }

    public func delete(id: UUID) async throws {
        apply(try await fileWorker.delete(id: id))
    }

    public func deletePartial(_ partial: PartialDownload) async {
        apply(await fileWorker.deletePartial(partial))
    }

    public func totalDiskUsageBytes() -> Int64 {
        models.reduce(Int64(0)) { $0 + $1.sizeBytes }
            + partials.reduce(Int64(0)) { $0 + $1.bytesOnDisk }
    }

    public func syncContextLength(_ contextLength: Int, for id: UUID) async throws {
        apply(try await fileWorker.syncContextLength(contextLength, for: id))
    }

    public func writeMetadata(_ model: InstalledModel) async throws {
        apply(try await fileWorker.writeMetadata(model))
    }

    private func apply(_ snapshot: ModelLibrarySnapshot) {
        models = snapshot.models
        partials = snapshot.partials
    }
}

private struct ModelLibrarySnapshot: Sendable, Hashable {
    var models: [InstalledModel]
    var partials: [PartialDownload]

    static let empty = ModelLibrarySnapshot(models: [], partials: [])
}

private struct ModelLibraryInstallResult: Sendable, Hashable {
    var model: InstalledModel
    var snapshot: ModelLibrarySnapshot
}

private final class ModelLibraryFileWorker: @unchecked Sendable {
    private enum InstallStrategy {
        case move(URL)
        case copy(URL)
    }

    private let root: URL
    private let fileManager: FileManager
    private let contextLengthProbe: ModelContextLengthProbe?
    private let queue: DispatchQueue

    init(
        root: URL,
        fileManager: FileManager,
        contextLengthProbe: ModelContextLengthProbe?
    ) {
        self.root = root
        self.fileManager = fileManager
        self.contextLengthProbe = contextLengthProbe
        self.queue = DispatchQueue(
            label: "com.carbocation.CarbocationLocalLLM.ModelLibraryFileWorker",
            qos: .utility
        )
    }

    func snapshot() async -> ModelLibrarySnapshot {
        await runSafely {
            self.loadSnapshot()
        }
    }

    func add(
        weightsAt tempURL: URL,
        displayName: String,
        filename: String,
        sizeBytes: Int64,
        source: ModelSource,
        hfRepo: String?,
        hfFilename: String?,
        sha256: String?,
        contextLength: Int,
        quantization: String?
    ) async throws -> ModelLibraryInstallResult {
        try await run {
            try self.install(
                strategy: .move(tempURL),
                displayName: displayName,
                filename: filename,
                sizeBytes: sizeBytes,
                source: source,
                hfRepo: hfRepo,
                hfFilename: hfFilename,
                sha256: sha256,
                contextLength: contextLength,
                quantization: quantization
            )
        }
    }

    func importFile(at sourceURL: URL, displayName: String?) async throws -> ModelLibraryInstallResult {
        try await run {
            let filename = sourceURL.lastPathComponent
            guard filename.lowercased().hasSuffix(".gguf") else {
                throw ModelLibraryError.notAGGUF(filename)
            }
            guard self.fileManager.fileExists(atPath: sourceURL.path) else {
                throw ModelLibraryError.sourceFileMissing(sourceURL)
            }

            let resolvedName = displayName?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
                ?? filename.replacingOccurrences(of: ".gguf", with: "")

            return try self.install(
                strategy: .copy(sourceURL),
                displayName: resolvedName,
                filename: filename,
                sizeBytes: 0,
                source: .imported,
                hfRepo: nil,
                hfFilename: nil,
                sha256: nil,
                contextLength: 0,
                quantization: nil
            )
        }
    }

    func delete(id: UUID) async throws -> ModelLibrarySnapshot {
        try await run {
            try self.ensureRootExists()
            let directory = self.root.appendingPathComponent(id.uuidString, isDirectory: true)
            if self.fileManager.fileExists(atPath: directory.path) {
                try self.fileManager.removeItem(at: directory)
            }
            return self.loadSnapshot()
        }
    }

    func deletePartial(_ partial: PartialDownload) async -> ModelLibrarySnapshot {
        await runSafely {
            ModelDownloader.deletePartial(partial)
            return self.loadSnapshot()
        }
    }

    func syncContextLength(_ contextLength: Int, for id: UUID) async throws -> ModelLibrarySnapshot {
        try await run {
            guard contextLength > 0 else {
                return self.loadSnapshot()
            }

            var snapshot = self.loadSnapshot()
            guard let index = snapshot.models.firstIndex(where: { $0.id == id }),
                  snapshot.models[index].contextLength != contextLength
            else { return snapshot }

            snapshot.models[index].contextLength = contextLength
            try self.writeMetadata(snapshot.models[index], root: self.root)
            return self.loadSnapshot()
        }
    }

    func writeMetadata(_ model: InstalledModel) async throws -> ModelLibrarySnapshot {
        try await run {
            try self.writeMetadata(model, root: self.root)
            return self.loadSnapshot()
        }
    }

    private func install(
        strategy: InstallStrategy,
        displayName: String,
        filename: String,
        sizeBytes requestedSizeBytes: Int64,
        source: ModelSource,
        hfRepo: String?,
        hfFilename: String?,
        sha256: String?,
        contextLength: Int,
        quantization: String?
    ) throws -> ModelLibraryInstallResult {
        guard filename.lowercased().hasSuffix(".gguf") else {
            throw ModelLibraryError.notAGGUF(filename)
        }

        let sourceURL: URL
        switch strategy {
        case .move(let url), .copy(let url):
            sourceURL = url
        }
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            throw ModelLibraryError.sourceFileMissing(sourceURL)
        }

        try ensureRootExists()
        let id = UUID()
        let finalDirectory = root.appendingPathComponent(id.uuidString, isDirectory: true)
        guard !fileManager.fileExists(atPath: finalDirectory.path) else {
            throw ModelLibraryError.destinationExists(finalDirectory)
        }

        let stagingRoot = root.appendingPathComponent(".staging", isDirectory: true)
        let stagingDirectory = stagingRoot.appendingPathComponent(
            "\(id.uuidString)-\(UUID().uuidString)",
            isDirectory: true
        )

        do {
            try fileManager.createDirectory(at: stagingDirectory, withIntermediateDirectories: true)
            let destination = stagingDirectory.appendingPathComponent(filename)
            try fileManager.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            switch strategy {
            case .move(let tempURL):
                try fileManager.moveItem(at: tempURL, to: destination)
            case .copy(let sourceURL):
                try fileManager.copyItem(at: sourceURL, to: destination)
            }

            let sizeBytes: Int64
            if requestedSizeBytes > 0 {
                sizeBytes = requestedSizeBytes
            } else {
                sizeBytes = fileSize(at: destination)
            }

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
                try writeMetadata(metadata, directory: stagingDirectory)
            } catch {
                throw ModelLibraryError.metadataWriteFailed(error.localizedDescription)
            }

            guard !fileManager.fileExists(atPath: finalDirectory.path) else {
                throw ModelLibraryError.destinationExists(finalDirectory)
            }

            try fileManager.moveItem(at: stagingDirectory, to: finalDirectory)
            return ModelLibraryInstallResult(model: metadata, snapshot: loadSnapshot())
        } catch {
            try? fileManager.removeItem(at: stagingDirectory)
            removeDirectoryIfOwned(finalDirectory, expectedID: id)
            throw error
        }
    }

    private func loadSnapshot() -> ModelLibrarySnapshot {
        var found: [InstalledModel] = []
        let decoder = LocalLLMJSON.makeDecoder()

        try? ensureRootExists()
        guard let entries = try? fileManager.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return .empty
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

        return ModelLibrarySnapshot(
            models: found.sorted {
                $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending
            },
            partials: ModelDownloader.listPartials(in: root)
        )
    }

    private func synthesizeMetadata(for directory: URL) -> InstalledModel? {
        guard let uuid = UUID(uuidString: directory.lastPathComponent),
              let files = try? fileManager.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.fileSizeKey]
              ),
              let gguf = files.first(where: { $0.pathExtension.lowercased() == "gguf" })
        else { return nil }

        return InstalledModel(
            id: uuid,
            displayName: gguf.deletingPathExtension().lastPathComponent,
            filename: gguf.lastPathComponent,
            sizeBytes: fileSize(at: gguf),
            contextLength: 0,
            quantization: InstalledModel.inferQuantization(from: gguf.lastPathComponent),
            source: .imported,
            installedAt: Date()
        )
    }

    private func writeMetadata(_ model: InstalledModel, root: URL) throws {
        try writeMetadata(model, directory: model.directory(in: root))
    }

    private func writeMetadata(_ model: InstalledModel, directory: URL) throws {
        let encoder = LocalLLMJSON.makePrettyEncoder()
        let data = try encoder.encode(model)
        let url = directory.appendingPathComponent("metadata.json")
        try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }

    private func ensureRootExists() throws {
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
    }

    private func fileSize(at url: URL) -> Int64 {
        guard let value = try? fileManager.attributesOfItem(atPath: url.path)[.size] else {
            return 0
        }

        if let number = value as? NSNumber {
            return number.int64Value
        }
        if let size = value as? Int64 {
            return size
        }
        if let size = value as? UInt64 {
            return Int64(size)
        }
        if let size = value as? Int {
            return Int64(size)
        }
        return 0
    }

    private func removeDirectoryIfOwned(_ directory: URL, expectedID: UUID) {
        let metadataURL = directory.appendingPathComponent("metadata.json")
        guard let data = try? Data(contentsOf: metadataURL),
              let metadata = try? LocalLLMJSON.makeDecoder().decode(InstalledModel.self, from: data),
              metadata.id == expectedID
        else { return }

        try? fileManager.removeItem(at: directory)
    }

    private func run<T: Sendable>(_ operation: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    continuation.resume(returning: try operation())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func runSafely(_ operation: @escaping @Sendable () -> ModelLibrarySnapshot) async -> ModelLibrarySnapshot {
        await withCheckedContinuation { continuation in
            queue.async {
                continuation.resume(returning: operation())
            }
        }
    }
}

private extension String {
    var nilIfEmpty: String? {
        trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : self
    }
}
