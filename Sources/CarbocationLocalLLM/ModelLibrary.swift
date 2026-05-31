import Foundation

public enum ModelLibraryError: Error, LocalizedError, Sendable {
    case sourceFileMissing(URL)
    case destinationExists(URL)
    case metadataWriteFailed(String)
    case notAGGUF(String)
    case missingPrimaryArtifact
    case unsafeArtifactPath(String)
    case readOnlyModel(String)

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
        case .missingPrimaryArtifact:
            return "No primary GGUF model artifact was provided."
        case .unsafeArtifactPath(let path):
            return "Invalid model artifact path: \(path)"
        case .readOnlyModel(let name):
            return "\(name) is read-only because it is outside the managed model library."
        }
    }
}

public typealias ModelContextLengthProbe = @Sendable (URL) -> Int?

public struct ModelLibrarySearchConfiguration: Hashable, Sendable {
    public var includesManagedModels: Bool
    public var externalHuggingFaceHubCacheDirectories: [URL]

    public init(
        includesManagedModels: Bool = true,
        externalHuggingFaceHubCacheDirectories: [URL] = []
    ) {
        self.includesManagedModels = includesManagedModels
        self.externalHuggingFaceHubCacheDirectories = externalHuggingFaceHubCacheDirectories
    }

    public static let managedOnly = ModelLibrarySearchConfiguration()

    public static func platformDefault(
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ModelLibrarySearchConfiguration {
        #if os(macOS)
        let cacheDirectories = ModelStorage.huggingFaceHubCacheDirectory(
            environment: environment,
            fileManager: fileManager
        ).map { [$0] } ?? []
        return ModelLibrarySearchConfiguration(
            includesManagedModels: true,
            externalHuggingFaceHubCacheDirectories: cacheDirectories
        )
        #else
        _ = fileManager
        _ = environment
        return .managedOnly
        #endif
    }
}

@MainActor
public final class ModelLibrary {
    public private(set) var models: [InstalledModel] = []
    public private(set) var partials: [PartialDownload] = []

    public let root: URL
    private let fileWorker: ModelLibraryFileWorker

    public init(
        root: URL,
        fileManager: FileManager = .default,
        searchConfiguration: ModelLibrarySearchConfiguration = .platformDefault(),
        contextLengthProbe: ModelContextLengthProbe? = nil
    ) {
        self.root = root
        self.fileWorker = ModelLibraryFileWorker(
            root: root,
            fileManager: fileManager,
            searchConfiguration: searchConfiguration,
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

    public func add(
        artifacts: [ModelLibraryInstallArtifact],
        displayName: String,
        source: ModelSource,
        hfRepo: String? = nil,
        hfFilename: String? = nil,
        sha256: String? = nil,
        contextLength: Int = 0,
        quantization: String? = nil
    ) async throws -> InstalledModel {
        let result = try await fileWorker.add(
            artifacts: artifacts,
            displayName: displayName,
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

    public func importFiles(at sourceURLs: [URL], displayName: String? = nil) async throws -> InstalledModel {
        let result = try await fileWorker.importFiles(at: sourceURLs, displayName: displayName)
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
    private let root: URL
    private let fileManager: FileManager
    private let searchConfiguration: ModelLibrarySearchConfiguration
    private let contextLengthProbe: ModelContextLengthProbe?
    private let queue: DispatchQueue

    init(
        root: URL,
        fileManager: FileManager,
        searchConfiguration: ModelLibrarySearchConfiguration,
        contextLengthProbe: ModelContextLengthProbe?
    ) {
        self.root = root
        self.fileManager = fileManager
        self.searchConfiguration = searchConfiguration
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
                artifacts: [
                    ModelLibraryInstallArtifact(
                        sourceURL: tempURL,
                        role: .primaryModel,
                        relativePath: filename,
                        sizeBytes: sizeBytes,
                        sha256: sha256
                    )
                ],
                displayName: displayName,
                source: source,
                hfRepo: hfRepo,
                hfFilename: hfFilename,
                sha256: sha256,
                contextLength: contextLength,
                quantization: quantization
            )
        }
    }

    func add(
        artifacts: [ModelLibraryInstallArtifact],
        displayName: String,
        source: ModelSource,
        hfRepo: String?,
        hfFilename: String?,
        sha256: String?,
        contextLength: Int,
        quantization: String?
    ) async throws -> ModelLibraryInstallResult {
        try await run {
            try self.install(
                artifacts: artifacts,
                displayName: displayName,
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
        try await importFiles(at: [sourceURL], displayName: displayName)
    }

    func importFiles(at sourceURLs: [URL], displayName: String?) async throws -> ModelLibraryInstallResult {
        try await run {
            guard let primaryURL = sourceURLs.first(where: { Self.isModelGGUF($0.lastPathComponent) }) else {
                if let invalid = sourceURLs.first(where: { !$0.lastPathComponent.lowercased().hasSuffix(".gguf") }) {
                    throw ModelLibraryError.notAGGUF(invalid.lastPathComponent)
                }
                throw ModelLibraryError.missingPrimaryArtifact
            }
            guard self.fileManager.fileExists(atPath: primaryURL.path) else {
                throw ModelLibraryError.sourceFileMissing(primaryURL)
            }

            let resolvedName = displayName?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
                ?? primaryURL.lastPathComponent.replacingOccurrences(of: ".gguf", with: "")

            return try self.install(
                artifacts: self.importArtifacts(primaryURL: primaryURL, selectedURLs: sourceURLs),
                displayName: resolvedName,
                source: .imported,
                hfRepo: nil,
                hfFilename: nil,
                sha256: nil,
                contextLength: 0,
                quantization: nil
            )
        }
    }

    private func importArtifacts(primaryURL: URL, selectedURLs: [URL]) -> [ModelLibraryInstallArtifact] {
        var artifacts = [
            ModelLibraryInstallArtifact(
                sourceURL: primaryURL,
                role: .primaryModel,
                relativePath: primaryURL.lastPathComponent,
                sizeBytes: 0,
                copySource: true
            )
        ]

        if let mmprojURL = selectMMProj(for: primaryURL, selectedURLs: selectedURLs) {
            artifacts.append(ModelLibraryInstallArtifact(
                sourceURL: mmprojURL,
                role: .mmproj,
                relativePath: mmprojURL.lastPathComponent,
                sizeBytes: 0,
                copySource: true
            ))
        }

        return artifacts
    }

    private func selectMMProj(for primaryURL: URL, selectedURLs: [URL]) -> URL? {
        let selectedPaths = Set(selectedURLs.map { $0.standardizedFileURL.path })
        let primaryPath = primaryURL.standardizedFileURL.path
        var candidates = selectedURLs

        if let siblingURLs = try? fileManager.contentsOfDirectory(
            at: primaryURL.deletingLastPathComponent(),
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) {
            candidates.append(contentsOf: siblingURLs)
        }

        var seen: Set<String> = []
        let primaryBits = Self.quantBits(for: primaryURL.lastPathComponent)
        var best: (url: URL, selected: Bool, diff: Int)?

        for candidate in candidates {
            let path = candidate.standardizedFileURL.path
            guard path != primaryPath,
                  seen.insert(path).inserted,
                  Self.isMMProj(candidate.lastPathComponent),
                  fileManager.fileExists(atPath: path),
                  fileManager.isReadableFile(atPath: path)
            else { continue }

            let isSelected = selectedPaths.contains(path)
            let diff = abs(Self.quantBits(for: candidate.lastPathComponent) - primaryBits)
            if best == nil
                || (isSelected && !best!.selected)
                || (isSelected == best!.selected && diff < best!.diff)
                || (isSelected == best!.selected
                    && diff == best!.diff
                    && candidate.lastPathComponent.localizedStandardCompare(best!.url.lastPathComponent) == .orderedAscending) {
                best = (candidate, isSelected, diff)
            }
        }

        return best?.url
    }

    func delete(id: UUID) async throws -> ModelLibrarySnapshot {
        try await run {
            let currentSnapshot = self.loadSnapshot()
            if let model = currentSnapshot.models.first(where: { $0.id == id }),
               model.isReadOnly {
                throw ModelLibraryError.readOnlyModel(model.displayName)
            }

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
            guard !snapshot.models[index].isReadOnly else {
                throw ModelLibraryError.readOnlyModel(snapshot.models[index].displayName)
            }

            snapshot.models[index].contextLength = contextLength
            try self.writeMetadata(snapshot.models[index], root: self.root)
            return self.loadSnapshot()
        }
    }

    func writeMetadata(_ model: InstalledModel) async throws -> ModelLibrarySnapshot {
        try await run {
            guard !model.isReadOnly else {
                throw ModelLibraryError.readOnlyModel(model.displayName)
            }
            try self.writeMetadata(model, root: self.root)
            return self.loadSnapshot()
        }
    }

    private func install(
        artifacts installArtifacts: [ModelLibraryInstallArtifact],
        displayName: String,
        source: ModelSource,
        hfRepo: String?,
        hfFilename: String?,
        sha256: String?,
        contextLength: Int,
        quantization: String?
    ) throws -> ModelLibraryInstallResult {
        guard let primaryArtifact = installArtifacts.first(where: { $0.role == .primaryModel }) else {
            throw ModelLibraryError.missingPrimaryArtifact
        }
        guard primaryArtifact.relativePath.lowercased().hasSuffix(".gguf") else {
            throw ModelLibraryError.notAGGUF(primaryArtifact.relativePath)
        }
        for artifact in installArtifacts {
            try validateRelativeArtifactPath(artifact.relativePath)
            guard fileManager.fileExists(atPath: artifact.sourceURL.path) else {
                throw ModelLibraryError.sourceFileMissing(artifact.sourceURL)
            }
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

            var metadataArtifacts: [InstalledModelArtifact] = []
            metadataArtifacts.reserveCapacity(installArtifacts.count)
            for artifact in installArtifacts {
                let destination = stagingDirectory.appendingPathComponent(artifact.relativePath)
                try fileManager.createDirectory(
                    at: destination.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                if artifact.copySource {
                    try fileManager.copyItem(at: artifact.sourceURL, to: destination)
                } else {
                    try fileManager.moveItem(at: artifact.sourceURL, to: destination)
                }
                metadataArtifacts.append(InstalledModelArtifact(
                    role: artifact.role,
                    relativePath: artifact.relativePath,
                    sizeBytes: artifact.sizeBytes > 0 ? artifact.sizeBytes : fileSize(at: destination),
                    sha256: artifact.sha256
                ))
            }

            let primaryDestination = stagingDirectory.appendingPathComponent(primaryArtifact.relativePath)
            let sizeBytes = metadataArtifacts.reduce(Int64(0)) { $0 + $1.sizeBytes }
            let primaryFilename = primaryArtifact.relativePath

            let resolvedContextLength: Int
            if contextLength > 0 {
                resolvedContextLength = contextLength
            } else {
                resolvedContextLength = GGUFMetadata.trainingContextLength(at: primaryDestination)
                    ?? contextLengthProbe?(primaryDestination)
                    ?? 0
            }
            let metadata = InstalledModel(
                id: id,
                displayName: displayName,
                filename: primaryFilename,
                sizeBytes: sizeBytes,
                contextLength: resolvedContextLength,
                quantization: quantization ?? InstalledModel.inferQuantization(from: primaryFilename),
                source: source,
                hfRepo: hfRepo,
                hfFilename: hfFilename,
                sha256: sha256,
                artifacts: metadataArtifacts,
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

        if searchConfiguration.includesManagedModels {
            try? ensureRootExists()
            if let entries = try? fileManager.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
                options: [.skipsHiddenFiles]
            ) {
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
            }
        }

        for cacheDirectory in searchConfiguration.externalHuggingFaceHubCacheDirectories {
            let scanner = HuggingFaceCacheModelScanner(
                hubRoot: cacheDirectory,
                fileManager: fileManager,
                contextLengthProbe: contextLengthProbe
            )
            found.append(contentsOf: scanner.scan())
        }

        let deduplicated = deduplicate(found)

        return ModelLibrarySnapshot(
            models: deduplicated.sorted(by: Self.modelSortPrecedes),
            partials: searchConfiguration.includesManagedModels ? ModelDownloader.listPartials(in: root) : []
        )
    }

    private static func modelSortPrecedes(_ lhs: InstalledModel, _ rhs: InstalledModel) -> Bool {
        if lhs.sizeBytes != rhs.sizeBytes {
            return lhs.sizeBytes < rhs.sizeBytes
        }

        let nameComparison = lhs.displayName.localizedCaseInsensitiveCompare(rhs.displayName)
        if nameComparison != .orderedSame {
            return nameComparison == .orderedAscending
        }

        return lhs.id.uuidString < rhs.id.uuidString
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
            artifacts: [
                InstalledModelArtifact(
                    role: .primaryModel,
                    relativePath: gguf.lastPathComponent,
                    sizeBytes: fileSize(at: gguf)
                )
            ],
            installedAt: Date()
        )
    }

    private func writeMetadata(_ model: InstalledModel, root: URL) throws {
        guard !model.isReadOnly else {
            throw ModelLibraryError.readOnlyModel(model.displayName)
        }
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

    private func validateRelativeArtifactPath(_ path: String) throws {
        guard !path.isEmpty,
              !path.hasPrefix("/"),
              !path.split(separator: "/").contains("..")
        else {
            throw ModelLibraryError.unsafeArtifactPath(path)
        }
    }

    private func fileSize(at url: URL) -> Int64 {
        if let handle = try? FileHandle(forReadingFrom: url) {
            defer { try? handle.close() }
            if let size = try? handle.seekToEnd() {
                return Int64(min(size, UInt64(Int64.max)))
            }
        }

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

    private func deduplicate(_ models: [InstalledModel]) -> [InstalledModel] {
        let managedModels = models.filter { !$0.isReadOnly }
        let managedRepoFiles = Set(managedModels.compactMap(repoFileKey))
        let managedExactKeys = Set(managedModels.compactMap(exactContentKey))

        var seenExternalExactKeys: Set<String> = []
        var seenExternalRepoFiles: Set<String> = []
        var result = managedModels

        for model in models where model.isReadOnly {
            if let repoFile = repoFileKey(model),
               managedRepoFiles.contains(repoFile) {
                continue
            }
            if let exact = exactContentKey(model),
               managedExactKeys.contains(exact) || seenExternalExactKeys.contains(exact) {
                continue
            }
            if let repoFile = repoFileKey(model),
               seenExternalRepoFiles.contains(repoFile) {
                continue
            }

            if let exact = exactContentKey(model) {
                seenExternalExactKeys.insert(exact)
            }
            if let repoFile = repoFileKey(model) {
                seenExternalRepoFiles.insert(repoFile)
            }
            result.append(model)
        }

        return result
    }

    private func repoFileKey(_ model: InstalledModel) -> String? {
        guard let repo = model.hfRepo?.lowercased(),
              let filename = model.hfFilename?.lowercased()
        else { return nil }
        return "\(repo)|\(filename)"
    }

    private func exactContentKey(_ model: InstalledModel) -> String? {
        guard let repoFile = repoFileKey(model) else { return nil }
        return "\(repoFile)|\(model.sha256?.lowercased() ?? "-")"
    }

    private struct SplitInfo {
        var tag: String
    }

    private static func isModelGGUF(_ path: String) -> Bool {
        let filename = URL(fileURLWithPath: path).lastPathComponent.lowercased()
        return filename.hasSuffix(".gguf")
            && !filename.contains("mmproj")
            && !filename.contains("imatrix")
    }

    private static func isMMProj(_ path: String) -> Bool {
        let filename = URL(fileURLWithPath: path).lastPathComponent.lowercased()
        return filename.hasSuffix(".gguf") && filename.contains("mmproj")
    }

    private static func quantBits(for path: String) -> Int {
        let tag = splitInfo(for: path).tag
        guard let digit = tag.firstIndex(where: { $0.isNumber }) else { return 0 }
        return Int(tag[digit...].prefix { $0.isNumber }) ?? 0
    }

    private static func splitInfo(for path: String) -> SplitInfo {
        var prefix = path
        if prefix.lowercased().hasSuffix(".gguf") {
            prefix.removeLast(5)
        }
        if let range = prefix.range(
            of: "-[0-9]{5}-of-[0-9]{5}$",
            options: [.regularExpression, .caseInsensitive]
        ) {
            prefix.removeSubrange(range)
        }

        var tag = ""
        if let range = prefix.range(
            of: "[-.][A-Za-z0-9_]+$",
            options: [.regularExpression]
        ) {
            tag = String(prefix[range].dropFirst()).uppercased()
        }
        return SplitInfo(tag: tag)
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
