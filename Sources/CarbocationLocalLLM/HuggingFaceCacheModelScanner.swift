import CryptoKit
import Foundation

struct HuggingFaceCacheModelScanner {
    private struct CacheFile: Hashable {
        var url: URL
        var path: String
        var sizeBytes: Int64
        var blobHash: String?

        var sha256: String? {
            guard let blobHash, blobHash.count == 64 else { return nil }
            return blobHash
        }
    }

    private struct SplitInfo: Hashable {
        var prefix: String
        var tag: String
        var index: Int
        var count: Int
    }

    var hubRoot: URL
    var fileManager: FileManager
    var contextLengthProbe: ModelContextLengthProbe?

    func scan() -> [InstalledModel] {
        guard let repos = try? fileManager.contentsOfDirectory(
            at: hubRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var models: [InstalledModel] = []
        for repoDirectory in repos.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            guard repoDirectory.lastPathComponent.hasPrefix("models--"),
                  isDirectory(repoDirectory),
                  let repo = repoID(fromCacheFolderName: repoDirectory.lastPathComponent)
            else { continue }

            models.append(contentsOf: scanRepository(repo: repo, directory: repoDirectory))
        }
        return models
    }

    private func scanRepository(repo: String, directory: URL) -> [InstalledModel] {
        let snapshots = snapshotDirectories(in: directory)
        var models: [InstalledModel] = []

        for snapshot in snapshots {
            let files = ggufFiles(in: snapshot.directory)
            guard !files.isEmpty else { continue }
            models.append(contentsOf: modelsInSnapshot(
                repo: repo,
                commit: snapshot.commit,
                snapshotDirectory: snapshot.directory,
                files: files
            ))
        }

        return models
    }

    private func snapshotDirectories(in repoDirectory: URL) -> [(commit: String, directory: URL)] {
        let snapshotsRoot = repoDirectory.appendingPathComponent("snapshots", isDirectory: true)
        var commits = resolvedRefCommits(in: repoDirectory)

        if commits.isEmpty,
           let snapshotEntries = try? fileManager.contentsOfDirectory(
            at: snapshotsRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
           ) {
            commits = snapshotEntries
                .filter(isDirectory)
                .map(\.lastPathComponent)
                .filter(Self.isCommitHash)
                .sorted()
        }

        return commits.compactMap { commit in
            let directory = snapshotsRoot.appendingPathComponent(commit, isDirectory: true)
            return isDirectory(directory) ? (commit, directory) : nil
        }
    }

    private func resolvedRefCommits(in repoDirectory: URL) -> [String] {
        let refsRoot = repoDirectory.appendingPathComponent("refs", isDirectory: true)
        guard isDirectory(refsRoot) else { return [] }

        var refs: [(name: String, commit: String)] = []
        if let main = readCommit(from: refsRoot.appendingPathComponent("main")) {
            refs.append(("main", main))
        }

        if let enumerator = fileManager.enumerator(
            at: refsRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for case let url as URL in enumerator {
                guard !isDirectory(url),
                      relativePath(for: url, under: refsRoot) != "main",
                      let commit = readCommit(from: url)
                else { continue }
                refs.append((relativePath(for: url, under: refsRoot), commit))
            }
        }

        var seen: Set<String> = []
        return refs
            .sorted {
                if $0.name == "main" { return true }
                if $1.name == "main" { return false }
                return $0.name.localizedStandardCompare($1.name) == .orderedAscending
            }
            .compactMap { ref in
                guard !seen.contains(ref.commit) else { return nil }
                seen.insert(ref.commit)
                return ref.commit
            }
    }

    private func readCommit(from url: URL) -> String? {
        guard let data = try? Data(contentsOf: url),
              let value = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
              Self.isCommitHash(value)
        else { return nil }
        return value
    }

    private func ggufFiles(in snapshotDirectory: URL) -> [CacheFile] {
        guard let enumerator = fileManager.enumerator(
            at: snapshotDirectory,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var files: [CacheFile] = []
        for case let url as URL in enumerator {
            guard url.pathExtension.lowercased() == "gguf",
                  fileManager.fileExists(atPath: url.path)
            else { continue }
            files.append(CacheFile(
                url: url,
                path: relativePath(for: url, under: snapshotDirectory),
                sizeBytes: fileSize(at: url),
                blobHash: blobHash(for: url)
            ))
        }
        return files.sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending }
    }

    private func modelsInSnapshot(
        repo: String,
        commit: String,
        snapshotDirectory: URL,
        files: [CacheFile]
    ) -> [InstalledModel] {
        let modelFiles = files.filter { Self.isModelGGUF($0.path) }
        var consumed: Set<String> = []
        var models: [InstalledModel] = []

        for file in modelFiles {
            guard !consumed.contains(file.path) else { continue }
            let split = splitInfo(for: file.path)

            if split.count > 1 {
                guard split.index == 1 else { continue }
                let splitFiles = completeSplitFiles(for: split, in: modelFiles)
                guard splitFiles.count == split.count else { continue }
                splitFiles.forEach { consumed.insert($0.path) }
                models.append(makeModel(
                    repo: repo,
                    commit: commit,
                    snapshotDirectory: snapshotDirectory,
                    primary: splitFiles[0],
                    splitFiles: Array(splitFiles.dropFirst()),
                    mmproj: selectMMProj(for: splitFiles[0], in: files)
                ))
            } else {
                consumed.insert(file.path)
                models.append(makeModel(
                    repo: repo,
                    commit: commit,
                    snapshotDirectory: snapshotDirectory,
                    primary: file,
                    splitFiles: [],
                    mmproj: selectMMProj(for: file, in: files)
                ))
            }
        }

        return models
    }

    private func completeSplitFiles(for primarySplit: SplitInfo, in files: [CacheFile]) -> [CacheFile] {
        let matches = files
            .filter {
                let split = splitInfo(for: $0.path)
                return split.prefix == primarySplit.prefix
                    && split.tag == primarySplit.tag
                    && split.count == primarySplit.count
            }
            .sorted { splitInfo(for: $0.path).index < splitInfo(for: $1.path).index }

        guard matches.count == primarySplit.count,
              matches.enumerated().allSatisfy({ offset, file in
                splitInfo(for: file.path).index == offset + 1
              })
        else { return [] }
        return matches
    }

    private func makeModel(
        repo: String,
        commit: String,
        snapshotDirectory: URL,
        primary: CacheFile,
        splitFiles: [CacheFile],
        mmproj: CacheFile?
    ) -> InstalledModel {
        var artifacts: [InstalledModelArtifact] = [
            artifact(primary, role: .primaryModel)
        ]
        artifacts.append(contentsOf: splitFiles.map { artifact($0, role: .splitModel) })
        if let mmproj {
            artifacts.append(artifact(mmproj, role: .mmproj))
        }

        let sizeBytes = artifacts.reduce(Int64(0)) { $0 + $1.sizeBytes }
        let resolvedContextLength = GGUFMetadata.trainingContextLength(at: primary.url)
            ?? contextLengthProbe?(primary.url)
            ?? 0
        let id = Self.stableModelID(
            repo: repo,
            primaryPath: primary.path,
            contentIdentifier: primary.blobHash ?? commit
        )
        return InstalledModel(
            id: id,
            displayName: displayName(for: primary.path),
            filename: primary.path,
            sizeBytes: sizeBytes,
            contextLength: resolvedContextLength,
            quantization: splitInfo(for: primary.path).tag.nilIfBlank
                ?? InstalledModel.inferQuantization(from: primary.path),
            source: source(for: repo, path: primary.path),
            hfRepo: repo,
            hfFilename: primary.path,
            sha256: primary.sha256,
            artifacts: artifacts,
            storageLocation: .external(directory: snapshotDirectory),
            installedAt: modificationDate(for: primary.url)
        )
    }

    private func artifact(_ file: CacheFile, role: InstalledModelArtifactRole) -> InstalledModelArtifact {
        InstalledModelArtifact(
            role: role,
            relativePath: file.path,
            sizeBytes: file.sizeBytes,
            sha256: file.sha256
        )
    }

    private func source(for repo: String, path: String) -> ModelSource {
        CuratedModelCatalog.all.contains {
            $0.hfRepo.caseInsensitiveCompare(repo) == .orderedSame
                && $0.hfFilename.caseInsensitiveCompare(path) == .orderedSame
        } ? .curated : .customHF
    }

    private func selectMMProj(for primary: CacheFile, in files: [CacheFile]) -> CacheFile? {
        MMProjSelector.selectBest(primaryPath: primary.path, candidates: files) { $0.path }
    }

    private func repoID(fromCacheFolderName name: String) -> String? {
        let prefix = "models--"
        guard name.hasPrefix(prefix), name.count > prefix.count else { return nil }
        return String(name.dropFirst(prefix.count)).replacingOccurrences(of: "--", with: "/")
    }

    private func blobHash(for url: URL) -> String? {
        guard let destination = try? fileManager.destinationOfSymbolicLink(atPath: url.path) else {
            return nil
        }
        return URL(fileURLWithPath: destination).lastPathComponent.nilIfBlank
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
            return Int64(min(size, UInt64(Int64.max)))
        }
        if let size = value as? Int {
            return Int64(size)
        }
        return 0
    }

    private func modificationDate(for url: URL) -> Date {
        (try? fileManager.attributesOfItem(atPath: url.path)[.modificationDate] as? Date)
            ?? Date(timeIntervalSince1970: 0)
    }

    private func isDirectory(_ url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        return fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
            && isDirectory.boolValue
    }

    private func relativePath(for url: URL, under root: URL) -> String {
        let rootPath = root.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        guard path.hasPrefix(rootPath) else { return url.lastPathComponent }
        let relative = path.dropFirst(rootPath.count).drop { $0 == "/" }
        return String(relative)
    }

    private static func isCommitHash(_ value: String) -> Bool {
        value.count == 40 && value.allSatisfy(\.isHexDigit)
    }

    private static func isModelGGUF(_ path: String) -> Bool {
        let filename = URL(fileURLWithPath: path).lastPathComponent.lowercased()
        return filename.hasSuffix(".gguf")
            && !filename.contains("mmproj")
            && !filename.contains("imatrix")
    }

    private func splitInfo(for path: String) -> SplitInfo {
        var prefix = path
        guard prefix.lowercased().hasSuffix(".gguf") else {
            return SplitInfo(prefix: path, tag: "", index: 1, count: 1)
        }
        prefix.removeLast(5)

        var index = 1
        var count = 1
        if let range = prefix.range(
            of: "-[0-9]{5}-of-[0-9]{5}$",
            options: [.regularExpression, .caseInsensitive]
        ) {
            let suffix = String(prefix[range]).dropFirst()
            let parts = suffix.split(separator: "-")
            if parts.count == 3,
               let parsedIndex = Int(parts[0]),
               let parsedCount = Int(parts[2]) {
                index = parsedIndex
                count = parsedCount
                prefix.removeSubrange(range)
            }
        }

        var tag = ""
        if let range = prefix.range(
            of: "[-.][A-Za-z0-9_]+$",
            options: [.regularExpression]
        ) {
            tag = String(prefix[range].dropFirst()).uppercased()
        }
        return SplitInfo(prefix: prefix, tag: tag, index: index, count: count)
    }

    private func displayName(for path: String) -> String {
        var name = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        if let range = name.range(
            of: "-[0-9]{5}-of-[0-9]{5}$",
            options: [.regularExpression, .caseInsensitive]
        ) {
            name.removeSubrange(range)
        }
        return name
    }

    private static func stableModelID(
        repo: String,
        primaryPath: String,
        contentIdentifier: String
    ) -> UUID {
        let input = "hf-cache|\(repo)|\(primaryPath)|\(contentIdentifier)"
        let digest = SHA256.hash(data: Data(input.utf8))
        var bytes = Array(digest.prefix(16))
        bytes[6] = (bytes[6] & 0x0f) | 0x50
        bytes[8] = (bytes[8] & 0x3f) | 0x80
        return UUID(uuid: (
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15]
        ))
    }
}

private extension String {
    var nilIfBlank: String? {
        let trimmed = trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : self
    }
}
