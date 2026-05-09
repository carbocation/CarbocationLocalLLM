import CryptoKit
import Foundation

extension ModelDownloader {
    public static func partialsDirectory(in modelsRoot: URL) throws -> URL {
        let directory = modelsRoot.appendingPathComponent(".partials", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    public static func listPartials(in modelsRoot: URL) -> [PartialDownload] {
        guard let partialsRoot = try? partialsDirectory(in: modelsRoot),
              let entries = try? FileManager.default.contentsOfDirectory(
                at: partialsRoot,
                includingPropertiesForKeys: [.fileSizeKey],
                options: []
              )
        else { return [] }

        var result: [PartialDownload] = []
        for sidecarURL in entries where isPartialSidecar(sidecarURL) {
            let stem = sidecarURL.deletingPathExtension().lastPathComponent
            let partialURL = sidecarURL.deletingLastPathComponent().appendingPathComponent("\(stem).gguf")
            guard FileManager.default.fileExists(atPath: partialURL.path) else {
                try? FileManager.default.removeItem(at: sidecarURL)
                continue
            }
            guard let data = try? Data(contentsOf: sidecarURL),
                  let sidecar = try? JSONDecoder().decode(PartialSidecar.self, from: data),
                  let sourceURL = URL(string: sidecar.url),
                  sidecar.totalBytes > 0
            else { continue }

            let key = partialKey(fromStem: stem) ?? stem
            let bytesOnDisk = bytesOnDisk(for: sidecar, partialURL: partialURL)
            let (hfRepo, hfFilename) = parseHFCoords(from: sourceURL) ?? (nil, nil)
            let displayName = sidecar.displayName
                ?? hfFilename?.replacingOccurrences(of: ".gguf", with: "")
                ?? sourceURL.lastPathComponent

            result.append(PartialDownload(
                id: key,
                partialURL: partialURL,
                sidecarURL: sidecarURL,
                sourceURL: sourceURL,
                displayName: displayName,
                hfRepo: hfRepo,
                hfFilename: hfFilename,
                totalBytes: sidecar.totalBytes,
                bytesOnDisk: min(bytesOnDisk, sidecar.totalBytes)
            ))
        }

        return result.sorted {
            $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending
        }
    }

    public static func deletePartial(_ partial: PartialDownload) {
        discardPartial(partialURL: partial.partialURL, sidecarURL: partial.sidecarURL)
    }


    static func partialPaths(for url: URL, modelsRoot: URL) throws -> (partial: URL, sidecar: URL) {
        let directory = try partialsDirectory(in: modelsRoot)
        let key = partialKey(for: url)

        return partialURLs(prefix: currentPartialPrefix, key: key, directory: directory)
    }

    private static func partialURLs(
        prefix: String,
        key: String,
        directory: URL
    ) -> (partial: URL, sidecar: URL) {
        (
            directory.appendingPathComponent("\(prefix)\(key).gguf"),
            directory.appendingPathComponent("\(prefix)\(key).json")
        )
    }

    private static func partialKey(for url: URL) -> String {
        SHA256.hash(data: Data(url.absoluteString.utf8))
            .prefix(6)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func isPartialSidecar(_ url: URL) -> Bool {
        guard url.pathExtension == "json" else { return false }
        return partialKey(fromStem: url.deletingPathExtension().lastPathComponent) != nil
    }

    private static func partialKey(fromStem stem: String) -> String? {
        guard stem.hasPrefix(currentPartialPrefix) else { return nil }
        return String(stem.dropFirst(currentPartialPrefix.count))
    }

    static func loadChunkedState(
        partialURL: URL,
        sidecarURL: URL,
        sourceURL: URL
    ) -> (ChunkPlan, PartialSidecar)? {
        guard FileManager.default.fileExists(atPath: partialURL.path),
              FileManager.default.fileExists(atPath: sidecarURL.path),
              let data = try? Data(contentsOf: sidecarURL),
              var sidecar = try? JSONDecoder().decode(PartialSidecar.self, from: data),
              sidecar.schemaVersion == 2,
              sidecar.url == sourceURL.absoluteString,
              sidecar.totalBytes > 0
        else { return nil }

        let chunkSize = sidecar.chunkSize ?? ChunkPlan.defaultChunkSize
        var plan = ChunkPlan(
            totalBytes: sidecar.totalBytes,
            chunkSize: chunkSize,
            doneChunks: Set(sidecar.doneChunks ?? [])
        )
        plan.doneChunks = plan.doneChunks.filter { $0 >= 0 && $0 < plan.chunkCount }
        sidecar.doneChunks = Array(plan.doneChunks).sorted()
        return (plan, sidecar)
    }

    static func loadSingleStreamState(
        partialURL: URL,
        sidecarURL: URL,
        sourceURL: URL
    ) -> PartialDownloadState? {
        guard FileManager.default.fileExists(atPath: partialURL.path),
              FileManager.default.fileExists(atPath: sidecarURL.path),
              let data = try? Data(contentsOf: sidecarURL),
              let sidecar = try? JSONDecoder().decode(PartialSidecar.self, from: data),
              sidecar.schemaVersion == nil || sidecar.schemaVersion == 1,
              sidecar.url == sourceURL.absoluteString,
              sidecar.totalBytes > 0
        else { return nil }

        let existingSize = fileSize(at: partialURL)
        guard existingSize > 0, existingSize < sidecar.totalBytes else {
            discardPartial(partialURL: partialURL, sidecarURL: sidecarURL)
            return nil
        }

        return PartialDownloadState(
            existingSize: existingSize,
            totalBytes: sidecar.totalBytes,
            etag: sidecar.etag,
            lastModified: sidecar.lastModified
        )
    }

    static func writeSingleStreamSidecar(
        to url: URL,
        sourceURL: URL,
        totalBytes: Int64,
        etag: String?,
        lastModified: String?,
        displayName: String?
    ) {
        let sidecar = PartialSidecar(
            url: sourceURL.absoluteString,
            etag: etag,
            lastModified: lastModified,
            totalBytes: totalBytes,
            displayName: displayName,
            schemaVersion: 1,
            chunkSize: nil,
            doneChunks: nil
        )
        writeSidecarValue(sidecar, to: url)
    }

    static func writeSidecarValue(_ sidecar: PartialSidecar, to url: URL) {
        if let data = try? JSONEncoder().encode(sidecar) {
            try? data.write(to: url, options: .atomic)
        }
    }

    static func discardPartial(partialURL: URL, sidecarURL: URL) {
        try? FileManager.default.removeItem(at: partialURL)
        try? FileManager.default.removeItem(at: sidecarURL)
    }

    private static func bytesOnDisk(for sidecar: PartialSidecar, partialURL: URL) -> Int64 {
        if sidecar.schemaVersion == 2,
           let chunkSize = sidecar.chunkSize,
           let doneChunks = sidecar.doneChunks {
            var plan = ChunkPlan(totalBytes: sidecar.totalBytes, chunkSize: chunkSize, doneChunks: Set(doneChunks))
            plan.doneChunks = plan.doneChunks.filter { $0 >= 0 && $0 < plan.chunkCount }
            return plan.completedBytes()
        }
        return fileSize(at: partialURL)
    }

    private static func fileSize(at url: URL) -> Int64 {
        guard let value = try? FileManager.default
            .attributesOfItem(atPath: url.path)[.size]
        else { return 0 }

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

}
