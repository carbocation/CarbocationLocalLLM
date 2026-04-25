import CryptoKit
import Darwin
import Foundation
import OSLog

private let modelDownloaderLog = Logger(
    subsystem: "com.carbocation.CarbocationLocalLLM",
    category: "ModelDownloader"
)

public enum ModelDownloaderError: Error, LocalizedError, Sendable {
    case httpStatus(Int)
    case badURL(String)
    case hashMismatch(expected: String, actual: String)
    case noContentLength
    case incompleteResponse
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .httpStatus(let code):
            return "Download failed with HTTP \(code)."
        case .badURL(let value):
            return "Invalid download URL: \(value)"
        case .hashMismatch(let expected, let actual):
            return "SHA256 mismatch; expected \(expected.prefix(12)), got \(actual.prefix(12))."
        case .noContentLength:
            return "Server did not report a content length."
        case .incompleteResponse:
            return "Server closed the connection before the model download completed."
        case .cancelled:
            return "Download cancelled."
        }
    }
}

public struct DownloadProgress: Sendable, Hashable {
    public let bytesReceived: Int64
    public let totalBytes: Int64
    public let bytesPerSecond: Double

    public init(bytesReceived: Int64, totalBytes: Int64, bytesPerSecond: Double) {
        self.bytesReceived = bytesReceived
        self.totalBytes = totalBytes
        self.bytesPerSecond = bytesPerSecond
    }

    public var fractionComplete: Double {
        totalBytes > 0 ? Double(bytesReceived) / Double(totalBytes) : 0
    }
}

public struct PartialDownload: Identifiable, Sendable, Hashable {
    public var id: String
    public var partialURL: URL
    public var sidecarURL: URL
    public var sourceURL: URL
    public var displayName: String
    public var hfRepo: String?
    public var hfFilename: String?
    public var totalBytes: Int64
    public var bytesOnDisk: Int64

    public init(
        id: String,
        partialURL: URL,
        sidecarURL: URL,
        sourceURL: URL,
        displayName: String,
        hfRepo: String?,
        hfFilename: String?,
        totalBytes: Int64,
        bytesOnDisk: Int64
    ) {
        self.id = id
        self.partialURL = partialURL
        self.sidecarURL = sidecarURL
        self.sourceURL = sourceURL
        self.displayName = displayName
        self.hfRepo = hfRepo
        self.hfFilename = hfFilename
        self.totalBytes = totalBytes
        self.bytesOnDisk = bytesOnDisk
    }

    public var fractionComplete: Double {
        totalBytes > 0 ? Double(bytesOnDisk) / Double(totalBytes) : 0
    }
}

public struct ModelDownloadResult: Sendable, Hashable {
    public let tempURL: URL
    public let sizeBytes: Int64
    public let sha256: String

    public init(tempURL: URL, sizeBytes: Int64, sha256: String) {
        self.tempURL = tempURL
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
    }
}

public struct ModelDownloadConfiguration: Sendable, Hashable {
    public static let defaultChunkSize: Int64 = 16 * 1_024 * 1_024
    public static let defaultParallelConnections = 12
    public static let maximumParallelConnections = 32

    public var parallelConnections: Int
    public var chunkSize: Int64
    public var requestTimeout: TimeInterval

    public init(
        parallelConnections: Int = Self.defaultParallelConnections,
        chunkSize: Int64 = Self.defaultChunkSize,
        requestTimeout: TimeInterval = 3_600
    ) {
        self.parallelConnections = min(
            max(1, parallelConnections),
            Self.maximumParallelConnections
        )
        self.chunkSize = max(1_024 * 1_024, chunkSize)
        self.requestTimeout = max(30, requestTimeout)
    }

    public static let `default` = ModelDownloadConfiguration()
}

private struct PartialSidecar: Codable, Sendable {
    var url: String
    var etag: String?
    var lastModified: String?
    var totalBytes: Int64
    var displayName: String?
    var schemaVersion: Int?
    var chunkSize: Int64?
    var doneChunks: [Int]?
}

private struct PartialDownloadState: Sendable {
    let existingSize: Int64
    let totalBytes: Int64
    let etag: String?
    let lastModified: String?
}

struct ChunkRange: Sendable, Hashable {
    let index: Int
    let start: Int64
    let end: Int64

    var length: Int64 {
        end - start + 1
    }
}

struct ChunkPlan: Sendable, Hashable {
    static let defaultChunkSize: Int64 = ModelDownloadConfiguration.defaultChunkSize

    let totalBytes: Int64
    let chunkSize: Int64
    var doneChunks: Set<Int>

    init(
        totalBytes: Int64,
        chunkSize: Int64 = defaultChunkSize,
        doneChunks: Set<Int> = []
    ) {
        self.totalBytes = totalBytes
        self.chunkSize = chunkSize
        self.doneChunks = doneChunks
    }

    var chunkCount: Int {
        Int((totalBytes + chunkSize - 1) / chunkSize)
    }

    var isComplete: Bool {
        doneChunks.count == chunkCount
    }

    func chunkRange(for index: Int) -> ChunkRange {
        let start = Int64(index) * chunkSize
        let end = min(start + chunkSize - 1, totalBytes - 1)
        return ChunkRange(index: index, start: start, end: end)
    }

    func pendingRanges() -> [ChunkRange] {
        (0..<chunkCount)
            .filter { !doneChunks.contains($0) }
            .map { chunkRange(for: $0) }
    }

    func completedBytes() -> Int64 {
        doneChunks.reduce(Int64(0)) { partial, index in
            partial + chunkRange(for: index).length
        }
    }
}

private final class RandomAccessFileWriter: @unchecked Sendable {
    private let lock = NSLock()
    private var fd: Int32

    init(url: URL) throws {
        let openedFD = url.withUnsafeFileSystemRepresentation { path in
            path.map { Darwin.open($0, O_RDWR) } ?? -1
        }
        guard openedFD >= 0 else {
            throw POSIXError(POSIXError.Code(rawValue: errno) ?? .EIO)
        }
        fd = openedFD
    }

    func write(_ data: Data, at offset: Int64) throws {
        try data.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress, rawBuffer.count > 0 else { return }

            lock.lock()
            defer { lock.unlock() }

            guard fd >= 0 else {
                throw POSIXError(.EBADF)
            }

            var bytesRemaining = rawBuffer.count
            var localOffset = 0
            while bytesRemaining > 0 {
                let written = Darwin.pwrite(
                    fd,
                    baseAddress.advanced(by: localOffset),
                    bytesRemaining,
                    off_t(offset + Int64(localOffset))
                )
                if written < 0 {
                    if errno == EINTR { continue }
                    throw POSIXError(POSIXError.Code(rawValue: errno) ?? .EIO)
                }
                guard written > 0 else {
                    throw POSIXError(.EIO)
                }
                bytesRemaining -= written
                localOffset += written
            }
        }
    }

    func close() throws {
        lock.lock()
        defer { lock.unlock() }
        guard fd >= 0 else { return }

        let result = Darwin.close(fd)
        fd = -1
        if result != 0 {
            throw POSIXError(POSIXError.Code(rawValue: errno) ?? .EIO)
        }
    }
}

private actor ChunkWorkQueue {
    private var pending: [ChunkRange]
    private var done: Set<Int>
    private let sidecarURL: URL
    private var sidecar: PartialSidecar
    private var lastPersist: Date = .distantPast

    init(
        pending: [ChunkRange],
        done: Set<Int>,
        sidecarURL: URL,
        sidecar: PartialSidecar
    ) {
        self.pending = pending
        self.done = done
        self.sidecarURL = sidecarURL
        self.sidecar = sidecar
    }

    func nextChunk() -> ChunkRange? {
        guard !pending.isEmpty else { return nil }
        return pending.removeFirst()
    }

    func markDone(_ index: Int) {
        done.insert(index)
        let now = Date()
        if now.timeIntervalSince(lastPersist) >= 1 {
            lastPersist = now
            persistSidecar()
        }
    }

    func flush() {
        persistSidecar()
    }

    var completedCount: Int {
        done.count
    }

    private func persistSidecar() {
        sidecar.doneChunks = Array(done).sorted()
        if let data = try? JSONEncoder().encode(sidecar) {
            try? data.write(to: sidecarURL, options: .atomic)
        }
    }
}

private final class ProgressTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var chunkBytes: [Int: Int64] = [:]
    private let alreadyHad: Int64
    private let totalBytes: Int64
    private let started = Date()
    private var lastEmit = Date(timeIntervalSince1970: 0)

    init(alreadyHad: Int64, totalBytes: Int64) {
        self.alreadyHad = alreadyHad
        self.totalBytes = totalBytes
    }

    func add(_ bytes: Int64, forChunk index: Int) {
        lock.lock()
        defer { lock.unlock() }
        chunkBytes[index, default: 0] += bytes
    }

    var received: Int64 {
        lock.lock()
        defer { lock.unlock() }
        return alreadyHad + chunkBytes.values.reduce(0, +)
    }

    func maybeEmit(_ onProgress: @Sendable (DownloadProgress) -> Void) {
        let progress: DownloadProgress?
        let now = Date()
        lock.lock()
        if now.timeIntervalSince(lastEmit) >= 0.25 {
            lastEmit = now
            let received = alreadyHad + chunkBytes.values.reduce(0, +)
            let elapsed = now.timeIntervalSince(started)
            let bytesPerSecond = elapsed > 0 ? Double(received - alreadyHad) / elapsed : 0
            progress = DownloadProgress(
                bytesReceived: received,
                totalBytes: totalBytes,
                bytesPerSecond: bytesPerSecond
            )
        } else {
            progress = nil
        }
        lock.unlock()

        if let progress {
            onProgress(progress)
        }
    }
}

private final class URLSessionTaskBox: @unchecked Sendable {
    private let lock = NSLock()
    private var task: URLSessionTask?
    private var cancelled = false

    func set(_ task: URLSessionTask) {
        lock.lock()
        self.task = task
        let shouldCancel = cancelled
        lock.unlock()
        if shouldCancel {
            task.cancel()
        }
    }

    func cancel() {
        lock.lock()
        cancelled = true
        let task = task
        lock.unlock()
        task?.cancel()
    }
}

private final class RangedChunkDownloadState {
    let chunk: ChunkRange
    let writer: RandomAccessFileWriter
    let tracker: ProgressTracker
    let onProgress: @Sendable (DownloadProgress) -> Void
    let continuation: CheckedContinuation<Void, Error>

    private let lock = NSLock()
    private var receivedBytes: Int64 = 0

    init(
        chunk: ChunkRange,
        writer: RandomAccessFileWriter,
        tracker: ProgressTracker,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void,
        continuation: CheckedContinuation<Void, Error>
    ) {
        self.chunk = chunk
        self.writer = writer
        self.tracker = tracker
        self.onProgress = onProgress
        self.continuation = continuation
    }

    func reserveWriteOffset(byteCount: Int) throws -> Int64 {
        let byteCount = Int64(byteCount)
        lock.lock()
        defer { lock.unlock() }

        guard receivedBytes + byteCount <= chunk.length else {
            throw ModelDownloaderError.incompleteResponse
        }

        let offset = chunk.start + receivedBytes
        receivedBytes += byteCount
        return offset
    }

    var isComplete: Bool {
        lock.lock()
        defer { lock.unlock() }
        return receivedBytes == chunk.length
    }
}

private final class RangedDownloadDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var states: [Int: RangedChunkDownloadState] = [:]

    func download(
        request: URLRequest,
        chunk: ChunkRange,
        session: URLSession,
        writer: RandomAccessFileWriter,
        tracker: ProgressTracker,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void
    ) async throws {
        let taskBox = URLSessionTaskBox()

        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            try await withCheckedThrowingContinuation { continuation in
                let task = session.dataTask(with: request)
                let state = RangedChunkDownloadState(
                    chunk: chunk,
                    writer: writer,
                    tracker: tracker,
                    onProgress: onProgress,
                    continuation: continuation
                )
                register(state, for: task)
                taskBox.set(task)
                task.resume()
            }
        } onCancel: {
            taskBox.cancel()
        }
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let http = response as? HTTPURLResponse else {
            complete(taskIdentifier: dataTask.taskIdentifier, with: .failure(ModelDownloaderError.httpStatus(-1)))
            completionHandler(.cancel)
            return
        }

        guard http.statusCode == 206 else {
            complete(taskIdentifier: dataTask.taskIdentifier, with: .failure(ModelDownloaderError.httpStatus(http.statusCode)))
            completionHandler(.cancel)
            return
        }

        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard let state = state(for: dataTask.taskIdentifier) else { return }

        do {
            let offset = try state.reserveWriteOffset(byteCount: data.count)
            try state.writer.write(data, at: offset)
            state.tracker.add(Int64(data.count), forChunk: state.chunk.index)
            state.tracker.maybeEmit(state.onProgress)
        } catch {
            complete(taskIdentifier: dataTask.taskIdentifier, with: .failure(error))
            dataTask.cancel()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let state = state(for: task.taskIdentifier) else { return }

        if let error {
            complete(taskIdentifier: task.taskIdentifier, with: .failure(error))
            return
        }

        guard state.isComplete else {
            complete(taskIdentifier: task.taskIdentifier, with: .failure(ModelDownloaderError.incompleteResponse))
            return
        }

        complete(taskIdentifier: task.taskIdentifier, with: .success(()))
    }

    private func register(_ state: RangedChunkDownloadState, for task: URLSessionTask) {
        lock.lock()
        states[task.taskIdentifier] = state
        lock.unlock()
    }

    private func state(for taskIdentifier: Int) -> RangedChunkDownloadState? {
        lock.lock()
        defer { lock.unlock() }
        return states[taskIdentifier]
    }

    private func complete(taskIdentifier: Int, with result: Result<Void, Error>) {
        lock.lock()
        let state = states.removeValue(forKey: taskIdentifier)
        lock.unlock()

        guard let state else { return }

        switch result {
        case .success:
            state.continuation.resume()
        case .failure(let error):
            state.continuation.resume(throwing: error)
        }
    }
}

public enum ModelDownloader {
    private static let currentPartialPrefix = "cllm-partial-"
    private static let userAgent = "CarbocationLocalLLM/1.0"

    public static func huggingFaceResolveURL(hfRepo: String, hfFilename: String) throws -> URL {
        let urlString = "https://huggingface.co/\(hfRepo)/resolve/main/\(hfFilename)?download=true"
        guard let url = URL(string: urlString) else {
            throw ModelDownloaderError.badURL(urlString)
        }
        return url
    }

    public static func download(
        hfRepo: String,
        hfFilename: String,
        modelsRoot: URL,
        displayName: String? = nil,
        expectedSHA256: String? = nil,
        configuration: ModelDownloadConfiguration = .default,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void = { _ in }
    ) async throws -> ModelDownloadResult {
        try await download(
            from: huggingFaceResolveURL(hfRepo: hfRepo, hfFilename: hfFilename),
            modelsRoot: modelsRoot,
            displayName: displayName,
            expectedSHA256: expectedSHA256,
            configuration: configuration,
            onProgress: onProgress
        )
    }

    /// Downloads a GGUF file to the model cache's `.partials` directory.
    /// Range-capable servers use a chunked v2 sidecar and parallel workers.
    /// Servers without Range support fall back to single-stream resume.
    public static func download(
        from url: URL,
        modelsRoot: URL,
        displayName: String? = nil,
        expectedSHA256: String? = nil,
        configuration: ModelDownloadConfiguration = .default,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void = { _ in }
    ) async throws -> ModelDownloadResult {
        let (partialURL, sidecarURL) = try partialPaths(for: url, modelsRoot: modelsRoot)

        if let (plan, sidecar) = loadChunkedState(
            partialURL: partialURL,
            sidecarURL: sidecarURL,
            sourceURL: url
        ) {
            var updatedSidecar = sidecar
            if updatedSidecar.displayName == nil, let displayName {
                updatedSidecar.displayName = displayName
                writeSidecarValue(updatedSidecar, to: sidecarURL)
            }

            modelDownloaderLog.info(
                "Resuming chunked \(url.lastPathComponent, privacy: .public) (\(plan.doneChunks.count)/\(plan.chunkCount) chunks done)"
            )

            return try await downloadParallel(
                url: url,
                partialURL: partialURL,
                sidecarURL: sidecarURL,
                plan: plan,
                sidecar: updatedSidecar,
                configuration: configuration,
                expectedSHA256: expectedSHA256,
                onProgress: onProgress
            )
        }

        let legacy = loadSingleStreamState(
            partialURL: partialURL,
            sidecarURL: sidecarURL,
            sourceURL: url
        )

        guard let probe = try await probeServer(url: url) else {
            modelDownloaderLog.info(
                "Server does not support Range for \(url.lastPathComponent, privacy: .public); using single stream"
            )
            return try await downloadSingleStream(
                url: url,
                partialURL: partialURL,
                sidecarURL: sidecarURL,
                prior: legacy,
                displayName: displayName,
                expectedSHA256: expectedSHA256,
                onProgress: onProgress
            )
        }

        let totalBytes = probe.totalBytes
        var doneChunks = Set<Int>()
        if let legacy {
            if legacy.totalBytes == totalBytes {
                let fullChunks = legacy.existingSize / configuration.chunkSize
                for index in 0..<Int(fullChunks) {
                    doneChunks.insert(index)
                }
                modelDownloaderLog.info("Credited \(fullChunks) legacy chunks from single-stream partial")
            } else {
                discardPartial(partialURL: partialURL, sidecarURL: sidecarURL)
            }
        }

        let plan = ChunkPlan(
            totalBytes: totalBytes,
            chunkSize: configuration.chunkSize,
            doneChunks: doneChunks
        )
        let sidecar = PartialSidecar(
            url: url.absoluteString,
            etag: probe.etag,
            lastModified: probe.lastModified,
            totalBytes: totalBytes,
            displayName: displayName,
            schemaVersion: 2,
            chunkSize: configuration.chunkSize,
            doneChunks: Array(doneChunks).sorted()
        )

        if !FileManager.default.fileExists(atPath: partialURL.path) {
            FileManager.default.createFile(atPath: partialURL.path, contents: nil)
        }
        let allocationHandle = try FileHandle(forWritingTo: partialURL)
        try allocationHandle.truncate(atOffset: UInt64(totalBytes))
        try allocationHandle.close()

        writeSidecarValue(sidecar, to: sidecarURL)

        modelDownloaderLog.info(
            "Starting parallel download \(url.lastPathComponent, privacy: .public) (\(totalBytes) bytes, \(plan.chunkCount) chunks, \(configuration.parallelConnections) connections)"
        )

        return try await downloadParallel(
            url: url,
            partialURL: partialURL,
            sidecarURL: sidecarURL,
            plan: plan,
            sidecar: sidecar,
            configuration: configuration,
            expectedSHA256: expectedSHA256,
            onProgress: onProgress
        )
    }

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

    private static func downloadParallel(
        url: URL,
        partialURL: URL,
        sidecarURL: URL,
        plan: ChunkPlan,
        sidecar: PartialSidecar,
        configuration: ModelDownloadConfiguration,
        expectedSHA256: String?,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void
    ) async throws -> ModelDownloadResult {
        let alreadyHad = plan.completedBytes()
        let queue = ChunkWorkQueue(
            pending: plan.pendingRanges(),
            done: plan.doneChunks,
            sidecarURL: sidecarURL,
            sidecar: sidecar
        )
        let tracker = ProgressTracker(alreadyHad: alreadyHad, totalBytes: plan.totalBytes)
        let writer = try RandomAccessFileWriter(url: partialURL)
        let validator = sidecar.etag ?? sidecar.lastModified
        let delegate = RangedDownloadDelegate()
        let session = makeURLSession(configuration: configuration, delegate: delegate)
        defer { session.invalidateAndCancel() }

        do {
            try await withThrowingTaskGroup(of: Void.self) { group in
                for _ in 0..<configuration.parallelConnections {
                    group.addTask {
                        while let chunk = await queue.nextChunk() {
                            try Task.checkCancellation()
                            try await downloadChunk(
                                chunk: chunk,
                                from: url,
                                validator: validator,
                                session: session,
                                delegate: delegate,
                                requestTimeout: configuration.requestTimeout,
                                writer: writer,
                                queue: queue,
                                tracker: tracker,
                                onProgress: onProgress
                            )
                        }
                    }
                }
                try await group.waitForAll()
            }
            try writer.close()
        } catch is CancellationError {
            try? writer.close()
            await queue.flush()
            throw ModelDownloaderError.cancelled
        } catch let error as URLError where error.code == .cancelled {
            try? writer.close()
            await queue.flush()
            throw ModelDownloaderError.cancelled
        } catch {
            try? writer.close()
            await queue.flush()
            throw error
        }

        await queue.flush()
        let completedCount = await queue.completedCount
        guard completedCount == plan.chunkCount else {
            throw ModelDownloaderError.httpStatus(-2)
        }

        let digest: String
        do {
            digest = try verifyFinalHash(at: partialURL, expected: expectedSHA256)
        } catch {
            discardPartial(partialURL: partialURL, sidecarURL: sidecarURL)
            throw error
        }

        try? FileManager.default.removeItem(at: sidecarURL)
        onProgress(DownloadProgress(
            bytesReceived: plan.totalBytes,
            totalBytes: plan.totalBytes,
            bytesPerSecond: 0
        ))

        return ModelDownloadResult(tempURL: partialURL, sizeBytes: plan.totalBytes, sha256: digest)
    }

    private static func downloadChunk(
        chunk: ChunkRange,
        from url: URL,
        validator: String?,
        session: URLSession,
        delegate: RangedDownloadDelegate,
        requestTimeout: TimeInterval,
        writer: RandomAccessFileWriter,
        queue: ChunkWorkQueue,
        tracker: ProgressTracker,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void
    ) async throws {
        var request = URLRequest(url: url)
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("bytes=\(chunk.start)-\(chunk.end)", forHTTPHeaderField: "Range")
        if let validator {
            request.setValue(validator, forHTTPHeaderField: "If-Range")
        }
        request.timeoutInterval = requestTimeout

        try Task.checkCancellation()
        try await delegate.download(
            request: request,
            chunk: chunk,
            session: session,
            writer: writer,
            tracker: tracker,
            onProgress: onProgress
        )
        await queue.markDone(chunk.index)
        tracker.maybeEmit(onProgress)
    }

    private static func makeURLSession(
        configuration: ModelDownloadConfiguration,
        delegate: URLSessionDataDelegate? = nil
    ) -> URLSession {
        let urlSessionConfiguration = URLSessionConfiguration.default
        urlSessionConfiguration.httpMaximumConnectionsPerHost = configuration.parallelConnections
        urlSessionConfiguration.requestCachePolicy = .reloadIgnoringLocalCacheData
        urlSessionConfiguration.timeoutIntervalForRequest = configuration.requestTimeout
        urlSessionConfiguration.timeoutIntervalForResource = configuration.requestTimeout
        urlSessionConfiguration.urlCache = nil
        urlSessionConfiguration.waitsForConnectivity = true

        let delegateQueue = OperationQueue()
        delegateQueue.name = "com.carbocation.CarbocationLocalLLM.ModelDownloader"
        delegateQueue.maxConcurrentOperationCount = 1

        return URLSession(
            configuration: urlSessionConfiguration,
            delegate: delegate,
            delegateQueue: delegate == nil ? nil : delegateQueue
        )
    }

    private struct ProbeResult: Sendable {
        let totalBytes: Int64
        let etag: String?
        let lastModified: String?
    }

    private static func probeServer(url: URL) async throws -> ProbeResult? {
        var request = URLRequest(url: url)
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("bytes=0-0", forHTTPHeaderField: "Range")
        request.timeoutInterval = 30

        let (_, response) = try await URLSession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 206 else {
            return nil
        }
        guard let totalBytes = parseContentRangeTotal(http), totalBytes > 0 else {
            return nil
        }

        return ProbeResult(
            totalBytes: totalBytes,
            etag: http.value(forHTTPHeaderField: "ETag"),
            lastModified: http.value(forHTTPHeaderField: "Last-Modified")
        )
    }

    private static func downloadSingleStream(
        url: URL,
        partialURL: URL,
        sidecarURL: URL,
        prior: PartialDownloadState?,
        displayName: String?,
        expectedSHA256: String?,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void
    ) async throws -> ModelDownloadResult {
        var request = URLRequest(url: url)
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        request.timeoutInterval = 3_600

        if let prior {
            request.setValue("bytes=\(prior.existingSize)-", forHTTPHeaderField: "Range")
            if let validator = prior.etag ?? prior.lastModified {
                request.setValue(validator, forHTTPHeaderField: "If-Range")
            }
            modelDownloaderLog.info(
                "Resuming single-stream \(url.lastPathComponent, privacy: .public) from byte \(prior.existingSize)"
            )
        }

        let (stream, response) = try await URLSession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ModelDownloaderError.httpStatus(-1)
        }
        guard (200...299).contains(http.statusCode) else {
            throw ModelDownloaderError.httpStatus(http.statusCode)
        }

        let isResume = http.statusCode == 206
        let existingSize: Int64
        let totalBytes: Int64
        if isResume, let prior {
            if let rangeTotal = parseContentRangeTotal(http), rangeTotal != prior.totalBytes {
                discardPartial(partialURL: partialURL, sidecarURL: sidecarURL)
                throw ModelDownloaderError.httpStatus(200)
            }
            existingSize = prior.existingSize
            totalBytes = prior.totalBytes
        } else {
            existingSize = 0
            totalBytes = http.expectedContentLength
            discardPartial(partialURL: partialURL, sidecarURL: sidecarURL)
            try FileManager.default.createDirectory(
                at: partialURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            FileManager.default.createFile(atPath: partialURL.path, contents: nil)
            if totalBytes > 0 {
                writeSingleStreamSidecar(
                    to: sidecarURL,
                    sourceURL: url,
                    totalBytes: totalBytes,
                    etag: http.value(forHTTPHeaderField: "ETag"),
                    lastModified: http.value(forHTTPHeaderField: "Last-Modified"),
                    displayName: displayName
                )
            }
        }

        var hasher = SHA256()
        if existingSize > 0 {
            try updateHasher(&hasher, withPrefixOf: partialURL, byteCount: existingSize)
        }

        let handle = try FileHandle(forWritingTo: partialURL)
        defer { try? handle.close() }
        if isResume {
            try handle.seekToEnd()
        }

        let started = Date()
        var lastEmit = Date(timeIntervalSince1970: 0)
        var received = existingSize
        var buffer: [UInt8] = []
        buffer.reserveCapacity(1 << 20)

        do {
            for try await byte in stream {
                try Task.checkCancellation()
                buffer.append(byte)
                if buffer.count >= (1 << 20) {
                    let data = Data(buffer)
                    try handle.write(contentsOf: data)
                    hasher.update(data: data)
                    received += Int64(data.count)
                    emitProgressIfNeeded(
                        received: received,
                        total: totalBytes,
                        started: started,
                        lastEmit: &lastEmit,
                        onProgress: onProgress
                    )
                    buffer.removeAll(keepingCapacity: true)
                }
            }
        } catch is CancellationError {
            throw ModelDownloaderError.cancelled
        } catch let error as URLError where error.code == .cancelled {
            throw ModelDownloaderError.cancelled
        }

        if !buffer.isEmpty {
            let data = Data(buffer)
            try handle.write(contentsOf: data)
            hasher.update(data: data)
            received += Int64(data.count)
        }

        guard totalBytes <= 0 || received == totalBytes else {
            throw ModelDownloaderError.incompleteResponse
        }

        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        if let expected = expectedSHA256?.lowercased(),
           !expected.isEmpty,
           expected != digest.lowercased() {
            discardPartial(partialURL: partialURL, sidecarURL: sidecarURL)
            throw ModelDownloaderError.hashMismatch(expected: expected, actual: digest)
        }

        try? FileManager.default.removeItem(at: sidecarURL)
        onProgress(DownloadProgress(
            bytesReceived: received,
            totalBytes: max(totalBytes, received),
            bytesPerSecond: 0
        ))
        return ModelDownloadResult(tempURL: partialURL, sizeBytes: received, sha256: digest)
    }

    private static func partialPaths(for url: URL, modelsRoot: URL) throws -> (partial: URL, sidecar: URL) {
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

    private static func loadChunkedState(
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

    private static func loadSingleStreamState(
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

    private static func writeSingleStreamSidecar(
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

    private static func writeSidecarValue(_ sidecar: PartialSidecar, to url: URL) {
        if let data = try? JSONEncoder().encode(sidecar) {
            try? data.write(to: url, options: .atomic)
        }
    }

    private static func discardPartial(partialURL: URL, sidecarURL: URL) {
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

    private static func updateHasher(_ hasher: inout SHA256, withPrefixOf url: URL, byteCount: Int64) throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var remaining = byteCount
        while remaining > 0 {
            let count = Int(min(Int64(1 << 20), remaining))
            guard let chunk = try handle.read(upToCount: count), !chunk.isEmpty else { break }
            hasher.update(data: chunk)
            remaining -= Int64(chunk.count)
        }
    }

    private static func emitProgressIfNeeded(
        received: Int64,
        total: Int64,
        started: Date,
        lastEmit: inout Date,
        onProgress: @escaping @Sendable (DownloadProgress) -> Void
    ) {
        let now = Date()
        guard now.timeIntervalSince(lastEmit) >= 0.25 else { return }
        lastEmit = now
        let elapsed = now.timeIntervalSince(started)
        let bytesPerSecond = elapsed > 0 ? Double(received) / elapsed : 0
        onProgress(DownloadProgress(bytesReceived: received, totalBytes: total, bytesPerSecond: bytesPerSecond))
    }

    private static func verifyFinalHash(at url: URL, expected: String?) throws -> String {
        let actual = try computeSHA256(at: url)
        if let expected = expected?.lowercased(),
           !expected.isEmpty,
           expected != actual.lowercased() {
            throw ModelDownloaderError.hashMismatch(expected: expected, actual: actual)
        }
        return actual
    }

    private static func computeSHA256(at url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var hasher = SHA256()
        while true {
            guard let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty else { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func parseContentRangeTotal(_ response: HTTPURLResponse) -> Int64? {
        guard let header = response.value(forHTTPHeaderField: "Content-Range"),
              let slash = header.lastIndex(of: "/")
        else { return nil }
        return Int64(header[header.index(after: slash)...].trimmingCharacters(in: .whitespaces))
    }

    private static func parseHFCoords(from url: URL) -> (repo: String, filename: String)? {
        guard url.host?.contains("huggingface.co") == true else { return nil }
        let parts = url.pathComponents.filter { $0 != "/" }
        guard parts.count >= 5, parts[2] == "resolve" || parts[2] == "blob" else { return nil }
        return ("\(parts[0])/\(parts[1])", parts.suffix(from: 4).joined(separator: "/"))
    }
}

public enum HuggingFaceURL {
    public static func parse(_ input: String) -> (repo: String, filename: String)? {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let url = URL(string: trimmed),
           let host = url.host,
           host.contains("huggingface.co") {
            let parts = url.pathComponents.filter { $0 != "/" }
            if parts.count >= 5, parts[2] == "resolve" || parts[2] == "blob" {
                let repo = "\(parts[0])/\(parts[1])"
                let filename = parts.suffix(from: 4).joined(separator: "/")
                if filename.lowercased().hasSuffix(".gguf") {
                    return (repo, filename)
                }
            }
            return nil
        }

        let parts = trimmed.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
        if parts.count >= 3, parts.last?.lowercased().hasSuffix(".gguf") == true {
            return ("\(parts[0])/\(parts[1])", parts.suffix(from: 2).joined(separator: "/"))
        }
        return nil
    }
}
