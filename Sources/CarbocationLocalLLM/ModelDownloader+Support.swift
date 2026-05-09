import Darwin
import Foundation

struct PartialSidecar: Codable, Sendable {
    var url: String
    var etag: String?
    var lastModified: String?
    var totalBytes: Int64
    var displayName: String?
    var schemaVersion: Int?
    var chunkSize: Int64?
    var doneChunks: [Int]?
}

struct PartialDownloadState: Sendable {
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

final class RandomAccessFileWriter: @unchecked Sendable {
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

actor ChunkWorkQueue {
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

final class ProgressTracker: @unchecked Sendable {
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

final class AggregateDownloadProgressTracker: @unchecked Sendable {
    private let lock = NSLock()
    private let expectedTotals: [String: Int64]
    private var received: [String: Int64] = [:]
    private var totals: [String: Int64] = [:]
    private var speeds: [String: Double] = [:]

    init(artifacts: [HuggingFaceResolvedArtifact]) {
        var expected: [String: Int64] = [:]
        for artifact in artifacts {
            expected[artifact.path] = max(0, artifact.sizeBytes)
        }
        self.expectedTotals = expected
        self.totals = expected
    }

    func update(path: String, progress: DownloadProgress) -> DownloadProgress {
        lock.lock()
        received[path] = progress.bytesReceived
        totals[path] = max(progress.totalBytes, expectedTotals[path] ?? 0)
        speeds[path] = progress.bytesPerSecond
        let aggregateReceived = received.values.reduce(Int64(0), +)
        let aggregateTotal = totals.values.reduce(Int64(0), +)
        let aggregateSpeed = speeds.values.reduce(0, +)
        lock.unlock()
        return DownloadProgress(
            bytesReceived: aggregateReceived,
            totalBytes: aggregateTotal,
            bytesPerSecond: aggregateSpeed
        )
    }
}

final class URLSessionTaskBox: @unchecked Sendable {
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

final class RangedChunkDownloadState {
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

final class RangedDownloadDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
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
