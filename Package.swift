// swift-tools-version: 5.9

import PackageDescription
import Foundation

private let llamaBuildScriptRevision = "10"
private let llamaXCFrameworkStampSchema = "carbocation.llama.xcframework.v3"

private func fileContents(atPath path: String) -> String? {
    try? String(contentsOfFile: path, encoding: .utf8)
}

private func trimmedFileContents(atPath path: String) -> String? {
    fileContents(atPath: path)?.trimmingCharacters(in: .whitespacesAndNewlines)
}

private func normalizedGitObjectID(_ value: String) -> String? {
    let candidate = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard candidate.count == 40 || candidate.count == 64,
          candidate.unicodeScalars.allSatisfy({ scalar in
              (48...57).contains(scalar.value) || (97...102).contains(scalar.value)
          }) else {
        return nil
    }
    return candidate
}

private func resolvedPath(_ path: String, relativeTo directory: String) -> String {
    if path.hasPrefix("/") {
        return URL(fileURLWithPath: path).standardizedFileURL.path
    }
    return URL(fileURLWithPath: directory, isDirectory: true)
        .appendingPathComponent(path)
        .standardizedFileURL.path
}

private func gitDirectory(forWorkingTree workingTree: String) -> String? {
    let dotGitPath = "\(workingTree)/.git"
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: dotGitPath, isDirectory: &isDirectory) else {
        return nil
    }
    if isDirectory.boolValue {
        return dotGitPath
    }

    guard let pointer = trimmedFileContents(atPath: dotGitPath),
          pointer.hasPrefix("gitdir:") else {
        return nil
    }
    let gitDirectoryPath = String(pointer.dropFirst("gitdir:".count))
        .trimmingCharacters(in: .whitespacesAndNewlines)
    return resolvedPath(gitDirectoryPath, relativeTo: workingTree)
}

private func gitCommonDirectory(forGitDirectory gitDirectory: String) -> String {
    guard let commonDirectoryPath = trimmedFileContents(atPath: "\(gitDirectory)/commondir"),
          !commonDirectoryPath.isEmpty else {
        return gitDirectory
    }
    return resolvedPath(commonDirectoryPath, relativeTo: gitDirectory)
}

private func resolvedGitReference(
    _ reference: String,
    gitDirectory: String,
    commonDirectory: String
) -> String? {
    for directory in [gitDirectory, commonDirectory] {
        if let value = trimmedFileContents(atPath: "\(directory)/\(reference)"),
           let objectID = normalizedGitObjectID(value) {
            return objectID
        }
    }

    guard let packedReferences = trimmedFileContents(atPath: "\(commonDirectory)/packed-refs") else {
        return nil
    }
    for line in packedReferences.split(whereSeparator: \.isNewline) {
        if line.hasPrefix("#") || line.hasPrefix("^") {
            continue
        }
        let fields = line.split(separator: " ", maxSplits: 1)
        if fields.count == 2,
           fields[1] == Substring(reference),
           let objectID = normalizedGitObjectID(String(fields[0])) {
            return objectID
        }
    }
    return nil
}

private func checkedOutGitCommit(atWorkingTree workingTree: String) -> String? {
    guard let gitDirectory = gitDirectory(forWorkingTree: workingTree),
          let head = trimmedFileContents(atPath: "\(gitDirectory)/HEAD") else {
        return nil
    }
    if let detachedCommit = normalizedGitObjectID(head) {
        return detachedCommit
    }
    guard head.hasPrefix("ref:") else {
        return nil
    }
    let reference = String(head.dropFirst("ref:".count))
        .trimmingCharacters(in: .whitespacesAndNewlines)
    return resolvedGitReference(
        reference,
        gitDirectory: gitDirectory,
        commonDirectory: gitCommonDirectory(forGitDirectory: gitDirectory)
    )
}

private struct LlamaStageStamp {
    let scriptRevision: String
    let cmakeVersion: String
    let commit: String
    let platform: String
    let sdkPath: String
    let architectures: String
    let configuration: String
    let deploymentTarget: String

    init?(_ value: String) {
        let fields = value.split(separator: "|", omittingEmptySubsequences: false)
        guard fields.count == 8,
              !fields[1].isEmpty,
              let commit = normalizedGitObjectID(String(fields[2])),
              !fields[4].isEmpty else {
            return nil
        }
        scriptRevision = String(fields[0])
        cmakeVersion = String(fields[1])
        self.commit = commit
        platform = String(fields[3])
        sdkPath = String(fields[4])
        architectures = String(fields[5])
        configuration = String(fields[6])
        deploymentTarget = String(fields[7])
    }

    func certifies(
        commit expectedCommit: String,
        platform expectedPlatform: String,
        architectures expectedArchitectures: String,
        deploymentTarget expectedDeploymentTarget: String
    ) -> Bool {
        scriptRevision == llamaBuildScriptRevision
            && commit == expectedCommit
            && platform == expectedPlatform
            && architectures == expectedArchitectures
            && configuration == "Release"
            && deploymentTarget == expectedDeploymentTarget
    }
}

private func compositeXCFrameworkStampIsCurrent(
    atPath path: String,
    checkedOutCommit: String
) -> Bool {
    guard var contents = fileContents(atPath: path),
          !contents.contains("\r") else {
        return false
    }
    if contents.hasSuffix("\n") {
        contents.removeLast()
    }
    guard !contents.hasSuffix("\n") else {
        return false
    }
    let lines = contents.components(separatedBy: "\n")
    guard lines.count == 4,
          lines[0] == "schema=\(llamaXCFrameworkStampSchema)" else {
        return false
    }

    func stage(onLine index: Int, key: String) -> LlamaStageStamp? {
        let prefix = "\(key)="
        guard lines[index].hasPrefix(prefix) else {
            return nil
        }
        return LlamaStageStamp(String(lines[index].dropFirst(prefix.count)))
    }

    guard let macOS = stage(onLine: 1, key: "macos"),
          let iOS = stage(onLine: 2, key: "ios"),
          let iOSSimulator = stage(onLine: 3, key: "ios-simulator") else {
        return false
    }
    return macOS.certifies(
        commit: checkedOutCommit,
        platform: "macos",
        architectures: "arm64",
        deploymentTarget: "14.0"
    ) && iOS.certifies(
        commit: checkedOutCommit,
        platform: "ios",
        architectures: "arm64",
        deploymentTarget: "17.0"
    ) && iOSSimulator.certifies(
        commit: checkedOutCommit,
        platform: "ios-simulator",
        architectures: "arm64",
        deploymentTarget: "17.0"
    )
}

private func sourceArchiveIsCurrent(
    archivePath: String,
    stampPath: String,
    checkedOutCommit: String
) -> Bool {
    guard FileManager.default.fileExists(atPath: archivePath),
          let stampContents = fileContents(atPath: stampPath),
          !stampContents.contains("\n"),
          !stampContents.contains("\r"),
          let stamp = LlamaStageStamp(stampContents) else {
        return false
    }
    return stamp.certifies(
        commit: checkedOutCommit,
        platform: "macos",
        architectures: "arm64",
        deploymentTarget: "14.0"
    )
}

let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let llamaCombinedLibrary = "\(packageRoot)/Vendor/llama-artifacts/current/lib/libllama-combined.a"
let llamaSourceArtifactStamp = "\(packageRoot)/Vendor/llama-artifacts/current/.stamp"
let llamaWorkingTree = "\(packageRoot)/Vendor/llama.cpp"
let llamaCheckedOutCommit = checkedOutGitCommit(atWorkingTree: llamaWorkingTree)
let localLlamaBinaryArtifactPath = "Vendor/llama-artifacts/release/llama.xcframework"
let localLlamaBinaryArtifactAbsolutePath = "\(packageRoot)/\(localLlamaBinaryArtifactPath)"
let localLlamaBinaryArtifactIsCurrent = llamaCheckedOutCommit.map { commit in
    FileManager.default.fileExists(atPath: localLlamaBinaryArtifactAbsolutePath)
        && compositeXCFrameworkStampIsCurrent(
            atPath: "\(localLlamaBinaryArtifactAbsolutePath)/.stamp",
            checkedOutCommit: commit
        )
} ?? false
let localLlamaSourceArtifactIsCurrent = llamaCheckedOutCommit.map { commit in
    sourceArchiveIsCurrent(
        archivePath: llamaCombinedLibrary,
        stampPath: llamaSourceArtifactStamp,
        checkedOutCommit: commit
    )
} ?? false
let llamaBinaryArtifactURL = "https://github.com/carbocation/CarbocationLocalLLM/releases/download/v0.55.0/llama.xcframework.zip"
let llamaBinaryArtifactChecksum = "2110fe18853efcf55fdb25136a27e60608887e795b3900460ea0e304a54bb79c"
let llamaBinaryArtifactPath = ProcessInfo.processInfo.environment["CARBOCATION_LOCAL_LLM_BINARY_ARTIFACT_PATH"] ?? ""
let forceSourceLlama = ProcessInfo.processInfo.environment["CARBOCATION_LOCAL_LLM_FORCE_SOURCE_LLAMA"] == "1"

let llamaTarget: Target
let llamaUnsafeLinkerSettings: [LinkerSetting]

if forceSourceLlama {
    guard localLlamaSourceArtifactIsCurrent else {
        fatalError(
            "CARBOCATION_LOCAL_LLM_FORCE_SOURCE_LLAMA requires a current arm64 Release archive. "
                + "Run Scripts/build-llama-macos.sh and retry."
        )
    }
    llamaTarget = .systemLibrary(
        name: "llama",
        path: "Sources/llama"
    )
    llamaUnsafeLinkerSettings = [.unsafeFlags([llamaCombinedLibrary])]
} else if !llamaBinaryArtifactPath.isEmpty {
    llamaTarget = .binaryTarget(
        name: "llama",
        path: llamaBinaryArtifactPath
    )
    llamaUnsafeLinkerSettings = []
} else if !llamaBinaryArtifactURL.isEmpty && !llamaBinaryArtifactChecksum.isEmpty {
    llamaTarget = .binaryTarget(
        name: "llama",
        url: llamaBinaryArtifactURL,
        checksum: llamaBinaryArtifactChecksum
    )
    llamaUnsafeLinkerSettings = []
} else if localLlamaBinaryArtifactIsCurrent {
    llamaTarget = .binaryTarget(
        name: "llama",
        path: localLlamaBinaryArtifactPath
    )
    llamaUnsafeLinkerSettings = []
} else if localLlamaSourceArtifactIsCurrent {
    llamaTarget = .systemLibrary(
        name: "llama",
        path: "Sources/llama"
    )
    llamaUnsafeLinkerSettings = [.unsafeFlags([llamaCombinedLibrary])]
} else {
    fatalError(
        "No current llama artifact is available: the automatic local XCFramework and source archive "
            + "are missing, stale, or do not match Vendor/llama.cpp. Run Scripts/build-llama-xcframework.sh "
            + "or Scripts/build-llama-macos.sh, then retry. An explicit "
            + "CARBOCATION_LOCAL_LLM_BINARY_ARTIFACT_PATH remains an intentional override."
    )
}

let package = Package(
    name: "CarbocationLocalLLM",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "CarbocationLocalLLM",
            targets: ["CarbocationLocalLLM"]
        ),
        .library(
            name: "CarbocationLlamaRuntime",
            targets: ["CarbocationLlamaRuntime"]
        ),
        .library(
            name: "CarbocationLocalLLMRuntime",
            targets: ["CarbocationLocalLLMRuntime"]
        ),
        .library(
            name: "CarbocationLocalLLMTools",
            targets: ["CarbocationLocalLLMTools"]
        ),
        .library(
            name: "CarbocationLocalLLMUI",
            targets: ["CarbocationLocalLLMUI"]
        ),
        .library(
            name: "CarbocationLocalLLMRuntimeUI",
            targets: ["CarbocationLocalLLMRuntimeUI"]
        ),
        .executable(
            name: "CLLMMTPReproCommand",
            targets: ["CLLMMTPReproCommand"]
        ),
        .executable(
            name: "CLLMBenchmarkCommand",
            targets: ["CLLMBenchmarkCommand"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/scinfu/SwiftSoup.git", exact: "2.9.6")
    ],
    targets: [
        .target(
            name: "CarbocationLocalLLM"
        ),
        llamaTarget,
        .target(
            name: "CarbocationLlamaCommonBridge",
            dependencies: [
                "llama"
            ],
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("../../Vendor/llama.cpp/common"),
                .headerSearchPath("../../Vendor/llama.cpp/include"),
                .headerSearchPath("../../Vendor/llama.cpp/ggml/include")
            ],
            cxxSettings: [
                .headerSearchPath("../../Vendor/llama.cpp/common"),
                .headerSearchPath("../../Vendor/llama.cpp/include"),
                .headerSearchPath("../../Vendor/llama.cpp/ggml/include"),
                .headerSearchPath("../../Vendor/llama.cpp/ggml/src"),
                .headerSearchPath("../../Vendor/llama.cpp/src"),
                .headerSearchPath("../../Vendor/llama.cpp/vendor")
            ]
        ),
        .target(
            name: "CarbocationLlamaMTMDBridge",
            dependencies: ["llama"],
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("../../Vendor/llama.cpp/tools/mtmd"),
                .headerSearchPath("../../Vendor/llama.cpp/include"),
                .headerSearchPath("../../Vendor/llama.cpp/ggml/include")
            ]
        ),
        .target(
            name: "CarbocationLlamaRuntime",
            dependencies: [
                "CarbocationLocalLLM",
                "CarbocationLlamaCommonBridge",
                "CarbocationLlamaMTMDBridge",
                "llama"
            ],
            linkerSettings: llamaUnsafeLinkerSettings + [
                .linkedLibrary("c++"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Foundation")
            ]
        ),
        .target(
            name: "CarbocationAppleIntelligenceRuntime",
            dependencies: ["CarbocationLocalLLM"]
        ),
        .target(
            name: "CarbocationLocalLLMRuntime",
            dependencies: [
                "CarbocationLocalLLM",
                "CarbocationLlamaRuntime",
                "CarbocationAppleIntelligenceRuntime"
            ]
        ),
        .target(
            name: "CarbocationLocalLLMTools",
            dependencies: [
                "CarbocationLocalLLM",
                "SwiftSoup"
            ]
        ),
        .target(
            name: "CarbocationLocalLLMUI",
            dependencies: ["CarbocationLocalLLM"]
        ),
        .target(
            name: "CarbocationLocalLLMRuntimeUI",
            dependencies: [
                "CarbocationLocalLLM",
                "CarbocationLocalLLMRuntime",
                "CarbocationLocalLLMUI"
            ]
        ),
        .executableTarget(
            name: "CLLMMTPReproCommand",
            dependencies: [
                "CarbocationLocalLLM",
                "CarbocationLlamaRuntime"
            ]
        ),
        .executableTarget(
            name: "CLLMBenchmarkCommand",
            dependencies: [
                "CarbocationLocalLLM",
                "CarbocationLlamaRuntime"
            ]
        ),
        .testTarget(
            name: "CarbocationLocalLLMTests",
            dependencies: [
                "CarbocationLocalLLM"
            ]
        ),
        .testTarget(
            name: "CarbocationLlamaRuntimeTests",
            dependencies: ["CarbocationLlamaRuntime"]
        ),
        .testTarget(
            name: "CarbocationLlamaCommonBridgeTests",
            dependencies: [
                "CarbocationLlamaCommonBridge",
                "llama"
            ]
        ),
        .testTarget(
            name: "CarbocationAppleIntelligenceRuntimeTests",
            dependencies: ["CarbocationAppleIntelligenceRuntime"]
        ),
        .testTarget(
            name: "CarbocationLocalLLMRuntimeTests",
            dependencies: ["CarbocationLocalLLMRuntime"]
        ),
        .testTarget(
            name: "CarbocationLocalLLMToolsTests",
            dependencies: [
                "CarbocationLocalLLM",
                "CarbocationLocalLLMTools"
            ]
        ),
        .testTarget(
            name: "CarbocationLocalLLMUITests",
            dependencies: ["CarbocationLocalLLMUI"]
        ),
        .testTarget(
            name: "CarbocationLocalLLMRuntimeUITests",
            dependencies: ["CarbocationLocalLLMRuntimeUI"]
        )
    ],
    cxxLanguageStandard: .cxx17
)
