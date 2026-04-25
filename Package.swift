// swift-tools-version: 5.9

import PackageDescription
import Foundation

let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let llamaCombinedLibrary = "\(packageRoot)/Vendor/llama-artifacts/current/lib/libllama-combined.a"
let llamaBinaryArtifactURL = "https://github.com/carbocation/CarbocationLocalLLM/releases/download/v0.1.0/llama.xcframework.zip"
let llamaBinaryArtifactChecksum = "66bb2ab17fbf453b89a280e9d753c12febae4c2819b88f7d55f4e3dfe9ff0bb5"
let llamaBinaryArtifactPath = ProcessInfo.processInfo.environment["CARBOCATION_LOCAL_LLM_BINARY_ARTIFACT_PATH"] ?? ""
let forceSourceLlama = ProcessInfo.processInfo.environment["CARBOCATION_LOCAL_LLM_FORCE_SOURCE_LLAMA"] == "1"

let llamaTarget: Target
let llamaUnsafeLinkerSettings: [LinkerSetting]

if forceSourceLlama {
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
} else {
    llamaTarget = .systemLibrary(
        name: "llama",
        path: "Sources/llama"
    )
    llamaUnsafeLinkerSettings = [.unsafeFlags([llamaCombinedLibrary])]
}

let package = Package(
    name: "CarbocationLocalLLM",
    platforms: [
        .macOS(.v14)
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
            name: "CarbocationLocalLLMUI",
            targets: ["CarbocationLocalLLMUI"]
        ),
        .executable(
            name: "CLLMSmoke",
            targets: ["CLLMSmoke"]
        )
    ],
    targets: [
        .target(
            name: "CarbocationLocalLLM"
        ),
        llamaTarget,
        .target(
            name: "CarbocationLlamaRuntime",
            dependencies: [
                "CarbocationLocalLLM",
                "llama"
            ],
            linkerSettings: llamaUnsafeLinkerSettings + [
                .linkedLibrary("c++"),
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Foundation")
            ]
        ),
        .target(
            name: "CarbocationLocalLLMUI",
            dependencies: ["CarbocationLocalLLM"]
        ),
        .executableTarget(
            name: "CLLMSmoke",
            dependencies: [
                "CarbocationLocalLLMUI",
                "CarbocationLlamaRuntime"
            ]
        ),
        .testTarget(
            name: "CarbocationLocalLLMTests",
            dependencies: ["CarbocationLocalLLM"]
        ),
        .testTarget(
            name: "CarbocationLlamaRuntimeTests",
            dependencies: ["CarbocationLlamaRuntime"]
        )
    ]
)
