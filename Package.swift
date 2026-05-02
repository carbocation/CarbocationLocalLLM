// swift-tools-version: 5.9

import PackageDescription
import Foundation

let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let llamaCombinedLibrary = "\(packageRoot)/Vendor/llama-artifacts/current/lib/libllama-combined.a"
let localLlamaBinaryArtifactPath = "Vendor/llama-artifacts/release/llama.xcframework"
let localLlamaBinaryArtifactExists = FileManager.default.fileExists(
    atPath: "\(packageRoot)/\(localLlamaBinaryArtifactPath)"
)
let llamaBinaryArtifactURL = "https://github.com/carbocation/CarbocationLocalLLM/releases/download/v0.6.0/llama.xcframework.zip"
let llamaBinaryArtifactChecksum = "1d9eb3f0a96a2f6938338e79a9652dc4075ca57d9cb5d8a0c07bae1659730217"
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
} else if localLlamaBinaryArtifactExists {
    llamaTarget = .binaryTarget(
        name: "llama",
        path: localLlamaBinaryArtifactPath
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
            name: "CarbocationLocalLLMUI",
            targets: ["CarbocationLocalLLMUI"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.0.0")
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
                "llama",
                .product(name: "Jinja", package: "swift-jinja")
            ],
            linkerSettings: llamaUnsafeLinkerSettings + [
                .linkedLibrary("c++"),
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
            name: "CarbocationLocalLLMUI",
            dependencies: ["CarbocationLocalLLM"]
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
            name: "CarbocationAppleIntelligenceRuntimeTests",
            dependencies: ["CarbocationAppleIntelligenceRuntime"]
        ),
        .testTarget(
            name: "CarbocationLocalLLMRuntimeTests",
            dependencies: ["CarbocationLocalLLMRuntime"]
        ),
        .testTarget(
            name: "CarbocationLocalLLMUITests",
            dependencies: ["CarbocationLocalLLMUI"]
        )
    ]
)
