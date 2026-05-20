// swift-tools-version: 5.9

import PackageDescription
import Foundation

let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let llamaCombinedLibrary = "\(packageRoot)/Vendor/llama-artifacts/current/lib/libllama-combined.a"
let localLlamaBinaryArtifactPath = "Vendor/llama-artifacts/release/llama.xcframework"
let localLlamaBinaryArtifactExists = FileManager.default.fileExists(
    atPath: "\(packageRoot)/\(localLlamaBinaryArtifactPath)"
)
let llamaBinaryArtifactURL = "https://github.com/carbocation/CarbocationLocalLLM/releases/download/v0.29.0/llama.xcframework.zip"
let llamaBinaryArtifactChecksum = "42d74a5acff2523fbe49c312e03a5de28487adc009bedd3ebdab63923ffc87ab"
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
        )
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.0.0"),
        .package(url: "https://github.com/scinfu/SwiftSoup.git", exact: "2.9.6")
    ],
    targets: [
        .target(
            name: "CarbocationLocalLLM"
        ),
        llamaTarget,
        .target(
            name: "LlamaCppCommon",
            dependencies: [
                "llama"
            ],
            path: "Vendor/llama.cpp/common",
            sources: [
                "common.cpp",
                "fit.cpp",
                "log.cpp",
                "ngram-cache.cpp",
                "ngram-map.cpp",
                "ngram-mod.cpp",
                "sampling.cpp",
                "speculative.cpp",
                "unicode.cpp"
            ],
            publicHeadersPath: ".",
            cxxSettings: [
                .headerSearchPath("."),
                .headerSearchPath("../include"),
                .headerSearchPath("../ggml/include"),
                .headerSearchPath("../ggml/src"),
                .headerSearchPath("../src"),
                .headerSearchPath("../vendor")
            ]
        ),
        .target(
            name: "CarbocationLlamaCommonBridge",
            dependencies: [
                "llama",
                "LlamaCppCommon"
            ],
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("../../Vendor/llama.cpp/include"),
                .headerSearchPath("../../Vendor/llama.cpp/common")
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
            name: "CarbocationLlamaRuntime",
            dependencies: [
                "CarbocationLocalLLM",
                "CarbocationLlamaCommonBridge",
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
