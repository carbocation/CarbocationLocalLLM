// swift-tools-version: 5.9

import PackageDescription
import Foundation

let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let llamaCombinedLibrary = "\(packageRoot)/Vendor/llama-artifacts/current/lib/libllama-combined.a"

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
        .systemLibrary(
            name: "llama",
            path: "Sources/llama"
        ),
        .target(
            name: "CarbocationLlamaRuntime",
            dependencies: [
                "CarbocationLocalLLM",
                "llama"
            ],
            linkerSettings: [
                .unsafeFlags([llamaCombinedLibrary]),
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
