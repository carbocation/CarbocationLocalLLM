# CarbocationLocalLLM

CarbocationLocalLLM gives macOS and iOS apps local, on-device text generation — llama.cpp or Apple Intelligence, behind one Swift API. It handles model storage, downloads, model probing, provider selection, and ships a SwiftUI model-management pane. You bring the prompts, generation policy, and product UX.

The package owns shared LLM infrastructure: neutral model storage, GGUF model management, llama.cpp runtime access, Apple Intelligence integration, a unified runtime facade, generation options, JSON helpers, and SwiftUI surfaces. Host apps own product behavior: app-specific prompts, grammars, settings policy, onboarding, command parsing, and post-generation cleanup.

> **Who is this for?** If you are wiring this package into an app, start with [Quick Start](#quick-start). If you are working on the package itself, jump to [For Package Developers](#for-package-developers).

## Contents

- [Quick Start](#quick-start)
- [Integration Guide](#integration-guide)
- [Requirements](#requirements)
- [License](#license)
- [Reference](#reference)
- [For Package Developers](#for-package-developers)

## Quick Start

For app integration, use a release tag. Release tags ship a prebuilt `llama.xcframework` with macOS, iOS device, and iOS simulator slices, so your app does not need this repo's submodules, build scripts, or local artifact environment variables.

Do not point a shipping app at `main`. `main` is source-development friendly and may require a locally built llama artifact for GGUF inference.

### Supported platforms

| Provider | Supported OS | Notes |
| --- | --- | --- |
| GGUF / llama.cpp | macOS 14+, iOS/iPadOS 17+ | Uses the release `llama.xcframework` when you pin a published tag. |
| Apple Intelligence | macOS 26+, iOS/iPadOS 26+ | Optional. Runtime-gated by SDK, OS, device support, and the user's Settings state. |

### Add the package

In Xcode, use `File > Add Package Dependencies…` with this URL:

```text
https://github.com/carbocation/CarbocationLocalLLM.git
```

Choose `Exact Version` `0.3.0` or the current release tag you want to ship.

For a SwiftPM host package:

```swift
dependencies: [
    .package(
        url: "https://github.com/carbocation/CarbocationLocalLLM.git",
        exact: "0.3.0"
    )
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "CarbocationLocalLLM", package: "CarbocationLocalLLM"),
            .product(name: "CarbocationLocalLLMRuntime", package: "CarbocationLocalLLM"),
            .product(name: "CarbocationLocalLLMUI", package: "CarbocationLocalLLM")
        ]
    )
]
```

### Pick your products

Most apps add these three package products to the app target:

```swift
import CarbocationLocalLLM         // core types
import CarbocationLocalLLMRuntime  // unified llama.cpp + Apple Intelligence engine
import CarbocationLocalLLMUI       // built-in model library picker
```

See [Products](#products) for the full list.

### Minimal working integration

Build a model library once at startup, refresh its cached state, then load a saved selection and generate:

```swift
@MainActor
func makeLibrary() -> ModelLibrary {
    ModelLibrary(
        root: ModelStorage.modelsDirectory(appSupportFolderName: "YourApp"),
        contextLengthProbe: { url in
            LocalLLMEngine.probeTrainingContext(at: url)
        }
    )
}

func generate(
    prompt: String,
    using library: ModelLibrary,
    storedSelection: String
) async throws -> String {
    guard let plan = await LocalLLMEngine.loadPlan(from: storedSelection, in: library) else {
        throw LocalLLMEngineError.invalidSelection(storedSelection)
    }

    let loaded = try await LocalLLMEngine.shared.load(
        selection: plan.selection,
        from: library,
        requestedContext: plan.requestedContext
    )

    let response = try await LocalLLMEngine.shared.generate(
        system: "You are a helpful assistant.",
        prompt: prompt,
        options: GenerationOptions(maxOutputTokens: 512)
    ) { _ in }

    return response
}
```

`storedSelection` is whatever you persisted in `@AppStorage` or `UserDefaults`. See [Pick and persist a provider](#pick-and-persist-a-provider) for how to produce that value.

## Integration Guide

### Set up a model library

Each app creates one `ModelLibrary`. The default helper writes models into a shared App Group (`group.com.carbocation.shared`) and falls back to your app's Application Support folder if the group is unavailable.

```swift
@MainActor
func makeLibrary() -> ModelLibrary {
    ModelLibrary(
        root: ModelStorage.modelsDirectory(appSupportFolderName: "YourApp"),
        contextLengthProbe: { url in
            LocalLLMEngine.probeTrainingContext(at: url)
        }
    )
}
```

The `contextLengthProbe` lets imported GGUF files record their training context up front, so the picker and engine can size contexts correctly.
Call `await library.refresh()` before reading `library.models` directly. For persisted selections, prefer `LocalLLMEngine.loadPlan(from:in:)`, which refreshes before resolving installed models. The bundled picker refreshes the library for you.

To share installed GGUF models across multiple of your apps, give them the same App Group entitlement and pass that identifier explicitly:

```swift
let modelsRoot = ModelStorage.modelsDirectory(
    sharedGroupIdentifier: "group.com.example.shared",
    appSupportFolderName: "YourApp"
)
let library = ModelLibrary(root: modelsRoot, contextLengthProbe: { url in
    LocalLLMEngine.probeTrainingContext(at: url)
})
```

For a fully custom location, bypass the helper:

```swift
let library = ModelLibrary(root: customModelsRoot)
```

Installed GGUF models live in UUID directories under the models root, each with a `metadata.json` and a `.gguf` weight file.

### Pick and persist a provider

Persist `LLMModelSelection.storageValue`, not a model filename. Installed GGUF models use UUID storage values; system providers use stable strings such as `system.apple-intelligence`.

```swift
let systemModels = LocalLLMEngine.availableSystemModels()
await library.refresh()
let installed = await MainActor.run { library.models.first }

let selection: LLMModelSelection
if let systemModel = systemModels.first {
    selection = systemModel.selection
} else if let model = installed {
    selection = .installed(model.id)
} else {
    throw LocalLLMEngineError.invalidSelection("No LLM provider is available.")
}

let valueToPersist = selection.storageValue
```

Restore later with:

```swift
guard let plan = await LocalLLMEngine.loadPlan(from: valueFromPreferences, in: library) else {
    // Show the picker or clear the stale preference.
    return
}
```

`LocalLLMEngine.availableSystemModels()` returns only system models that should be visible on this machine. On unsupported devices or builds without Foundation Models, Apple Intelligence is omitted from the picker entirely.
Use `LocalLLMEngine.loadPlan` for UI state such as "has usable model", settings labels, and context summaries. Direct model list displays can still use `await library.refresh()` followed by cached `library.models` reads.

### Generate text

```swift
guard let plan = await LocalLLMEngine.loadPlan(from: valueFromPreferences, in: library) else {
    // Show the picker or clear the stale preference.
    return
}

let loaded = try await LocalLLMEngine.shared.load(
    selection: plan.selection,
    from: library,
    requestedContext: plan.requestedContext
)

let response = try await LocalLLMEngine.shared.generate(
    system: "You are a helpful assistant.",
    prompt: userPrompt,
    options: GenerationOptions(maxOutputTokens: 512)
) { event in
    // Stream tokens to the UI here.
}
```

Apple Intelligence does not accept every llama option — GBNF grammars are rejected for it, and token counts are estimates rather than exact. Check `LocalLLMEngine.loadPlan(from:in:)`, `LocalLLMEngine.capabilities(for:in:)`, or the `LocalLLMLoadedModelInfo` returned by `load` before exposing provider-specific controls.

### Preflight a request

After loading a provider, call `preflight` to answer whether a specific prompt fits the context that actually initialized:

```swift
var options = GenerationOptions(maxOutputTokens: 512)
let preflight = try await LocalLLMEngine.shared.preflight(
    system: "You are a helpful assistant.",
    prompt: userPrompt,
    options: options
)

if !preflight.canGenerate {
    // Disable Run, trim input, or split the request before calling generate.
    return
}

options.maxOutputTokens = preflight.effectiveMaxOutputTokens
```

`loadedContextSize` is the usable context for the currently loaded model/provider. `modelTrainingContextSize` is the model's advertised training context when known. A GGUF model may advertise 256k context but be loaded at 16k because of app policy, default caps, or device limits; preflight budgets against the loaded 16k context. Preflight does not perform calibration probing.

GGUF preflight uses exact tokenization through the loaded llama vocabulary and chat template. Apple Intelligence preflight uses the same coarse token estimator as Apple Intelligence generation stats, so `usesExactTokenCounts` is `false`.

### Calibrate GGUF context

GGUF context calibration is explicit and user-triggered. It probes power-of-two context tiers, records the highest tier that can initialize on the current device, and saves the result in the shared App Group settings. The cache key includes a generated per-device id, the installed model fingerprint, and the llama runtime configuration, so the same user can share model files across apps while keeping separate limits for different devices and runtime knobs.

```swift
let record = try await LocalLLMEngine.calibrateContext(
    for: installedModel,
    in: library
) { progress in
    // Update a cancelable progress UI with progress.currentContext,
    // progress.lastSuccessfulContext, and progress.fractionCompleted.
}

print(record.maximumSupportedContext)
```

`LocalLLMEngine.loadPlan(from:in:)` uses a matching calibration record in automatic context mode. Manual mode still uses the explicit user value. If there is no matching calibration, automatic mode keeps the conservative defaults: 4,096 tokens on iOS and 16,384 tokens on desktop, bounded by the model training context when known.

### Keep a session live between queries

`LocalLLMEngine.shared.generate(system:prompt:options:)` is still the right default for one-shot and extraction workflows. It keeps a loaded llama model/context available internally, but the API remains stateless across requests.

For workflows that should explicitly keep provider state live, create a `LocalLLMSession` and hold it for the interaction:

```swift
let session = try await LocalLLMSession(
    selection: plan.selection,
    system: "You are a helpful assistant.",
    from: library,
    requestedContext: plan.requestedContext
)

let first = try await session.generate(
    prompt: userPrompt,
    options: GenerationOptions(maxOutputTokens: 512)
) { _ in }

let budget = try await session.preflight(
    prompt: "Now summarize that in one sentence.",
    options: GenerationOptions(maxOutputTokens: 128)
)

let followUp = try await session.generate(
    prompt: "Now summarize that in one sentence.",
    options: GenerationOptions(maxOutputTokens: budget.effectiveMaxOutputTokens)
) { _ in }

await session.unload()
```

For installed GGUF models, this owns a dedicated loaded llama engine for the session. For Apple Intelligence, this uses a reusable Foundation Models session instead of creating one per request.

### Enable thinking templates

`GenerationOptions.enableThinking` defaults to `false`, preserving extraction-safe behavior for models whose chat templates support reasoning channels. Apps that want model-native thinking behavior can opt in per request:

```swift
let response = try await LocalLLMEngine.shared.generate(
    system: "Solve carefully, then return the final answer.",
    prompt: userPrompt,
    options: GenerationOptions(maxOutputTokens: 1024, enableThinking: true)
) { _ in }
```

The option is passed through to embedded Jinja chat templates as `enable_thinking`. Templates that do not use that variable ignore it.

### Constrain JSON output

For JSON extraction, branch on `loaded.supportsGrammar`:

```swift
let options: GenerationOptions
let systemPrompt: String

if loaded.supportsGrammar {
    options = GenerationOptions.extractionSafe.with(grammar: jsonGrammar)
    systemPrompt = "Return only JSON matching the requested schema."
} else {
    options = GenerationOptions(maxOutputTokens: 512, stopAtBalancedJSON: true)
    systemPrompt = "Return only JSON matching the requested schema. Do not include prose."
}

let response = try await LocalLLMEngine.shared.generate(
    system: systemPrompt,
    prompt: userPrompt,
    options: options
) { _ in }
```

GGUF models use grammar-constrained generation. Apple Intelligence omits the grammar, leans on prompt guidance, and uses `stopAtBalancedJSON` so the shared post-processor can trim at a complete top-level JSON object or array.

### Install a GGUF model

The bundled SwiftUI picker handles `.gguf` imports, curated downloads, Hugging Face URL downloads, resume, and deletion. Drop it into a settings pane:

```swift
import CarbocationLocalLLM
import CarbocationLocalLLMRuntime
import CarbocationLocalLLMUI
import SwiftUI

@MainActor
struct LocalModelSettingsView: View {
    let library: ModelLibrary
    @AppStorage("SelectedLocalModelID") private var selectedModelID = ""

    var body: some View {
        ModelLibraryPickerView(
            library: library,
            selectedModelID: $selectedModelID,
            systemModels: LocalLLMEngine.availableSystemModels(),
            calibrationAdapter: ModelLibraryPickerCalibrationAdapter(
                runtimeFingerprint: LocalLLMEngine.contextCalibrationRuntimeFingerprint(),
                calibrate: { model, onProgress in
                    try await LocalLLMEngine.calibrateContext(
                        for: model,
                        in: library,
                        onProgress: { progress in
                            await MainActor.run {
                                onProgress(progress)
                            }
                        }
                    )
                }
            ),
            onModelDeleted: { deleted in
                Task {
                    if await LocalLLMEngine.shared.currentModelID() == deleted.id {
                        await LocalLLMEngine.shared.unload()
                    }
                }
            },
            onConfirmSelection: { selection in
                selectedModelID = selection.storageValue
            }
        )
    }
}
```

When a calibration adapter is supplied, installed model rows show the effective automatic context. Uncalibrated rows show labels such as `context 16,384 (Uncalibrated)` or `context 4,096 (Uncalibrated)` and a `Calibrate` action. Rows with a matching record show the calibrated value, `context 32,768 (Calibrated)`, and a `Recalibrate` action. While calibration runs, the row shows the current probed tier, last successful tier, progress, and a cancel button.

The picker is configurable. By default it shows `CuratedModelCatalog.all`, labels the hardware-recommended curated model, labels the best installed curated fallback when the recommendation is not installed, and marks Apple Intelligence as not recommended while a curated llama.cpp model fits the device memory. If no curated llama.cpp model fits and Apple Intelligence is available, Apple Intelligence receives the recommended label instead. Pass `curatedModels:` to replace the recommended download list, or `labelPolicy:` to replace or suppress picker labels.

GGUF weights are not bundled. Apps either import local `.gguf` files, use the curated Hugging Face downloads, or ship their own download UI.

### How it fits with a speech step

Dictation apps usually compose this package after a speech-to-text step:

```text
CarbocationLocalSpeechRuntime
  Apple Speech or Whisper -> transcript

CarbocationLocalLLMRuntime
  transcript -> cleanup, formatting, command classification

Host app
  hotkeys, Accessibility paste/type, settings policy, product UX
```

## Requirements

**Build**

- Xcode with Swift 5.9 or newer

| Runtime | Minimum OS | Additional requirements |
| --- | --- | --- |
| GGUF / llama.cpp | macOS 14, iOS/iPadOS 17 | Released tags include the binary XCFramework. |
| Apple Intelligence | macOS 26, iOS/iPadOS 26 | Apple Intelligence-compatible device, enabled in Settings, and an SDK that includes Foundation Models. |

**Permissions and entitlements**

Add the keys you actually use to your app:

| Capability | When you need it |
| --- | --- |
| Outgoing Network (`com.apple.security.network.client`) | A sandboxed app downloads GGUFs from Hugging Face or another remote URL |
| App Group | Multiple of your apps share installed GGUF models |

Apple Intelligence is exposed only when the SDK, OS, device, and user setting all support it. The package reports availability through `LocalLLMEngine.availableSystemModels()` and omits Apple Intelligence everywhere else. It additionally requires macOS 26 or iOS/iPadOS 26 or newer, Apple Intelligence enabled in Settings, a supported device, and an app build made with an SDK that includes Foundation Models.

GGUF weights are user data, not part of the Swift package. Let the app download or import them into the model library rather than bundling large model files into the app binary.

On iOS, model downloads use foreground `URLSession` work with resumable partial files. This package does not manage background transfer sessions in this release.

## License

CarbocationLocalLLM is licensed under the MIT License. See [LICENSE](LICENSE).

Published release assets that include `llama.xcframework.zip` redistribute static llama.cpp/ggml object code through `libllama-combined.a`; no separate `ggml*.dylib` files are shipped. Third-party notices for llama.cpp, ggml, and conservative vendored upstream notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Model weights are not distributed by this package. Downloaded or imported GGUF files remain governed by their own upstream model licenses.

## Reference

### Products

| Product | Purpose | Add when |
| --- | --- | --- |
| `CarbocationLocalLLM` | Core model library, selection, context policy, generation options, JSON helpers, download/import support, fake-engine testing helpers. | App code imports core types directly, or you only need shared model storage. |
| `CarbocationLocalLLMRuntime` | Unified facade that routes selections to llama.cpp or Apple Intelligence. | Most apps. This is the entry point. |
| `CarbocationLocalLLMUI` | SwiftUI model library picker, curated downloads, Hugging Face URL downloads, local import, delete, refresh. | You want the bundled UI surfaces. |
| `CarbocationLlamaRuntime` | Lower-level llama.cpp runtime — model probing, chat-template fallback, grammar-aware generation, streaming, cancellation. | You need provider-specific control the unified runtime does not expose. |

`CarbocationAppleIntelligenceRuntime` is an internal implementation target used by the unified runtime; consume Apple Intelligence through `CarbocationLocalLLMRuntime`.

### How the binary release works

For a published release tag such as `v0.3.0`, Xcode resolves the package from GitHub, downloads `llama.xcframework.zip` from the release asset URL recorded in that tag's `Package.swift`, links the products you chose, and builds your app.

The binary artifact is a static XCFramework containing:

- macOS `arm64` and `x86_64`
- iOS device `arm64`
- iOS simulator `arm64` and `x86_64`

SwiftPM handles the link step. The llama runtime declares its own system links for `Metal`, `Accelerate`, `Foundation`, and `libc++`. The archive redistributes static llama.cpp/ggml object code through `libllama-combined.a`; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for bundled and linked notices.

Apple Intelligence has no package artifact. When the SDK, OS, device, and user setting line up, the runtime exposes it as an available system model.

> **Heads up.** As a binary-target consumer your app does not need a sibling checkout, a `Vendor/llama.cpp` submodule, the `Scripts/build-llama-from-xcode.sh` build phase, the `CARBOCATION_LOCAL_LLM_ROOT` env var, a prebuilt `Vendor/llama-artifacts/current` directory, or `../CarbocationLocalLLM` in any path. Those exist only for local package development.

For unreleased work you can point at `branch: "main"`, but llama inference then needs a local source-built artifact on macOS or a local multi-platform XCFramework for iOS builds. Apple Intelligence and core APIs still work without a llama artifact — the runtime simply reports llama-backed generation as unavailable.

### Local development only: adjacent checkout

For active library development or migration work, a host app can temporarily use a local package reference to a sibling checkout:

```text
../CarbocationLocalLLM
```

Treat this as development wiring only. It is not the release-consumer path. For macOS source-artifact development, the host app build must generate this package's ignored llama artifacts before Xcode compiles `CarbocationLlamaRuntime`. To make that safe:

1. Copy this package's `Scripts/build-llama-from-xcode.sh` into the host app, for example as `Scripts/build-carbocation-llama.sh`.
2. Add a scheme Build Pre-action or CI prebuild step that runs:

   ```sh
   "$SRCROOT/Scripts/build-carbocation-llama.sh"
   ```

3. For a scheme Build Pre-action, set "Provide build settings from" to the host app target so `SRCROOT`, `BUILD_DIR`, and related Xcode paths are available.
4. Prefer setting `CARBOCATION_LOCAL_LLM_ROOT` to the package checkout path. The resolver also has a temporary `../CarbocationLocalLLM` fallback for Carbocation app migrations.

An app-target Run Script phase is only sufficient if your Xcode build graph runs it before Swift package dependency compilation. Scheme pre-actions and CI prebuild steps are safer.

---

## For Package Developers

### Clone and build

Clone with the `llama.cpp` submodule:

```sh
git clone --recurse-submodules https://github.com/carbocation/CarbocationLocalLLM.git
cd CarbocationLocalLLM
```

If you already cloned without the submodule:

```sh
git submodule update --init --recursive
```

Run the tests:

```sh
swift test
```

### Open in Xcode

There are two entry points and they show different schemes — pick the one that matches what you're working on:

- **Working on the library** (any `Carbocation*` target, runtime, UI, tests): open the package.
  ```sh
  xed Package.swift
  ```
  You get the SwiftPM library schemes (`CarbocationLocalLLM`, `CarbocationLocalLLMRuntime`, `CarbocationLocalLLMUI`, `CarbocationLlamaRuntime`, and the `CarbocationLocalLLM-Package` umbrella). Use these for editing library code and running the test targets.

- **Running the smoke and demo apps** (`CLLMSmokeMac`, `CLLMSmokeIOS`, `CLLMDemoMac`, `CLLMDemoIOS`): open the apps project.
  ```sh
  open Apps.xcodeproj
  ```
  You get the app schemes, runnable on their respective destinations.

Avoid `open . -a Xcode` — with both `Package.swift` and `Apps.xcodeproj` at the root, Xcode 26 picks package mode and the app schemes will not appear.

The Apple Intelligence live generation smoke test is skipped by default so normal CI does not depend on a supported device. To run it locally on an eligible macOS 26+ system:

```sh
CARBOCATION_RUN_APPLE_INTELLIGENCE_LIVE_TEST=1 swift test \
  --filter CarbocationAppleIntelligenceRuntimeTests/testLiveGenerationWhenExplicitlyEnabled
```

### Build the local llama source artifact

For local llama inference from this checkout:

```sh
ARCHS=arm64 Scripts/build-llama-macos.sh
swift build
```

The script writes:

```text
Vendor/llama-artifacts/current/lib/libllama-combined.a
Vendor/llama-artifacts/current/include/
```

Both paths are gitignored. The script uses a build lock so concurrent app builds do not corrupt shared artifacts.

For CI or a universal local build, omit `ARCHS=arm64`; the script defaults to `arm64 x86_64`.

### Use a local binary artifact

This is a package-development workflow, not the normal app-consumer path. To test the package as a binary-target consumer before publishing:

```sh
Scripts/build-llama-xcframework.sh
CARBOCATION_LOCAL_LLM_BINARY_ARTIFACT_PATH=Vendor/llama-artifacts/release/llama.xcframework swift test
```

`Package.swift` switches to a local `.binaryTarget` whenever `CARBOCATION_LOCAL_LLM_BINARY_ARTIFACT_PATH` is set.

The generated XCFramework includes macOS, iOS device, and iOS simulator slices. Use this path for local iOS app validation.

### Prepare a release artifact

Build, zip, and checksum the multi-platform XCFramework:

```sh
Scripts/build-llama-xcframework.sh
```

The script emits:

```text
Vendor/llama-artifacts/release/llama.xcframework
Vendor/llama-artifacts/release/llama.xcframework.zip
Vendor/llama-artifacts/release/llama.xcframework.zip.checksum
```

To prepare a release manifest manually:

```sh
Scripts/set-llama-binary-artifact.sh \
  "https://github.com/carbocation/CarbocationLocalLLM/releases/download/v0.3.0/llama.xcframework.zip" \
  "$(cat Vendor/llama-artifacts/release/llama.xcframework.zip.checksum)"
```

### Publish a binary release

Use the **Publish Llama Binary Artifact** GitHub workflow.

First run with:

- `tag`: the intended release tag, for example `v0.3.0`
- `prerelease`: `true` for shakedown releases
- `dry_run`: `true`

The dry run builds the artifact, stamps `Package.swift`, and validates the package against the local XCFramework without pushing. Validation includes macOS tests, iOS package imports, and iOS app-style links for device and simulator.

Then run the workflow again with the same tag and `dry_run=false`. The release run creates a tag-only release commit with the binary URL/checksum, creates the tag, uploads the release asset, and validates the published release from a clean temporary consumer package.

Keeping the manifest change on the release tag lets `main` stay source-build friendly while tagged consumers get the binary target.

### Validate a published release

```sh
Scripts/test-binary-release.sh v0.3.0
```

The release workflow runs a consumer import check after uploading the GitHub release asset, then builds the smoke and demo apps from the root Xcode project against the published artifact. This catches problems local validation cannot: tag resolution, checksum mismatch, asset availability, downstream product imports, app-style macOS/iOS links, and llama symbol linkage from the published binary target.

### Quick release checklist

For a normal release, use the GitHub workflow rather than creating the tag by hand. For example, to cut `v0.3.0`:

1. Finish and push the source changes that should be released.
2. Confirm the package is clean locally:

   ```sh
   swift test
   xcodebuild build \
     -project Apps.xcodeproj \
     -scheme CLLMSmokeMac \
     -destination 'generic/platform=macOS' \
     -derivedDataPath .build/CLLMSmokeMacDerivedData \
     CODE_SIGNING_ALLOWED=NO
   xcodebuild build \
     -project Apps.xcodeproj \
     -scheme CLLMSmokeIOS \
     -destination 'generic/platform=iOS' \
     -derivedDataPath .build/CLLMSmokeIOSDerivedData \
     CODE_SIGNING_ALLOWED=NO
   xcodebuild build \
     -project Apps.xcodeproj \
     -scheme CLLMDemoMac \
     -destination 'generic/platform=macOS' \
     -derivedDataPath .build/CLLMDemoMacDerivedData \
     CODE_SIGNING_ALLOWED=NO
   xcodebuild build \
     -project Apps.xcodeproj \
     -scheme CLLMDemoIOS \
     -destination 'generic/platform=iOS' \
     -derivedDataPath .build/CLLMDemoIOSDerivedData \
     CODE_SIGNING_ALLOWED=NO
   ```

3. In GitHub Actions, run `Publish Llama Binary Artifact` with `tag: v0.3.0`, `prerelease: false`, `dry_run: true`.
4. If the dry run passes, run the same workflow again with `dry_run: false`.
5. Optionally verify from a clean consumer package:

   ```sh
   Scripts/test-binary-release.sh v0.3.0
   ```

6. In host apps, update the Swift package version to `0.3.0`, add the `CarbocationLocalLLMRuntime` product, and route model selection/generation through `LLMModelSelection` and `LocalLLMEngine`.

### Runtime modes

`Package.swift` supports these llama runtime modes:

- **Source artifact** — uses `Vendor/llama-artifacts/current/lib/libllama-combined.a` when present.
- **Forced source mode** — set `CARBOCATION_LOCAL_LLM_FORCE_SOURCE_LLAMA=1`.
- **Local binary validation** — set `CARBOCATION_LOCAL_LLM_BINARY_ARTIFACT_PATH`.
- **Published binary release** — non-empty `llamaBinaryArtifactURL` and `llamaBinaryArtifactChecksum`.
- **No llama artifact** — the package builds, but llama inference reports the missing source artifact at runtime.

`main` should normally keep `llamaBinaryArtifactURL` and `llamaBinaryArtifactChecksum` empty so library development uses the source-build path. Release tags can point those constants at the published binary artifact.

### Smoke apps

`CLLMSmokeMac` and `CLLMSmokeIOS` are app schemes defined in `Apps.xcodeproj` (see [Open in Xcode](#open-in-xcode) for which entry point to use). They show Apple Intelligence in the picker when `LocalLLMEngine.availableSystemModels()` reports it available. The macOS smoke asks every provider for JSON; installed GGUF models use grammar-constrained generation, while Apple Intelligence uses prompt guidance plus balanced-JSON post-processing.

For macOS:

1. Select the `CLLMSmokeMac` scheme.
2. Set the destination to `My Mac`.
3. Run.
4. Select an installed model or available system model in the left pane.
5. Click `Run Smoke Test`.

The smoke app uses the shared model cache by default:

```text
~/Library/Group Containers/group.com.carbocation.shared/Models
```

For unsigned/dev builds where the App Group container is unavailable, the core storage helper falls back to per-app Application Support.

If Xcode says the build succeeded but no window appears, clean once with `Product > Clean Build Folder`, then run `CLLMSmokeMac` again.

A successful run prints model load details, streaming events, a normalized JSON response, and:

```text
smoke: ok
```

For iOS, select the `CLLMSmokeIOS` scheme with an iOS device or simulator destination and run it. Both the `CLLMSmokeMac` and `CLLMSmokeIOS` schemes compile from the unified source at `Apps/CLLMSmoke/CLLMSmokeApp.swift`, so the iOS smoke runs the same automated JSON flow as the macOS smoke and ends with `smoke: ok`.

For interactive exploratory testing, use the `CLLMDemoMac` or `CLLMDemoIOS` scheme. Both compile from `Apps/CLLMDemo/CLLMDemoApp.swift` and provide editable prompts, run/cancel controls, output, and a streaming event log.

On iOS, the default llama configuration loads GGUF models CPU-only with a smaller batch size to avoid Metal/backend allocation crashes on first load. Host apps can still opt into GPU offload by passing a nonzero `llamaGPULayerCount`.

Build the local multi-platform llama artifact first:

```sh
Scripts/build-llama-xcframework.sh
```

For GGUFs with an embedded chat template, the runtime should report `embeddedTemplate: true` with `formatter=swift-jinja` or, for templates accepted by llama.cpp's legacy C API, `formatter=legacy-c-api`. If both embedded-template paths fail, the runtime reports a template error instead of silently falling back to descriptor- or filename-inferred prompt tokens.

### Package layout

```text
Apps.xcodeproj                             Root Xcode project hosting first-party macOS and iOS app targets
Apps/
  CLLMSmoke/                              Unified macOS + iOS smoke app source; compiled by both root CLLMSmokeMac and CLLMSmokeIOS schemes
  CLLMDemo/                               Unified macOS + iOS interactive demo source; compiled by CLLMDemoMac and CLLMDemoIOS schemes
Sources/
  CarbocationLocalLLM/                    Core models, selection, context policy, generation options, JSON helpers
  CarbocationLocalLLMRuntime/             Unified facade over llama.cpp and Apple Intelligence
  CarbocationLlamaRuntime/                llama.cpp-backed runtime
  CarbocationAppleIntelligenceRuntime/    Foundation Models-backed runtime (consumed via the unified facade)
  CarbocationLocalLLMUI/                  SwiftUI model library picker
  llama/                                  module map for the llama.cpp build
Tests/
Scripts/
  build-llama-apple-platform.sh           Shared Apple-platform llama.cpp static library builder
  build-llama-macos.sh                    Local llama.cpp source build
  build-llama-from-xcode.sh               Adjacent-checkout build helper for host apps
  build-llama-xcframework.sh              Binary artifact packager
  set-llama-binary-artifact.sh            Release manifest stamper
  test-binary-release.sh                  Clean downstream release import check
Vendor/
  llama.cpp/                              git submodule
```

### Ownership boundaries

Stays in this package:

- GGUF model download, import, deletion, metadata, probing
- provider-aware model selection and persistence
- Apple Intelligence availability gating
- llama.cpp model/context loading, grammar-aware generation, streaming, cancellation
- chat-template fallback handling (including Gemma)
- shared generation options and balanced-JSON post-processing
- SwiftUI model-management surfaces
- smoke tests and diagnostics hooks

Stays in host apps:

- selected-model preference key
- app-specific curated-model list, when the shared default is not the right fit
- app-specific prompts, grammars, and operations
- context cap defaults and generation settings UI
- provider-selection UI policy around the system models returned by `LocalLLMEngine.availableSystemModels()`
- active-engine unload policy after deletion (via `onModelDeleted:`)
- onboarding text, settings copy, and migrations
- entitlement choices beyond documenting requirements
