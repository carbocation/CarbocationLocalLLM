# CarbocationLocalLLM

Shared local-LLM infrastructure for Carbocation macOS apps.

This package provides neutral model storage, model library management, llama.cpp runtime access, shared SwiftUI model-management UI, and a smoke-test app. Host apps should keep app-specific prompting, settings policy, migrations, onboarding, and workflows in the host app.

## Consuming Apps

Use a tagged release for normal app integration. A release tag points SwiftPM at a published `llama.xcframework.zip` asset, so the host app does not need a sibling checkout, a llama.cpp submodule, or a build script for the llama runtime.

A GitHub release does not replace the Swift package dependency. The host app still adds `CarbocationLocalLLM` by Git URL and selects a release version; Xcode/SwiftPM then reads that tag's `Package.swift` and downloads the binary artifact automatically.

Current public release: [`v0.1.0`](https://github.com/carbocation/CarbocationLocalLLM/releases/tag/v0.1.0).

### Add The Release In Xcode

1. Open the host macOS app project in Xcode.
2. Choose `File > Add Package Dependencies...`.
3. Paste the package URL:

```text
https://github.com/carbocation/CarbocationLocalLLM.git
```

4. Choose release `0.1.0` / tag `v0.1.0`.
   Prefer `Exact Version` while integrating, or `Up to Next Major Version` once the app has a tested upgrade policy.
5. Add the package products to the app target that will use them.

Do not choose a branch rule for `main` for a shipping app. `main` stays source-build friendly for library development, while release tags are the path that should resolve the llama runtime through the binary artifact.

Do not download or drag `llama.xcframework` into the app project manually. The release asset is referenced by the package manifest and is fetched by SwiftPM during package resolution.

### Pick Products

- `CarbocationLocalLLM`: core model-library types, selected-model state, context policy, generation options, JSON helpers, download/import support, and fake-engine testing helpers.
- `CarbocationLocalLLMUI`: shared SwiftUI model-library UI for installed models, curated downloads, Hugging Face URL downloads, local `.gguf` import, interrupted downloads, delete, refresh, and reveal folder.
- `CarbocationLlamaRuntime`: llama.cpp-backed generation, model/context loading, model probing, chat-template fallback handling, grammar-aware generation, streaming events, and cancellation.
- `CarbocationAppleIntelligenceRuntime`: optional Apple Intelligence / Foundation Models-backed generation for systems where Apple's on-device model is available. This product builds on older SDKs, but reports unavailable unless the app is built with the Foundation Models SDK and runs on a supported device with Apple Intelligence enabled.

Most apps that provide GGUF-backed local generation should add `CarbocationLocalLLM`, `CarbocationLocalLLMUI`, and `CarbocationLlamaRuntime` to the app target. Apps that want to offer Apple's system model before a user downloads a larger GGUF can also add `CarbocationAppleIntelligenceRuntime`. Apps that only need shared model storage or metadata can use `CarbocationLocalLLM` alone.

Host app source should import module names only:

```swift
import CarbocationLocalLLM
import CarbocationLocalLLMUI
import CarbocationLlamaRuntime
import CarbocationAppleIntelligenceRuntime
```

Filesystem paths such as `../CarbocationLocalLLM` should not appear in app source or Xcode package dependencies for the binary-release path.

### What Xcode Should Build

For a release tag, Xcode should:

1. Resolve `CarbocationLocalLLM` from GitHub.
2. Download `llama.xcframework.zip` from the release asset URL recorded in that tag's `Package.swift`.
3. Link the selected products into the host app target.
4. Build the app normally.

The host app should not add `Scripts/build-llama-from-xcode.sh`, initialize `Vendor/llama.cpp`, set `CARBOCATION_LOCAL_LLM_ROOT`, or prebuild `Vendor/llama-artifacts/current`. Those steps are only for local package development or temporary adjacent-checkout migration work.

The binary artifact is a static XCFramework. SwiftPM handles the package link step, and `CarbocationLlamaRuntime` declares its own system links for `Metal`, `Accelerate`, `Foundation`, and `libc++`.

`CarbocationAppleIntelligenceRuntime` has no model artifact. It calls Apple's system model through Foundation Models when the runtime reports available.

### Minimal Host-App Wiring

Create a single model library for the app and use the runtime probe when you want imported models to record their training context:

```swift
import CarbocationLocalLLM
import CarbocationLlamaRuntime

@MainActor
let modelLibrary = ModelLibrary(
    root: ModelStorage.modelsDirectory(
        sharedGroupIdentifier: ModelStorage.defaultSharedGroupID,
        appSupportFolderName: "YourAppName"
    ),
    contextLengthProbe: { url in
        LlamaEngine.probeTrainingContext(at: url)
    }
)
```

Add the shared picker wherever the app lets the user choose, download, import, or delete models:

```swift
import CarbocationLocalLLM
import CarbocationLocalLLMUI
import CarbocationLlamaRuntime
import CarbocationAppleIntelligenceRuntime
import SwiftUI

struct LocalModelSettingsView: View {
    let library: ModelLibrary
    @AppStorage("SelectedLocalModelID") private var selectedModelID = ""

    var body: some View {
        ModelLibraryPickerView(
            library: library,
            selectedModelID: $selectedModelID,
            systemModels: AppleIntelligenceEngine.systemModelOption().map { [$0] } ?? [],
            onConfirmSystemModel: { systemModel in
                if systemModel.id == AppleIntelligenceEngine.systemModelID {
                    // Route generation to AppleIntelligenceEngine.shared.
                }
            },
            onModelDeleted: { deleted in
                Task {
                    if await LlamaEngine.shared.currentModelID() == deleted.id {
                        await LlamaEngine.shared.unload()
                    }
                }
            },
            onConfirm: { model in
                selectedModelID = model.id.uuidString
            }
        )
    }
}
```

Before generation, load the selected model and resolve the requested context through the shared policy helpers:

```swift
guard let model = modelLibrary.model(id: selectedModelID) else {
    return
}

let requestedContext = LlamaContextPolicy.resolvedRequestedContext(for: model)
let engine = LlamaEngine.shared

try await engine.load(
    model: model,
    from: modelLibrary.root,
    requestedContext: requestedContext
)

let response = try await engine.generate(
    system: "You are a concise assistant.",
    prompt: userPrompt,
    options: .extractionSafe
) { event in
    // Update host-app progress UI if desired.
}
```

### Apple Intelligence Runtime

Apple's system model should be treated as a separate provider from installed GGUF models. Check availability before showing it in the host app:

```swift
import CarbocationAppleIntelligenceRuntime

let availability = AppleIntelligenceEngine.availability()
if availability.shouldOfferModelOption {
    // Show an "Apple Intelligence" model option.
}
```

`ModelLibraryPickerView` can render that option above installed GGUF models:

```swift
ModelLibraryPickerView(
    library: library,
    selectedModelID: $selectedModelID,
    systemModels: AppleIntelligenceEngine.systemModelOption().map { [$0] } ?? [],
    onConfirmSystemModel: { systemModel in
        guard systemModel.id == AppleIntelligenceEngine.systemModelID else { return }
        // Route generation to AppleIntelligenceEngine.shared.
    },
    onConfirm: { installedModel in
        // Route generation to LlamaEngine.shared.
    }
)
```

When selected, use the engine through the same `LLMEngine` surface:

```swift
let engine = AppleIntelligenceEngine.shared
let response = try await engine.generate(
    system: "You are a concise assistant.",
    prompt: userPrompt,
    options: .extractionSafe
) { event in
    // Update host-app progress UI if desired.
}
```

The Apple Intelligence runtime maps temperature, top-k/top-p style sampling, seeds, and maximum response tokens where Foundation Models exposes an equivalent. llama.cpp GBNF grammars are not supported by Apple's text response API; by default this runtime still tries to generate and applies stop-sequence and balanced-JSON post-processing. Set `AppleIntelligenceEngineConfiguration(unsupportedFeatureBehavior: .fail)` if the host app would rather reject unsupported options before generation.

### Xcode Target Settings

- Minimum deployment target: macOS 14.
- App Sandbox network client entitlement: required if the app downloads models from Hugging Face or another remote URL.
- App Group entitlement: required only if the app wants shared model storage through `ModelStorage.defaultSharedGroupID` or another shared group. Without a usable App Group container, `ModelStorage.modelsDirectory` falls back to the app's Application Support folder.
- Model files: `.gguf` weights are user data, not part of the Swift package. Let the app download/import them into the model library instead of bundling large model files into the app binary.
- Apple Intelligence: requires a supported device, Apple Intelligence enabled in System Settings, macOS 26 or newer, and an app build made with an SDK that includes Foundation Models.

### Host-App Responsibilities

This library deliberately avoids owning app policy. Host apps should still own:

- selected-model preference key
- app-specific curated-model list, when the shared default is not the right fit
- app-specific onboarding/settings copy
- context cap defaults
- generation settings UI
- provider-selection UI, including whether to offer Apple Intelligence when `AppleIntelligenceEngine.availability().shouldOfferModelOption` is true
- app-specific prompts, grammars, and operations
- active-engine unload policy after deletion
- migrations and invalid-selection warnings

`ModelLibraryPickerView` is configurable. By default it shows `CuratedModelCatalog.all`, but host apps can pass `curatedModels:` to replace the recommended download list. Apps can also pass `onModelDeleted:` to unload active engines or perform other host-owned cleanup after a model deletion succeeds.

### Temporary Adjacent-Checkout Path

For active library development or migration work, a host app can temporarily use a local package reference to a sibling checkout:

```text
../CarbocationLocalLLM
```

Treat this as development wiring only. It is not the release-consumer path. It is riskier because the app build must generate this package's ignored llama artifacts before Xcode compiles `CarbocationLlamaRuntime`.

For that temporary setup:

1. Copy this package's `Scripts/build-llama-from-xcode.sh` into the host app, for example as `Scripts/build-carbocation-llama.sh`.
2. Add a scheme Build Pre-action or CI prebuild step that runs:

```sh
"$SRCROOT/Scripts/build-carbocation-llama.sh"
```

3. For a scheme Build Pre-action, set "Provide build settings from" to the host app target so `SRCROOT`, `BUILD_DIR`, and related Xcode paths are available.
4. Prefer setting `CARBOCATION_LOCAL_LLM_ROOT` to the package checkout path. The resolver also has a temporary `../CarbocationLocalLLM` fallback for Carbocation app migrations.

An app-target Run Script phase is only sufficient if your Xcode build graph runs it before Swift package dependency compilation. Scheme pre-actions and CI prebuild steps are safer.

## Developers

Use this section when you are modifying `CarbocationLocalLLM` itself.

### Package Layout

- `Sources/CarbocationLocalLLM`: core Swift services and shared types.
- `Sources/CarbocationLocalLLMUI`: shared SwiftUI model-library UI.
- `Sources/CarbocationLlamaRuntime`: llama.cpp-backed runtime.
- `Sources/CLLMSmoke`: Xcode-friendly smoke app.
- `Sources/llama`: source-build module map for generated llama headers.
- `Scripts/build-llama-macos.sh`: local llama.cpp source build.
- `Scripts/build-llama-xcframework.sh`: binary artifact packager.
- `Scripts/set-llama-binary-artifact.sh`: release manifest stamper.
- `Scripts/test-binary-release.sh`: clean downstream release smoke test.

### Fresh Checkout

Initialize llama.cpp:

```sh
git submodule update --init --recursive
```

Build local llama artifacts:

```sh
ARCHS=arm64 Scripts/build-llama-macos.sh
```

The script writes generated headers and the combined static library under:

```text
Vendor/llama-artifacts/current
```

`Vendor/llama-artifacts/` is intentionally ignored by git. The script uses a build lock so concurrent app builds do not corrupt shared artifacts.

For CI or a universal local build, omit `ARCHS=arm64`; the script defaults to `arm64 x86_64`.

### Test

Run the package tests:

```sh
swift test
```

Expected current baseline:

```text
0 failures
```

The Apple Intelligence live generation smoke test is skipped by default so normal CI does not depend on a supported Apple Intelligence device. To run it locally on an eligible macOS 26+ system:

```sh
CARBOCATION_RUN_APPLE_INTELLIGENCE_LIVE_TEST=1 swift test --filter CarbocationAppleIntelligenceRuntimeTests/testLiveGenerationWhenExplicitlyEnabled
```

### Smoke App

`CLLMSmoke` shows Apple Intelligence in the picker when `AppleIntelligenceEngine.systemModelOption()` reports available. Its Apple Intelligence smoke path checks that the system model can generate a non-empty response; the GGUF path keeps the stricter grammar-constrained JSON smoke test.

Open `Package.swift` in Xcode.

1. Select the `CLLMSmoke` scheme.
2. Set the destination to `My Mac`.
3. Run.
4. Select an installed model in the left pane.
5. Click `Run Smoke Test`.

The smoke app uses the shared model cache by default:

```text
~/Library/Group Containers/group.com.carbocation.shared/Models
```

For unsigned/dev builds where the App Group container is unavailable, the core storage helper falls back to per-app Application Support.

If Xcode says the build succeeded but no window appears, clean once with `Product > Clean Build Folder`, then run `CLLMSmoke` again.

A successful run prints model load details, streaming events, a normalized JSON response, and:

```text
smoke: ok
```

For Gemma GGUFs, seeing `embeddedTemplate: true` together with `templateMode=gemma-fallback` is acceptable. It means the model exposes a template, but llama.cpp did not apply it successfully through the native template path, so the shared runtime used its known Gemma fallback prompt format.

### llama Runtime Modes

`Package.swift` supports three llama runtime modes:

- default source-build mode, using `Vendor/llama-artifacts/current`
- local binary validation mode, using `CARBOCATION_LOCAL_LLM_BINARY_ARTIFACT_PATH`
- published binary mode, using the `llamaBinaryArtifactURL` and `llamaBinaryArtifactChecksum` constants in `Package.swift`

`main` should normally keep `llamaBinaryArtifactURL` and `llamaBinaryArtifactChecksum` empty so library development uses the source-build path. Release tags can point those constants at the published binary artifact.

### Build A Binary Artifact

Build and validate a local XCFramework artifact:

```sh
Scripts/build-llama-xcframework.sh
CARBOCATION_LOCAL_LLM_BINARY_ARTIFACT_PATH=Vendor/llama-artifacts/release/llama.xcframework swift test
```

The packaging script emits:

```text
Vendor/llama-artifacts/release/llama.xcframework
Vendor/llama-artifacts/release/llama.xcframework.zip
Vendor/llama-artifacts/release/llama.xcframework.zip.checksum
```

To prepare a release manifest manually:

```sh
Scripts/set-llama-binary-artifact.sh \
  "https://github.com/carbocation/CarbocationLocalLLM/releases/download/vX.Y.Z/llama.xcframework.zip" \
  "$(cat Vendor/llama-artifacts/release/llama.xcframework.zip.checksum)"
```

### Publish A Binary Release

The preferred release path is the `Publish Llama Binary Artifact` GitHub workflow.

First run it with:

- `tag`: the intended release tag, for example `vX.Y.Z`
- `prerelease`: `true` for shakedown releases
- `dry_run`: `true`

The dry run builds the artifact, stamps `Package.swift`, and validates the package against the local XCFramework without pushing anything.

Then run the workflow again with the same tag and `dry_run=false`. The release run creates a tag-only release commit with the binary URL/checksum, creates the tag, uploads the release asset, and validates the published release from a clean temporary consumer package.

Keeping the manifest change on the release tag lets `main` stay source-build friendly while tagged consumers get the binary target.

### Validate A Published Release

After publishing, verify the release from a clean temporary consumer package:

```sh
Scripts/test-binary-release.sh vX.Y.Z
```

The release workflow runs the same smoke test after uploading the GitHub release asset. This catches problems that local binary validation cannot see, including tag resolution, checksum mismatch, release asset availability, downstream product imports, and llama symbol linkage from the published binary target.
