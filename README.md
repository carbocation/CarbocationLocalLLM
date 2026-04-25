# CarbocationLocalLLM

Shared local-LLM infrastructure for Carbocation macOS apps.

This package provides neutral model storage, model library management, llama.cpp runtime access, shared SwiftUI model-management UI, and a smoke-test app. Host apps should keep app-specific prompting, settings policy, migrations, onboarding, and workflows in the host app.

## Users

Use a tagged release unless you are actively developing this library. Tagged releases are intended to resolve the llama runtime through a published SwiftPM binary artifact, so consuming apps do not need a sibling checkout or a local llama.cpp build.

### Add The Package

Add the package by Git URL in Xcode or SwiftPM:

```text
https://github.com/carbocation/CarbocationLocalLLM.git
```

Select a release tag such as `vX.Y.Z`. Do not depend on `main` for normal app integration; `main` stays source-build friendly for library development.

Host app source should only use Swift module and product names:

```swift
import CarbocationLocalLLM
import CarbocationLocalLLMUI
import CarbocationLlamaRuntime
```

Filesystem paths such as `../CarbocationLocalLLM` should not appear in app source.

### Pick Products

- `CarbocationLocalLLM`: pure Swift core types and services.
  Use this for model library state, selected-model preferences, generation options, context policy, curated model metadata, fake-engine tests, response sanitizing, JSON salvage, and shared helpers.
- `CarbocationLocalLLMUI`: shared SwiftUI model-library UI.
  Use this when the app wants standard model selection, installed models, curated downloads, Hugging Face URL downloads, local `.gguf` import, interrupted download handling, delete, refresh, and reveal folder.
- `CarbocationLlamaRuntime`: llama.cpp-backed runtime.
  Use this when the app needs real local generation, model/context loading, chat-template fallback handling, grammar-aware generation, streaming events, cancellation, or model probing.

### Host-App Responsibilities

This library deliberately avoids owning app policy. Host apps should still own:

- selected-model preference key
- app-specific curated-model list, when the shared default is not the right fit
- app-specific onboarding/settings copy
- context cap defaults
- generation settings UI
- app-specific prompts, grammars, and operations
- active-engine unload policy after deletion
- migrations and invalid-selection warnings

`ModelLibraryPickerView` is configurable. By default it shows `CuratedModelCatalog.all`, but host apps can pass `curatedModels:` to replace the recommended download list. Apps can also pass `onModelDeleted:` to unload active engines or perform other host-owned cleanup after a model deletion succeeds.

### Binary Release Path

The preferred consumer path is:

1. Depend on a release tag.
2. Link the products your app needs.
3. Build normally in Xcode.

No llama build script should be required by consuming apps when the selected tag contains a published binary artifact URL/checksum in `Package.swift`.

### Temporary Adjacent-Checkout Path

For active migration work, a host app can use a local package reference to a sibling checkout:

```text
../CarbocationLocalLLM
```

Treat this as development wiring only. It is riskier than the binary release path because the app build must generate this package's ignored llama artifacts before Xcode compiles `CarbocationLlamaRuntime`.

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
24 tests, 0 failures
```

### Smoke App

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
