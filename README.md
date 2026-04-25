# CarbocationLocalLLM

Shared local-LLM infrastructure for Carbocation macOS apps.

This package provides neutral model storage, model library management, llama.cpp runtime access, shared SwiftUI model-management UI, and a smoke-test app. App-specific prompting, settings policy, migrations, and workflows should stay in the host apps.

## Targets

- `CarbocationLocalLLM`: pure Swift core types and services.
  Includes `InstalledModel`, `ModelLibrary`, `ModelDownloader`, `CuratedModelCatalog`, `GenerationOptions`, `LlamaContextPolicy`, `LLMEngine`, response sanitizing, JSON salvage, and shared helpers. Model downloads use resumable partials, 12-way parallel ranged chunks by default when the server supports them, and single-stream resume fallback otherwise.
- `CarbocationLlamaRuntime`: llama.cpp-backed runtime.
  Includes `LlamaEngine`, model/context loading, chat-template fallback handling, grammar-aware generation, streaming events, cancellation, and model probing.
- `CarbocationLocalLLMUI`: shared SwiftUI model-library UI.
  Includes model selection, installed models, curated downloads, Hugging Face URL downloads, local `.gguf` import, interrupted download handling, delete, refresh, and reveal folder.
- `CLLMSmoke`: Xcode-friendly smoke app.
  Embeds the shared model-library UI, lets you select an installed model, and runs a tiny grammar-constrained generation test.

## Fresh Checkout Setup

Initialize llama.cpp:

```sh
git submodule update --init --recursive
```

Build llama.cpp artifacts:

```sh
ARCHS=arm64 Scripts/build-llama-macos.sh
```

The script writes generated headers and the combined static library under:

```text
Vendor/llama-artifacts/current
```

`Vendor/llama-artifacts/` is intentionally ignored by git. The script uses a build lock so multiple app builds do not corrupt the shared artifacts.

For CI or a universal local build, omit `ARCHS=arm64`; the script defaults to `arm64 x86_64`.

## Verify

Run the package tests:

```sh
swift test
```

Expected current baseline:

```text
24 tests, 0 failures
```

## Smoke Test In Xcode

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

## Expected Smoke Output

A successful run prints model load details, streaming events, a normalized JSON response, and:

```text
smoke: ok
```

For Gemma GGUFs, seeing `embeddedTemplate: true` together with `templateMode=gemma-fallback` is acceptable. It means the model exposes a template, but llama.cpp did not apply it successfully through the native template path, so the shared runtime used its known Gemma fallback prompt format.

## App Integration Pattern

Host apps should depend on the package targets they need:

- Use `CarbocationLocalLLM` for model library state, selected-model preferences, options, context policy, and fake-engine tests.
- Use `CarbocationLocalLLMUI` when the app wants the standard shared model-management UI.
- Use `CarbocationLlamaRuntime` when the app needs real llama.cpp generation.

The shared UI should be embedded as a configurable component. By default it shows `CuratedModelCatalog.all`, but host apps can pass `curatedModels:` to replace the recommended download list for their workload. Apps can also pass `onModelDeleted:` to unload active engines or perform other host-owned cleanup after a model deletion succeeds.

Host apps should still own:

- selected-model preference key
- app-specific curated-model list, when the shared default is not the right fit
- app-specific onboarding/settings copy
- context cap defaults
- generation settings UI
- app-specific prompts, grammars, and operations
- active-engine unload policy after deletion
- migrations and invalid-selection warnings

## llama.cpp Build Notes

SwiftPM links against a generated static library rather than building llama.cpp itself. Apps that depend on `CarbocationLlamaRuntime` must invoke this package's build script before Xcode compiles that Swift package target. Prefer a scheme Build Pre-action or CI prebuild step; an app-target Run Script phase is only sufficient when your Xcode build graph runs it before package dependency compilation.

```sh
"$SRCROOT/Scripts/build-carbocation-llama.sh"
```

For a scheme Build Pre-action, set "Provide build settings from" to the host app target so `SRCROOT`, `BUILD_DIR`, and related Xcode paths are available.

`Scripts/build-carbocation-llama.sh` should be a copy of this package's `Scripts/build-llama-from-xcode.sh`. The resolver keeps app source and project settings independent of a fixed sibling checkout path:

- For local development, set `CARBOCATION_LOCAL_LLM_ROOT` to the package checkout path, or rely on the temporary `../CarbocationLocalLLM` fallback used by Carbocation app migrations.
- For a Git URL/tag package dependency, the resolver searches Xcode's `SourcePackages/checkouts` directory and invokes the checked-out package's `Scripts/build-llama-macos.sh`.

Then link/import the package normally. App source should only reference Swift module and product names such as `import CarbocationLocalLLM`; filesystem paths belong only in build wiring.

This source-build flow is still an internal development integration. For broader reuse, replace the generated local static-library setup with a published binary artifact, such as an XCFramework-backed SwiftPM binary target, so consumers can depend on a Git URL/tag without running a llama.cpp build script.

## Binary Artifact Publishing

The package manifest supports three llama runtime modes:

- default source-build mode, using `Vendor/llama-artifacts/current`
- local binary validation mode, using `CARBOCATION_LOCAL_LLM_BINARY_ARTIFACT_PATH`
- published binary mode, using the `llamaBinaryArtifactURL` and `llamaBinaryArtifactChecksum` constants in `Package.swift`

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

The preferred release path is the `Publish Llama Binary Artifact` GitHub workflow. Run it first with `dry_run=true`; that builds the artifact, stamps `Package.swift`, and validates the package against the local XCFramework without pushing anything. Run it again with `dry_run=false` to create a tag-only release commit with the binary URL/checksum, create the tag, and upload the release asset. Keeping the manifest change on the release tag lets `main` stay source-build friendly while tagged consumers get the binary target.
