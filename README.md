# CarbocationLocalLLM

Shared local-LLM infrastructure for Carbocation macOS apps.

This package provides neutral model storage, model library management, llama.cpp runtime access, shared SwiftUI model-management UI, and a smoke-test app. App-specific prompting, settings policy, migrations, and workflows should stay in the host apps.

## Targets

- `CarbocationLocalLLM`: pure Swift core types and services.
  Includes `InstalledModel`, `ModelLibrary`, `ModelDownloader`, `CuratedModelCatalog`, `GenerationOptions`, `LlamaContextPolicy`, `LLMEngine`, response sanitizing, JSON salvage, and shared helpers.
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
19 tests, 0 failures
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

The shared UI should be embedded as a configurable component. Host apps should still own:

- selected-model preference key
- app-specific onboarding/settings copy
- context cap defaults
- generation settings UI
- app-specific prompts, grammars, and operations
- active-engine unload policy after deletion
- migrations and invalid-selection warnings

## llama.cpp Build Notes

SwiftPM links against a generated static library rather than building llama.cpp itself. Apps should add an Xcode Run Script phase before Swift compilation that invokes this package's build script:

```sh
/Users/james/projects/CarbocationLocalLLM/Scripts/build-llama-macos.sh
```

Then link/import the package normally. This keeps llama.cpp's C++/Metal build complexity out of the pure Swift core while still giving app users a normal Xcode Run experience.
