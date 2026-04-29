# CLLMSmokeIOS

`CLLMSmokeIOS` is a small iOS demo app for validating CarbocationLocalLLM in an app-style environment.

It includes:

- `ModelLibraryPickerView` for GGUF download/import/delete/selection
- Apple Intelligence system-model selection when available
- Prompt input and generation output
- Streaming event log

The demo uses the package's iOS llama defaults: CPU-only GGUF loading, memory mapping enabled, and a smaller initial context to keep first-run memory pressure predictable.

## Run

From the repository root, build the local multi-platform llama artifact:

```sh
Scripts/build-llama-xcframework.sh
```

Then open the repository root in Xcode:

```sh
open . -a Xcode
```

Select the `CLLMSmokeIOS` scheme and an iOS device or simulator. The root project also contains the `CLLMSmokeMac` macOS smoke app. Both schemes run real app bundles from the root Xcode project.

For command-line validation:

```sh
xcodebuild \
  -project CarbocationLocalLLM.xcodeproj \
  -scheme CLLMSmokeIOS \
  -destination 'generic/platform=iOS' \
  -derivedDataPath .build/CLLMSmokeIOSDerivedData \
  CODE_SIGNING_ALLOWED=NO \
  build
```
