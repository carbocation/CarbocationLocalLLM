# Audio App Integration Guide

This library accepts audio input only as in-memory request payloads. The app owns attachment storage, file access, encryption, filenames, attachment IDs, recording UI, and history policy.

## Ownership Boundaries

Before calling the library, the app reads or decrypts the audio bytes and creates an `LLMAudioInput`:

```swift
let audio = LLMAudioInput.encoded(data: wavData, mimeType: "audio/wav")
let message = LLMChatMessage(role: .user, content: [
    .text("Transcribe the following speech segment in its original language."),
    .audio(audio)
])
```

For raw audio, provide normalized mono Float32 PCM at the sample rate reported by the loaded model's multimodal projector. Gemma 4 audio models are expected to use 16 kHz:

```swift
let audio = LLMAudioInput.pcmFloat32Mono(sampleRate: 16_000, data: pcmFloat32Data)
```

`LLMAudioInput` has no file path, filename, attachment ID, transcript, or diagnostic metadata. If the model should know a filename, timestamp, speaker label, or prior transcript, add that as adjacent `.text(...)` content.

## Conversation History

A later turn can inspect prior audio only if the app includes that audio bytes again in the later request. The library does not remember prior audio payloads after generation completes.

This means the app decides history policy:

- Include recent audio bytes again when follow-up questions should reference them.
- Omit audio bytes when the next turn should be text-only.
- Add app-owned transcripts or summaries as text when that is the desired fallback.
- Reject, trim, or chunk long audio according to the app's product policy.

## Capability And Preflight

Check model capability before showing audio controls or sending audio:

```swift
let info = await engine.currentLoadedModelInfo()
guard info?.supportsAudio == true else {
    // Hide audio sending or show a model capability error.
    return
}
```

For GGUF model families that ship multimodal support separately, install the text model and its companion `mmproj` GGUF. Hugging Face downloads and Hugging Face cache discovery include matching `mmproj` artifacts automatically when the repository exposes them. For local file import, select both files, or keep the companion `mmproj*.gguf`/`*-mmproj.*.gguf` readable next to the text model file.

Run preflight before generation:

```swift
let preflight = try await engine.preflight(messages: [message], options: options)
guard preflight.canGenerate else {
    // Ask the user to shorten context, remove audio, or choose a larger context.
    return
}
```

Preflight validates audio placement and data, counts text plus audio context cost for audio-capable llama models, and reports errors before generation whenever possible.

## Format Rules

Encoded audio can be WAV, MP3, or FLAC. If `mimeType` is omitted, the library sniffs by magic bytes. If `mimeType` is present and conflicts with sniffed content, the request fails.

Raw PCM audio must be tightly packed Float32 mono:

- `sampleRate` must match the loaded mtmd audio sample rate.
- `data.count > 0`
- `data.count % 4 == 0`
- Samples must be finite and normalized to `[-1, 1]`.
- No stereo, planar data, integer PCM, or implicit resampling.

For v1, each audio part is limited to 30 seconds.

## Current Runtime Limits

Audio parts are accepted only in user messages. Prompt cache reuse, MTP acceleration, and tool generation are disabled for multimodal requests. Apple Intelligence remains text-only and reports only `.text` support.
