# Native Dia 2 in audio.cpp

This fork adds an experimental native GGML implementation of [Nari Labs Dia2](https://github.com/nari-labs/dia2). It runs the temporal transformer, all 31 scheduled depth-transformer stages, text alignment/sampling, and the 32-codebook Mimi decoder in C++. Python is used only for conversion and reference validation.

The tested package is **Dia2-1B F16**: [markldn/Dia2-1B-GGUF](https://huggingface.co/markldn/Dia2-1B-GGUF). Download the complete package, including tokenizer sidecars and `mimi-f16.gguf`; the main GGUF alone is insufficient. The main file is approximately 2.15 GB, and Mimi adds approximately 193 MB (decimal units).

## Build and run

```sh
git clone https://github.com/markldn/audio.cpp-dia2
cd audio.cpp-dia2
./serve/build-dia2.sh cpu
hf download markldn/Dia2-1B-GGUF --local-dir models/Dia2-1B-GGUF
build/dia2-cpu/bin/audiocpp_cli --task tts --family dia2 \
  --model models/Dia2-1B-GGUF --backend cpu --threads 6 \
  --text 'Hello there. This is Dia two.' --out speech.wav --metrics
./serve/start-dia2.sh cpu
```

The separate CPU server listens on `127.0.0.1:8197`, with model ID `dia2-1b`. Existing audio.cpp servers can keep their binaries, model directories, ports and profiles. Use a separate checkout/build directory when installing this fork alongside a working installation.

An example user service is provided in `serve/audiocpp-dia2-server.service`; adjust its checkout location before installing. For a llama-swap chat proxy, add a separate TTS route pointing to this server, with model `dia2-1b`. Omit `voice` for an unconditioned voice, or supply a configured voice-library name for cloning. Allow up to 30 minutes for CPU requests. Keep this route outside any GPU1 model-unload logic.

```sh
curl http://127.0.0.1:8197/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"dia2-1b","input":"Hello there.","response_format":"wav"}' \
  --output speech.wav
```

Plain text defaults to speaker one; `[S1]` and `[S2]` markers support dialogue. Output is mono 24 kHz. Ordinary requests return a complete WAV; streaming requests deliver incremental PCM. CPU generation remains slower than realtime and competing workloads can increase latency substantially.

For AMD HIP, build with `HIP_ARCH=<your architecture> ./serve/build-dia2.sh hip`, then use `./serve/start-dia2.sh gpu1`. The supplied HIP profile selects device 1. For Vulkan, build with `./serve/build-dia2.sh vulkan`, then use `./serve/start-dia2.sh vulkan-gpu1`; this hides other Vulkan devices with `GGML_VK_VISIBLE_DEVICES=1` and selects visible device zero. CPU inference is validated; GPU builds are checked by compilation only. Do not evict existing GPU workloads to validate this port.

## Voice cloning and profiles

Supply reference audio plus its transcript. The native Mimi encoder produces 32-codebook prefix tokens, and the model warms its temporal cache while consuming forced prefix words before generating the new script. The reference prefix is cropped from output by default. Each reference must be 0.32–30 seconds, and prefix plus generated speech must fit the 1500-frame context. Speaker-two conditioning is also accepted using the options below; the measured profile tests use a single reference speaker.

```sh
curl http://127.0.0.1:8197/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"dia2-1b","input":"Hello there.","voice_ref":"/absolute/reference.wav","reference_text":"The exact words spoken in the reference.","response_format":"wav"}' \
  --output cloned.wav
```

For best alignment, pass `options["dia2.reference_words"]` as a JSON string containing `[{"text":"Hello","start":0.1,"end":0.5}, ...]`, with timestamps in seconds. A transcript alone uses approximate length-weighted word timing; this is convenient but can reduce clone quality relative to accurate word timestamps. A second speaker uses `dia2.prefix_speaker_2` (WAV path) plus `dia2.reference_text_2` or `dia2.reference_words_2`. Set `dia2.include_prefix=true` to retain the prefix. No Python or ASR model is required for native inference.

The profiles use the private `serve/voices/` directory. Put mono WAV files and a `prompt_text` file there, with lines `name|exact reference transcript`. Named profiles are discovered by `GET /v1/audio/voices?model=dia2-1b`; request them using `"voice":"name"`. Recordings and private transcripts are ignored by Git and are not uploaded with the GGUF package.

`serve/dia2-kitt-cpu.json` and `./serve/start-dia2.sh kitt-cpu` demonstrate a KITT CPU profile with two threads to reduce contention on a busy machine. On the local chat installation, the existing KITT and Lou recordings are reused as `dia2:kitt` / **Dia 2 KITT (CPU)** and `dia2:lou` / **Dia 2 Lou (CPU)**. Existing Chatterbox profiles remain available. Settings can choose a main, narrator or character voice, and the roleplay panel can override the character voice for a specific chat. KITT/Lou voice modes prefer an explicitly selected Dia2 profile and otherwise use the available engine choices.

## Incremental audio streaming

The supplied server profiles use `mode=streaming` and accept both complete WAV requests and incremental PCM requests. Streaming advances the same generation state and decodes each completed group of delayed audio frames through stateful Mimi. It does not regenerate text sentence by sentence. `dia2.chunk_frames` controls chunk size (1–25, default four frames / 320 ms).

```sh
curl -N http://127.0.0.1:8197/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"dia2-1b","input":"Hello there. This is streaming speech.","stream":true,"stream_format":"sse","response_format":"pcm"}'
```

SSE carries base64 signed 16-bit little-endian mono PCM at 24 kHz, followed by `speech.audio.done` and `[DONE]`. `stream_format=audio` delivers raw PCM bytes instead. WAV is supported for complete responses only. The chat proxy forwards raw PCM promptly and the browser schedules arriving chunks with WebAudio; stop cancels the request and queued audio. CPU generation can leave playback gaps because it is slower than realtime. Cloning adds prefix preparation latency. Incremental text input and live speech-to-speech input are not implemented; supply the complete requested script.

## Vocal cues and chat behavior

Dia2's official [demo](https://github.com/nari-labs/dia2/blob/main/gradio_app.py) uses `(laughs)`, and the bundled tokenizer defines cues including `(sighs)`, `(gasps)`, `(whispers)`, `(sobs)` and `(clears throat)`. These use parentheses; arbitrary `(happy)` or `(angry)` tags are not a documented control. The native parser preserves supported multiword tags as single tokenizer tokens. A recognized token does not guarantee the desired acoustic effect or intelligibility on every sample; try neutral text or another seed if a cue produces poor audio.

For example: `[S1] That was a good joke. (laughs) I will give you that one.` Use cues sparingly. Tone and subtle emotions also depend on wording and reference audio.

The local chat's optional **Let Dia 2 use occasional vocal cues when appropriate** setting instructs the conversation model to default to neutral speech and use at most one context-appropriate cue in a short reply. In roleplay the cue stays inside quoted character dialogue; the narrator voice is independent. This is model-guided behavior, not a guaranteed emotion classifier. No GPU model request is needed to configure the setting.

## Conversion and validation

Install the converter dependencies (`torch`, `safetensors`, `numpy`, `gguf`, `huggingface_hub`), download official `nari-labs/Dia2-1B` and `kyutai/mimi` snapshots, then run:

```sh
python tools/community_models/dia2/convert_dia2_gguf.py \
  models/Dia2-1B models/Dia2-1B-GGUF --mimi models/mimi --precision f16
build/dia2-cpu/bin/dia2_parity_probe models/Dia2-1B-GGUF tests/dia2/out-f16
python tests/dia2/compare_reference.py models/Dia2-1B-GGUF \
  models/Dia2-1B/model.safetensors /path/to/nari-labs/dia2 \
  tests/dia2/out-f16 --mimi models/mimi --f16-weights
```

Reference validation requires the dependencies of the upstream model modules and Hugging Face Transformers with Mimi support. The deterministic test compares three temporal steps, all 31 depth stages, and decoded Mimi PCM against the official PyTorch implementation. Native sampling has a different random-number generator, so complete utterances need not match Python sample-for-sample.

With compilation finished and the model resident, a subsequent CPU server request generated **exactly 10.0 seconds of audio in 61.20 seconds**, using six requested threads (RTF 6.12). See [the measured request](reports/dia2-cpu-benchmark.json). Cold loading and competing workloads add latency.

The validated CPU smoke test, run during HIP compilation, produced 4.48 seconds of audio in 739.32 seconds on a busy machine (six requested threads, peak RSS approximately 3.19 GiB). Parakeet recognized the sentence, with errors for the model name and spelled-out letters. The eight-codebook decoder regression and tokenizer/speaker-marker comparison also passed. Updated prefix, encoder, streaming and transformer results are recorded in [dia2-parity-report.json](reports/dia2-parity-report.json). HTTP timings and streaming/full equivalence are recorded in [dia2-streaming-validation.json](reports/dia2-streaming-validation.json).

GGUF stores short physical tensor names plus logical names, original ranks and shapes in metadata. Mimi conversion also translates HF module names and its rotary projection layout. The existing Mimi runtime now honors its configured codebook count; its default remains eight for existing users.

Supported and CPU-checked: English 1B TTS, speaker-marked dialogue, single-speaker audio-prefix cloning and incremental audio output. Two-speaker prefixes are implemented but have not been acoustically validated. Not yet validated or implemented: 2B model inference, incremental text/live audio input and non-English speech. The converter accepts F32/F16/Q8; only the F16 package has been validated.

## Attribution and licenses

Dia2 code and weights: [nari-labs/dia2](https://github.com/nari-labs/dia2), Apache-2.0. Mimi weights: [kyutai/mimi](https://huggingface.co/kyutai/mimi), CC-BY-4.0. This fork retains audio.cpp's upstream license and attribution. See the Hugging Face model card for exact source revisions and validation results.
