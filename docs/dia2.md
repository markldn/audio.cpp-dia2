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

An example user service is provided in `serve/audiocpp-dia2-server.service`; adjust its checkout location before installing. For a llama-swap chat proxy, add a separate TTS route pointing to this server, with model `dia2-1b` and no `voice` field. Allow up to 30 minutes for CPU requests. Keep this route outside any GPU1 model-unload logic.

```sh
curl http://127.0.0.1:8197/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"dia2-1b","input":"Hello there.","response_format":"wav"}' \
  --output speech.wav
```

Do not supply a `voice` field: voice-prefix conditioning is not implemented. Plain text defaults to speaker one; `[S1]` and `[S2]` markers support dialogue. Output is mono 24 kHz WAV. Generation is offline and can take several minutes on CPU, even for a short sentence. This is not a realtime CPU speech engine.

For AMD HIP, build with `HIP_ARCH=<your architecture> ./serve/build-dia2.sh hip`, then use `./serve/start-dia2.sh gpu1`. The supplied HIP profile selects device 1. CPU inference is validated; HIP performance and correctness must be checked on the target GPU before relying on that profile. Do not evict existing GPU workloads just to validate this port.

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

The validated CPU smoke test produced 4.48 seconds of audio in 739.32 seconds on a busy machine (six requested threads, peak RSS approximately 3.19 GiB). Parakeet recognized the sentence, with errors for the model name and spelled-out letters. The eight-codebook decoder regression and tokenizer/speaker-marker comparison also passed. Numerical results are recorded in [dia2-f16-parity.json](reports/dia2-f16-parity.json).

GGUF stores short physical tensor names plus logical names, original ranks and shapes in metadata. Mimi conversion also translates HF module names and its rotary projection layout. The existing Mimi runtime now honors its configured codebook count; its default remains eight for existing users.

Supported: English offline 1B TTS and speaker-marked dialogue. Not yet validated or implemented: 2B model inference, streaming, voice-prefix cloning and non-English speech. The converter accepts F32/F16/Q8; only the F16 package has been validated.

## Attribution and licenses

Dia2 code and weights: [nari-labs/dia2](https://github.com/nari-labs/dia2), Apache-2.0. Mimi weights: [kyutai/mimi](https://huggingface.co/kyutai/mimi), CC-BY-4.0. This fork retains audio.cpp's upstream license and attribution. See the Hugging Face model card for exact source revisions and validation results.
