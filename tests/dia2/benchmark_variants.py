#!/usr/bin/env python3
"""Compare warm TTS variants; adapted from the existing local tts_bench.py.

Requires an isolated audio.cpp server: it unloads that server's models between
variants. Each row uses identical text/seed and the median of three warm runs.
"""
import argparse
import base64
import io
import json
import statistics
import threading
import time
import urllib.request
import wave
from pathlib import Path


def request(base, path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def speak(base, payload):
    start = time.monotonic()
    with request(base, "/v1/audio/speech", payload) as response:
        body = response.read()
    elapsed = time.monotonic() - start
    with wave.open(io.BytesIO(body), "rb") as wav:
        assert wav.getnchannels() == 1 and wav.getframerate() == 24000
        pcm = wav.readframes(wav.getnframes())
        duration = wav.getnframes() / wav.getframerate()
    return body, pcm, duration, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8194")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--vram-file", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    text = ("Hello there. This is a longer speech test to measure how quickly "
            "Dia two generates audio on the processor. The graphics cards are "
            "busy, so this test uses only system memory.")
    rows = []
    for model in args.models:
        with request(args.base_url, "/v1/tasks/unload_all_models", {}) as response:
            response.read()
        peak = [0]
        stopped = threading.Event()

        def monitor():
            while not stopped.is_set():
                if args.vram_file:
                    peak[0] = max(peak[0], int(args.vram_file.read_text()))
                stopped.wait(0.05)

        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        try:
            payload = {"model": model, "input": "Hello there. This is a speech test.",
                       "seed": 12345, "response_format": "wav"}
            _, _, cold_audio, cold_wall = speak(args.base_url, payload)
            payload["input"] = text
            walls = []
            expected_pcm = None
            for repeat in range(args.repeats):
                body, pcm, duration, elapsed = speak(args.base_url, payload)
                if expected_pcm is not None:
                    assert pcm == expected_pcm, model + ": warm PCM changed"
                expected_pcm = pcm
                walls.append(elapsed)
                (args.output / (model + ".wav")).write_bytes(body)
            stream_payload = dict(payload, stream=True, stream_format="sse",
                                  response_format="pcm")
            start = time.monotonic()
            pieces, first_audio, done = [], None, False
            with request(args.base_url, "/v1/audio/speech", stream_payload) as response:
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("type") == "speech.audio.delta":
                        if first_audio is None:
                            first_audio = time.monotonic() - start
                        pieces.append(base64.b64decode(event["audio"]))
                    elif event.get("type") == "speech.audio.done":
                        done = True
                    elif "error" in event:
                        raise AssertionError(event)
            stream_wall = time.monotonic() - start
            assert done and len(pieces) > 1 and b"".join(pieces) == expected_pcm
            wall = statistics.median(walls)
            row = {"model": model, "audio_seconds": duration,
                   "warm_wall_seconds": wall, "warm_runs_seconds": walls,
                   "seconds_per_10s_audio": wall / duration * 10,
                   "rtf_wall_over_audio": wall / duration,
                   "cold_warmup_seconds": cold_wall, "cold_warmup_audio_seconds": cold_audio,
                   "stream_first_audio_seconds": first_audio,
                   "stream_wall_seconds": stream_wall, "stream_chunks": len(pieces),
                   "stream_full_pcm_identical": True,
                   "observed_peak_total_vram_bytes": peak[0]}
            rows.append(row)
            report = {"backend": "hip", "physical_gpu": 1, "seed": 12345,
                      "voice": "unconditioned", "repeats": args.repeats,
                      "text": text, "results": rows}
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        finally:
            stopped.set()
            watcher.join()
    with request(args.base_url, "/v1/tasks/unload_all_models", {}) as response:
        response.read()


if __name__ == "__main__":
    main()
