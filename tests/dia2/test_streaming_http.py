#!/usr/bin/env python3
"""Check real incremental PCM delivery, final WAV equivalence, and voice cloning."""
import argparse
import base64
import io
import json
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--url", default="http://127.0.0.1:8197/v1/audio/speech")
parser.add_argument("--model", default="dia2-1b")
parser.add_argument("--voice")
parser.add_argument("--text", default="Hello there. This is a streaming speech test.")
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

def request(payload):
    return urllib.request.Request(args.url, data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})

payload = {"model": args.model, "input": args.text, "seed": args.seed}
payload["options"] = {"dia2.dump_dir": str(args.output.resolve() / "full-dump")}
if args.voice:
    payload["voice"] = args.voice
stream_payload = dict(payload, stream=True, stream_format="sse", response_format="pcm")
stream_payload["options"] = {"dia2.dump_dir": str(args.output.resolve() / "stream-dump")}
pieces, arrival = [], []
start = time.monotonic()
done = False
with urllib.request.urlopen(request(stream_payload), timeout=1800) as response:
    for line in response:
        if not line.startswith(b"data: "):
            continue
        content = line[6:].strip()
        if content == b"[DONE]":
            break
        event = json.loads(content)
        if event.get("type") == "speech.audio.delta":
            pieces.append(base64.b64decode(event["audio"]))
            arrival.append(time.monotonic() - start)
        elif event.get("type") == "speech.audio.done":
            done = True
        elif "error" in event:
            raise AssertionError(event)
elapsed = time.monotonic() - start
assert done and len(pieces) > 1, (done, len(pieces))
assert arrival[0] < elapsed * 0.85, (arrival[0], elapsed)
pcm = b"".join(pieces)
assert len(pcm) % 2 == 0
with wave.open(str(args.output / "stream.wav"), "wb") as out:
    out.setnchannels(1)
    out.setsampwidth(2)
    out.setframerate(24000)
    out.writeframes(pcm)
payload["response_format"] = "wav"
with urllib.request.urlopen(request(payload), timeout=1800) as response:
    wav_bytes = response.read()
(args.output / "full.wav").write_bytes(wav_bytes)
with wave.open(io.BytesIO(wav_bytes)) as full:
    assert full.getframerate() == 24000 and full.getnchannels() == 1
    full_pcm = full.readframes(full.getnframes())
assert len(pcm) == len(full_pcm), (len(pcm), len(full_pcm))
error = np.max(np.abs(np.frombuffer(pcm, "<i2").astype(np.int32) -
                      np.frombuffer(full_pcm, "<i2").astype(np.int32)))
assert error <= 1, error
report = {"text": args.text, "voice": args.voice, "audio_seconds": len(pcm) / 48000,
          "first_audio_seconds": arrival[0], "wall_seconds": elapsed,
          "chunks": len(pieces), "chunk_arrival_seconds": arrival,
          "stream_full_max_pcm_error": int(error)}
(args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
