"""Plain transcription (no glossary) with word timestamps, cached per model.

Used to build the semi-automatic references, so no prompt or glossary is given: the models
should hear the audio, not the script.
Usage: python tools/transcribe_raw.py large-v3 large-v3-turbo
Output: output/transcripts/raw/<model>/<clip_id>.json
"""
import json
import sys
import time
from pathlib import Path

import torch  # noqa: F401  loads the CUDA 12 cuBLAS/cuDNN DLLs bundled with torch
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "data" / "audio"


def main() -> None:
    for name in sys.argv[1:]:
        out_dir = ROOT / "output" / "transcripts" / "raw" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        model = WhisperModel(name, device="cuda", compute_type="float16")
        for audio in sorted(AUDIO.glob("*.m4a")):
            out = out_dir / f"{audio.stem}.json"
            if out.exists():
                continue
            t0 = time.perf_counter()
            segs, info = model.transcribe(str(audio), language="es", beam_size=5, word_timestamps=True)
            words = [{"w": w.word, "start": round(w.start, 2), "end": round(w.end, 2)}
                     for s in segs for w in s.words]
            out.write_text(json.dumps({"model": name, "clip_id": audio.stem, "duration": info.duration,
                                       "seconds": round(time.perf_counter() - t0, 2),
                                       "text": "".join(w["w"] for w in words).strip(), "words": words},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{name} {audio.stem} {time.perf_counter() - t0:.1f}s", flush=True)
        del model


if __name__ == "__main__":
    main()
