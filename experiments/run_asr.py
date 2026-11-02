"""Transcription experiments. Every run is cached, so re-running only fills what is missing.

    python experiments/run_asr.py benchmark      # models x device x precision x glossary on L0
    python experiments/run_asr.py ladder         # chosen config on L0..L3
Output: output/transcripts/<run>/<level>/<clip>.json  and  output/metrics/asr_runs.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import asr  # noqa: E402
from config import load_config, path  # noqa: E402
from degrade import SR, load  # noqa: E402
from metrics import reference, term_recall, wer_cer  # noqa: E402

OUT = ROOT / "output"
BENCHMARK = [  # (model, device, compute_type, glossary)
    ("tiny", "cuda", "float16", False), ("base", "cuda", "float16", False), ("small", "cuda", "float16", False),
    ("medium", "cuda", "float16", False), ("large-v3-turbo", "cuda", "float16", False),
    ("large-v3", "cuda", "float16", False), ("large-v3", "cuda", "int8_float16", False),
    ("large-v3", "cuda", "int8", False), ("large-v3", "cuda", "float16", True),
    ("large-v3-turbo", "cuda", "float16", True), ("small", "cpu", "int8", False),
    ("large-v3-turbo", "cpu", "int8", False), ("large-v3", "cpu", "int8", False),
]


def run_name(model, device, compute, glossary) -> str:
    return f"{model}_{device}_{compute}" + ("_glossary" if glossary else "")


def clips():
    return sorted(p.stem for p in path("audio").glob("*.m4a"))


def audio_for(clip: str, level: str):
    if level == "L0":
        return load(path("audio") / f"{clip}.m4a")
    return load(ROOT / "data" / "degraded" / level / f"{clip}.wav")  # decoded back to 16 kHz


def run(model, device, compute, glossary, level="L0") -> list:
    name = run_name(model, device, compute, glossary)
    out_dir = OUT / "transcripts" / name / level
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in clips() if not (out_dir / f"{c}.json").exists()]
    if todo:
        warm = audio_for(todo[0], level)[: 5 * SR]  # warm-up so the first clip's time is not load time
        asr.transcribe(warm, "interview", model, device, compute, glossary)
    for clip in todo:
        task = clip.split("_")[0]
        tr = asr.transcribe(audio_for(clip, level), task, model, device, compute, glossary)
        tr.update(clip_id=clip, level=level, run=name)
        (out_dir / f"{clip}.json").write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name} {level} {clip} {tr['elapsed_seconds']}s on {tr['device']}", flush=True)
    asr.unload_all()
    rows = []
    for clip in clips():
        tr = json.loads((out_dir / f"{clip}.json").read_text(encoding="utf-8"))
        ref = reference(clip)
        terms = load_config()["glossary"][clip.split("_")[0]]
        rows.append({"run": name, "model": model, "device_requested": device, "device": tr["device"],
                     "compute_type": tr["compute_type"], "glossary": glossary, "level": level, "clip_id": clip,
                     "task": clip.split("_")[0], **wer_cer(ref, tr["text"]),
                     "term_recall": term_recall(ref, tr["text"], terms), "elapsed_seconds": tr["elapsed_seconds"],
                     "audio_seconds": tr["audio_seconds"], "rtf": tr["rtf"], "vram_model_mb": tr["vram_model_mb"],
                     "vram_peak_extra_mb": tr["vram_peak_extra_mb"], "gpu_name": tr["gpu_name"]})
    return rows


def save(rows):
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    f = OUT / "metrics" / "asr_runs.csv"
    new = pd.DataFrame(rows)
    if f.exists():
        old = pd.read_csv(f)
        old = old[~old.set_index(["run", "level", "clip_id"]).index.isin(new.set_index(["run", "level", "clip_id"]).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(f, index=False)


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "benchmark"
    if what == "benchmark":
        only = sys.argv[2:]  # optional filter by device, e.g. "cuda"
        for cfg in BENCHMARK:
            if not only or cfg[1] in only:
                save(run(*cfg))
    elif what == "ladder":
        a = load_config()["asr"]
        for level in load_config()["degradation"]["levels"]:
            save(run(a["model"], a["device"], a["compute_type"], a["use_glossary"], level))


if __name__ == "__main__":
    main()
