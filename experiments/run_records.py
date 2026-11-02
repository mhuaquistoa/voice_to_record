"""Record-layer experiment: transcripts -> engine (extract + verify + validate) -> score.

Inputs are the cached ladder transcripts (run_asr.py ladder) plus the reference text itself
("oracle": a perfect transcript), which separates extraction errors from transcription errors.
    python experiments/run_records.py [regex] [llm-3b] [llm-7b]     (default: all three)
Scores are always computed over every method already cached.
Output: output/records/<run>/<level>/<method>/<clip>.json,
        output/metrics/records.csv (one row per clip x level x method)
        output/metrics/validation.csv (one row per call x level x method)
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import engine  # noqa: E402
from config import ground_truth, load_config  # noqa: E402
from metrics import reference  # noqa: E402
from run_asr import clips, run_name  # noqa: E402
from scoring import CALL_FIELDS, INTERVIEW_FIELDS, score_call, score_interview, validation_outcome  # noqa: E402
from validate import validate  # noqa: E402
from verify import verify_call, verify_interview  # noqa: E402

OUT = ROOT / "output"
METHODS = {"regex": None, "llm-3b": "Qwen/Qwen2.5-3B-Instruct", "llm-7b": "Qwen/Qwen2.5-7B-Instruct"}  # label -> LLM


def inputs():
    a = load_config()["asr"]
    name = run_name(a["model"], a["device"], a["compute_type"], a["use_glossary"])
    for clip in clips():
        yield "oracle", "reference", clip, {"text": reference(clip), "words": [], "device": "n/a", "elapsed_seconds": 0,
                                            "model": "reference", "compute_type": None, "glossary": None,
                                            "audio_seconds": None, "rtf": None, "vram_model_mb": 0,
                                            "vram_peak_extra_mb": 0, "gpu_name": None}
    for level in load_config()["degradation"]["levels"]:
        for clip in clips():
            f = OUT / "transcripts" / name / level / f"{clip}.json"
            if f.exists():
                yield name, level, clip, json.loads(f.read_text(encoding="utf-8"))


def main() -> None:
    gold = {**ground_truth("interview"), **ground_truth("call")}
    rows, vrows = [], []
    only = sys.argv[1:] or list(METHODS)
    for method in METHODS:
        engine.extract_llm.set_model(METHODS[method])
        for run, level, clip, tr in inputs():
            task = clip.split("_")[0]
            out = OUT / "records" / run / level / method / f"{clip}.json"
            if out.exists():
                res = json.loads(out.read_text(encoding="utf-8"))
                if not res["error"]:  # verification and validation are cheap rules: always recompute them
                    res["verified"] = (verify_interview if task == "interview" else verify_call)(res["fields"], tr["text"])
                    if task == "call":
                        res["validation"] = validate(res["verified"], tr["text"], tr["words"])
            elif method not in only:
                continue
            else:
                res = engine.process(None, task, method="regex" if method == "regex" else "llm", transcript=tr)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({k: v for k, v in res.items() if k != "words"}, ensure_ascii=False, indent=1,
                                          default=str), encoding="utf-8")
                print(f"{method} {run} {level} {clip} {res['elapsed']} {res['error'] or ''}", flush=True)
            if res["error"]:
                rows.append({"run": run, "level": level, "method": method, "clip_id": clip, "task": task, "error": res["error"]})
                continue
            fields = INTERVIEW_FIELDS if task == "interview" else CALL_FIELDS
            s = (score_interview if task == "interview" else score_call)(res["verified"], gold[clip])
            raw = (score_interview if task == "interview" else score_call)(res["fields"], gold[clip])
            unverified = sum(res["verified"][f] == "sin_verificar" for f in fields)
            rows.append({"run": run, "level": level, "method": method, "clip_id": clip, "task": task,
                         "fields_correct": sum(bool(s[f]) for f in fields), "fields_total": len(fields),
                         "fields_correct_before_verification": sum(bool(raw[f]) for f in fields),
                         "fields_unverified": unverified, "extraction_seconds": res["elapsed"].get("extraction"),
                         **{f"ok_{k}": v for k, v in s.items()}})
            if task == "call":
                val = res["validation"]
                vrows.append({"run": run, "level": level, "method": method, "clip_id": clip,
                              "total_correcto_gold": gold[clip]["total_correcto"], "estado": val["estado"],
                              "diagnostico": val["diagnostico"], "total_calculado": val["total_calculado"],
                              "total_dicho": val["total_dicho"], "confianza_total_asr": val["confianza_total_asr"],
                              "outcome": validation_outcome(val, gold[clip]),
                              "precios_distintos": json.dumps(val["precios_distintos_al_menu"], ensure_ascii=False)})
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "metrics" / "records.csv", index=False)
    pd.DataFrame(vrows).to_csv(OUT / "metrics" / "validation.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["method", "level"])[["fields_correct", "fields_total"]].sum())


if __name__ == "__main__":
    main()
