"""The single engine shared by the app and every experiment. No UI imports.

    process(audio_path, task) -> {transcript, words, fields, verified, validation, device, elapsed, error}

task is "interview" or "call". Every option defaults to config.yaml and can be overridden per call.
"""
import time
import traceback

import torch

import asr
import extract_llm
import extract_regex
from config import load_config
from validate import validate
from verify import verify_call, verify_interview


def extract(transcript: str, task: str, method: str = None) -> dict:
    method = method or load_config()["extraction"]["method"]
    module = extract_llm if method == "llm" else extract_regex
    return module.extract_interview(transcript) if task == "interview" else module.extract_call(transcript)


def process(audio, task: str, model_name: str = None, device: str = None, compute_type: str = None,
            use_glossary: bool = None, method: str = None, transcript: dict = None) -> dict:
    """audio: path or 16 kHz mono array. A ready `transcript` (asr.transcribe output) skips ASR."""
    if task not in ("interview", "call"):
        raise ValueError("task must be 'interview' or 'call'")
    method = method or load_config()["extraction"]["method"]
    result = {"task": task, "method": method, "transcript": None, "words": [], "fields": None,
              "verified": None, "validation": None, "device": None, "elapsed": {}, "error": None}
    t0 = time.perf_counter()
    try:
        tr = transcript or asr.transcribe(audio, task, model_name, device, compute_type, use_glossary)
        result.update(transcript=tr["text"], words=tr["words"], device=tr["device"],
                      asr={k: tr[k] for k in ("model", "compute_type", "glossary", "audio_seconds", "rtf",
                                              "vram_model_mb", "vram_peak_extra_mb", "gpu_name")})
        result["elapsed"]["asr"] = tr["elapsed_seconds"]
        if method == "llm" and tr["device"] == "cuda" and torch.cuda.mem_get_info()[0] < 7 * 2**30:
            asr.unload_all()  # 8 GB GPU: Whisper large and the 3B LLM do not fit together
        t1 = time.perf_counter()
        fields = extract(tr["text"], task, method)
        result["elapsed"]["extraction"] = round(time.perf_counter() - t1, 2)
        result["fields"] = fields
        verify = verify_interview if task == "interview" else verify_call
        result["verified"] = verify(fields, tr["text"])
        if task == "call":
            result["validation"] = validate(result["verified"], tr["text"], tr["words"])
    except Exception as e:  # the engine reports errors instead of crashing the caller
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    result["elapsed"]["total"] = round(time.perf_counter() - t0, 2)
    return result
