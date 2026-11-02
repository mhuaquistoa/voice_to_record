"""Whisper transcription (faster-whisper / CTranslate2).

The device is read back at runtime: if CUDA was requested but fails, the run falls back to CPU and
says so in the result instead of pretending it ran on the GPU.
"""
import gc
import threading
import time

import torch  # noqa: F401  loads the CUDA 12 cuBLAS/cuDNN DLLs bundled with torch (Windows)
import ctranslate2
from faster_whisper import WhisperModel

from config import load_config

_MODELS = {}


def _free_vram() -> int:
    if ctranslate2.get_cuda_device_count() == 0:
        return 0
    return torch.cuda.mem_get_info()[0]


def get_model(name: str, device: str, compute_type: str):
    """Load (and cache) a model. Returns (model, device_actually_used, load_info)."""
    key = (name, device, compute_type)
    if key in _MODELS:
        return _MODELS[key]
    if device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
        device, compute_type = "cpu", "int8"
    free0, t0 = _free_vram(), time.perf_counter()
    try:
        model = WhisperModel(name, device=device, compute_type=compute_type,
                             cpu_threads=0 if device == "cuda" else 8)
    except (RuntimeError, ValueError):
        if device != "cuda":
            raise
        device, compute_type = "cpu", "int8"
        model = WhisperModel(name, device=device, compute_type=compute_type, cpu_threads=8)
    info = {"load_seconds": round(time.perf_counter() - t0, 2),
            "vram_model_mb": round((free0 - _free_vram()) / 2**20) if device == "cuda" else 0}
    loaded = (model, device, compute_type, info)
    if device == key[1]:  # never cache a CPU fallback as the GPU model: the next call retries the GPU
        _MODELS[key] = loaded
    return loaded


def unload_all() -> None:
    _MODELS.clear()
    gc.collect()
    torch.cuda.empty_cache()


def glossary_prompt(task: str) -> str:
    terms = load_config()["glossary"][task]
    return "Términos que pueden aparecer: " + ", ".join(terms) + "."


def transcribe(audio, task: str, model_name: str = None, device: str = None, compute_type: str = None,
               use_glossary: bool = None) -> dict:
    """audio: file path or float32 mono 16 kHz array."""
    asr = load_config()["asr"]
    model_name = model_name or asr["model"]
    device = device or asr["device"]
    compute_type = compute_type or asr["compute_type"]
    use_glossary = asr["use_glossary"] if use_glossary is None else use_glossary
    model, used_device, used_compute, load_info = get_model(model_name, device, compute_type)
    prompt = glossary_prompt(task) if use_glossary else None

    free0, lowest = _free_vram(), [_free_vram()]
    stop = threading.Event()

    def watch():  # CTranslate2 allocates outside torch, so sample free memory to get the peak
        while not stop.wait(0.05):
            lowest[0] = min(lowest[0], _free_vram())

    watcher = threading.Thread(target=watch, daemon=True)
    if used_device == "cuda":
        watcher.start()
    t0 = time.perf_counter()
    try:
        segs, info = model.transcribe(audio, language=asr["language"], beam_size=asr["beam_size"],
                                      initial_prompt=prompt, word_timestamps=True,
                                      condition_on_previous_text=False, vad_filter=False)
        words = [{"w": w.word, "start": round(w.start, 2), "end": round(w.end, 2), "p": round(w.probability, 3)}
                 for s in segs for w in s.words]  # segments are lazy: decoding happens here
    finally:
        elapsed = time.perf_counter() - t0
        stop.set()
    return {
        "text": "".join(w["w"] for w in words).strip(),
        "words": words,
        "model": model_name,
        "device": used_device,
        "compute_type": used_compute,
        "glossary": use_glossary,
        "audio_seconds": round(info.duration, 2),
        "elapsed_seconds": round(elapsed, 2),
        "rtf": round(elapsed / max(info.duration, 1e-6), 4),
        "vram_model_mb": load_info["vram_model_mb"],
        "vram_peak_extra_mb": round(max(0, free0 - lowest[0]) / 2**20) if used_device == "cuda" else 0,
        "gpu_name": torch.cuda.get_device_name(0) if used_device == "cuda" else None,
    }
