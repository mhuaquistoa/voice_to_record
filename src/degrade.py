"""Degradation ladder. Every level starts from the same 16 kHz mono decode of the original.

L0  original recording (iPhone AAC, 48 kHz stereo -> decoded to 16 kHz mono)
L1  WhatsApp-like voice note: Opus, mono, low bitrate (config: l1_bitrate)
L2  telephone channel: 300-3400 Hz band-pass, resampled to 8 kHz (config: l2_band_hz, l2_sample_rate)
L3  L1 plus white noise at a target SNR; the SNR actually achieved is measured and reported
Usage: python src/degrade.py   ->  data/degraded/<level>/<clip_id>.wav + data/degraded/levels.csv
"""
import io

import av
import numpy as np
import pandas as pd
import soundfile as sf
from faster_whisper.audio import decode_audio
from scipy.signal import butter, resample_poly, sosfiltfilt

from config import ROOT, load_config, path

SR = 16000
OUT = ROOT / "data" / "degraded"


def load(audio_path) -> np.ndarray:
    return decode_audio(str(audio_path), sampling_rate=SR)


def opus_roundtrip(x: np.ndarray, bitrate: str) -> np.ndarray:
    """Encode to Opus mono at `bitrate` and decode back to 16 kHz float32."""
    buf = io.BytesIO()
    with av.open(buf, "w", format="ogg") as out:
        st = out.add_stream("libopus", rate=SR, layout="mono")
        st.bit_rate = int(bitrate.rstrip("k")) * 1000
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)[None, :]
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = SR
        for p in st.encode(frame):
            out.mux(p)
        for p in st.encode(None):
            out.mux(p)
    buf.seek(0)
    y = decode_audio(buf, sampling_rate=SR)
    return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def telephone(x: np.ndarray, band, sr_out: int) -> np.ndarray:
    sos = butter(6, band, btype="bandpass", fs=SR, output="sos")
    return resample_poly(sosfiltfilt(sos, x), sr_out, SR).astype(np.float32)


def speech_power(x: np.ndarray) -> float:
    """Mean power over the louder half of 20 ms frames, so pauses do not dilute the SNR."""
    frames = x[: len(x) // 320 * 320].reshape(-1, 320)
    p = (frames ** 2).mean(axis=1)
    return float(p[p >= np.median(p)].mean())


def add_noise(x: np.ndarray, snr_db: float, seed: int):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(x)).astype(np.float32)
    noise *= np.sqrt(speech_power(x) / (10 ** (snr_db / 10)) / (noise ** 2).mean())
    y = x + noise
    measured = 10 * np.log10(speech_power(x) / (noise ** 2).mean())
    return (y / max(1.0, np.abs(y).max())).astype(np.float32), round(float(measured), 2)


def make_levels(x: np.ndarray, seed: int = 0) -> dict:
    """{level: (samples, sample_rate, measured_snr_db or None)}"""
    cfg = load_config()["degradation"]
    l1 = opus_roundtrip(x, cfg["l1_bitrate"])
    l3, snr = add_noise(l1, cfg["l3_snr_db"], seed)
    return {"L0": (x, SR, None), "L1": (l1, SR, None),
            "L2": (telephone(x, cfg["l2_band_hz"], cfg["l2_sample_rate"]), cfg["l2_sample_rate"], None),
            "L3": (l3, SR, snr)}


def main() -> None:
    seed = load_config()["seed"]
    rows = []
    for i, audio in enumerate(sorted(path("audio").glob("*.m4a"))):
        for level, (y, sr, snr) in make_levels(load(audio), seed + i).items():
            (OUT / level).mkdir(parents=True, exist_ok=True)
            out = OUT / level / f"{audio.stem}.wav"
            sf.write(out, y, sr, subtype="PCM_16")
            rows.append({"clip_id": audio.stem, "level": level, "sample_rate": sr,
                         "seconds": round(len(y) / sr, 2), "measured_snr_db": snr, "file": out.relative_to(ROOT).as_posix()})
    pd.DataFrame(rows).to_csv(OUT / "levels.csv", index=False)
    print(pd.DataFrame(rows).groupby("level")[["sample_rate", "measured_snr_db"]].agg(["min", "max"]))


if __name__ == "__main__":
    main()
