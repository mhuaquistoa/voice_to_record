"""Transcription-layer metrics. Both sides go through text_norm.normalize (the declared rule)."""
import jiwer

from config import path
from text_norm import normalize


def reference(clip_id: str) -> str:
    """Reference text without the 'A:'/'B:' turn labels."""
    lines = (path("references") / f"{clip_id}.txt").read_text(encoding="utf-8").splitlines()
    return " ".join(line.partition(":")[2].strip() for line in lines if line.strip())


def wer_cer(ref: str, hyp: str) -> dict:
    r, h = normalize(ref), normalize(hyp)
    m = jiwer.process_words(r, h)
    return {"wer": m.wer, "cer": jiwer.cer(r, h), "substitutions": m.substitutions,
            "deletions": m.deletions, "insertions": m.insertions, "ref_words": len(r.split())}


def term_recall(ref: str, hyp: str, terms: list) -> float | None:
    """Share of glossary terms present in the reference that are also in the hypothesis."""
    r, h = f" {normalize(ref)} ", f" {normalize(hyp)} "
    present = [t for t in terms if f" {normalize(t)} " in r]
    if not present:
        return None
    return sum(f" {normalize(t)} " in h for t in present) / len(present)
