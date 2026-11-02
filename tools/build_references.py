"""Semi-automatic reference transcripts: the script, corrected only where two Whisper models agree.

The speakers read the scripts but added words ("claro, claro", "sí, sí") and skipped others, so
the script is not what was said. Starting from the script (normalized with src/text_norm.py):
  - where large-v3 and large-v3-turbo BOTH hear something different from the script, take what
    they heard (two independent models rarely make the same mistake);
  - where they disagree, keep the script and, if the difference is 3+ words, flag it;
  - never auto-change protected words (numbers, technologies, products, payment, address) and
    always flag them: those are the field values, so a person confirms them by listening.
Output: data/references/<clip_id>.txt (one turn per line) and data/references/review.md.
Usage: python tools/build_references.py
"""
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text_norm import normalize, tokens  # noqa: E402

PRIMARY, SECOND = "large-v3", "large-v3-turbo"
RAW = ROOT / "output" / "transcripts" / "raw"
OUT = ROOT / "data" / "references"
PROTECTED = set(tokens(
    "cero un una dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince "
    "dieciseis veinte veinti veinticuatro treinta cuarenta cincuenta sesenta setenta ochenta noventa cien "
    "ciento doscientos trescientos cuatrocientos quinientos ochocientos mil millones medio media "
    "python sql power bi java spring boot docker kubernetes postgresql excel concar sap figma react "
    "typescript ux aws airflow spark dbt c1 basico intermedio avanzado junior senior "
    "yape plin efectivo tarjeta pepperoni hawaiana hawaianas americana americanas alitas brownie brownies "
    "gaseosa gaseosas pan ajo familiar mediana medianas arequipa junin pinos marina olivos santa rosa "
    "manzana lote torre departamento"))


def read_script(clip: str):
    """Script tokens plus the turn index of each token."""
    toks, turn_of, turns = [], [], []
    for line in (ROOT / "data" / "scripts" / f"{clip}.txt").read_text(encoding="utf-8").splitlines():
        speaker, _, text = line.partition(":")
        turns.append(speaker.strip())
        for t in tokens(text):
            toks.append(t)
            turn_of.append(len(turns) - 1)
    return toks, turn_of, turns


def read_hyp(model: str, clip: str):
    """Hypothesis tokens and the start time of the word each token came from."""
    words = json.loads((RAW / model / f"{clip}.json").read_text(encoding="utf-8"))["words"]
    groups = []  # Whisper splits "12.000" into " 12" + ".000": a piece without a leading space joins the previous word
    for w in words:
        if groups and not w["w"].startswith(" "):
            groups[-1][0] += w["w"]
        else:
            groups.append([w["w"], w["start"]])
    toks, times = [], []
    for text, start in groups:
        for t in tokens(text):
            toks.append(t)
            times.append(start)
    return toks, times


def ops(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()


def matched(opcodes, side: int, n: int) -> list:
    """For each token on side 0 (a) or 1 (b), whether it sits in an 'equal' block."""
    flags = [False] * n
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(*((i1, i2) if side == 0 else (j1, j2))):
                flags[k] = True
    return flags


def span_in(opcodes, i1: int, i2: int, b) -> list:
    """Tokens of b aligned to the script span [i1, i2) (for display)."""
    out = []
    for tag, a1, a2, b1, b2 in opcodes:
        if tag == "equal":
            lo, hi = max(a1, i1), min(a2, i2)
            if lo < hi:
                out += b[b1 + lo - a1: b1 + hi - a1]
        elif (a1 < i2 and a2 > i1) or (i1 == i2 and a1 <= i1 <= a2):
            out += b[b1:b2]
    return out


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def build(clip: str):
    script, turn_of, turns = read_script(clip)
    h1, t1 = read_hyp(PRIMARY, clip)
    h2, _ = read_hyp(SECOND, clip)
    s_h1, s_h2, h1_h2 = ops(script, h1), ops(script, h2), ops(h1, h2)
    h1_ok = matched(h1_h2, 0, len(h1))
    s_in_h2 = matched(s_h2, 0, len(script))

    ref = [[] for _ in turns]  # tokens per turn
    flags, applied = [], 0
    for tag, i1, i2, j1, j2 in s_h1:
        s, h = script[i1:i2], h1[j1:j2]
        turn = turn_of[min(i1, len(script) - 1)] if i2 == i1 else turn_of[i1]
        if tag == "equal":
            for k in range(i1, i2):
                ref[turn_of[k]].append(script[k])
            continue
        agreed = all(h1_ok[j1:j2]) if h else not any(s_in_h2[i1:i2])
        protected = bool(PROTECTED & (set(s) | set(h)))
        take = h if agreed and not (protected and s) else s
        if take is h:
            applied += 1
        ref[turn].extend(take)
        if protected or (not agreed and max(len(s), len(h)) >= 3):
            t = t1[j1] if j1 < len(t1) else (t1[-1] if t1 else 0.0)
            flags.append({"time": mmss(t), "script": " ".join(s) or "(nada)", PRIMARY: " ".join(h) or "(nada)",
                          SECOND: " ".join(span_in(s_h2, i1, i2, h2)) or "(nada)",
                          "decision": "se usó lo que oyeron los modelos" if take is h else "se dejó el guion"})
    lines = [f"{sp}: {' '.join(tk)}" for sp, tk in zip(turns, ref)]
    overrides = json.loads((OUT / "overrides.json").read_text(encoding="utf-8")).get(clip, [])
    for o in overrides:  # manual fixes, 1-indexed lines
        lines[o["line"] - 1] = o["text"]
    (OUT / f"{clip}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return flags, applied


def main() -> None:
    lines = ["# Revisión de referencias", "",
             "Escucha cada grabación SOLO en los minutos marcados. Si lo que se dijo no coincide con la",
             f"columna 'decisión', corrige `data/references/<clip>.txt`. Modelos: {PRIMARY} y {SECOND}.", ""]
    for audio in sorted((ROOT / "data" / "audio").glob("*.m4a")):
        flags, applied = build(audio.stem)
        lines += [f"## {audio.stem} — {len(flags)} puntos por revisar ({applied} cambios aceptados solos)", ""]
        if flags:
            lines += [f"| min | guion | {PRIMARY} | {SECOND} | decisión |", "|---|---|---|---|---|"]
            lines += [f"| {f['time']} | {f['script']} | {f[PRIMARY]} | {f[SECOND]} | {f['decision']} |" for f in flags]
        lines.append("")
        print(audio.stem, len(flags), "flags,", applied, "applied")
    (OUT / "review.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
