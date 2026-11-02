"""The one normalization rule applied to BOTH reference and hypothesis before WER/CER.

1. lowercase
2. digits -> Spanish words ("6.500" -> "seis mil quinientos", "20%" -> "veinte por ciento"),
   so "6.500" and "seis mil quinientos" count as the same words
   (English levels "C1", "B2" stay as one token: "c1", "b2")
3. "uno" -> "un" (the speaker says "ochenta y un soles"; num2words writes "ochenta y uno")
4. accents removed (ñ kept): Whisper is inconsistent with tildes and they carry no field value
5. punctuation removed, whitespace collapsed
Fillers and repetitions are KEPT: the reference is what was actually said.
"""
import re
import unicodedata

from num2words import num2words

_THOUSANDS = re.compile(r"(?<=\d)[.,](?=\d{3}\b)")
_NUMBER = re.compile(r"\d+(?:,\d+)?")


def _number_to_words(m: re.Match) -> str:
    whole, _, frac = m.group(0).partition(",")
    words = num2words(int(whole), lang="es")
    if frac:
        words += " coma " + num2words(int(frac), lang="es")
    return f" {words} "


def _strip_accents(text: str) -> str:
    text = text.replace("ñ", "\0")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text.replace("\0", "ñ")


def normalize(text: str) -> str:
    text = text.lower().replace("%", " por ciento ")
    text = re.sub(r"\b([abc])([12])\b", lambda m: f"cefr{m.group(1)}{'xuno' if m.group(2) == '1' else 'xdos'}", text)
    text = _THOUSANDS.sub("", text)
    text = re.sub(r"(?<=\d)(?=[a-zñ])|(?<=[a-zñ])(?=\d)", " ", text)  # "2085rb" -> "2085 rb"
    text = _NUMBER.sub(_number_to_words, text)
    text = _strip_accents(text)
    text = re.sub(r"[^\w\sñ]", " ", text)
    text = re.sub(r"\buno\b", "un", text)
    text = re.sub(r"\bcefr([abc])x(uno|dos)\b", lambda m: m.group(1) + ("1" if m.group(2) == "uno" else "2"), text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list:
    return normalize(text).split()
