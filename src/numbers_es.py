"""Spoken Spanish numbers -> values, on text already passed through text_norm.normalize
(so "6.500", "6500" and "seis mil quinientos" all arrive as words).

find_numbers("son dos años y medio") -> [(2.5, 1, 5)]   (value, first token, last token + 1)
"y" only joins tens and units ("treinta y dos"); "entre nueve y diez mil" stays two numbers,
and so does "ochenta y cinco ochenta y ocho".
"""
UNITS = {"cero": 0, "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
         "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
         "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiun": 21,
         "veintiuna": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
         "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29}
TENS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90}
HUNDREDS = {"cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200, "trescientos": 300,
            "trescientas": 300, "cuatrocientos": 400, "cuatrocientas": 400, "quinientos": 500, "quinientas": 500,
            "seiscientos": 600, "seiscientas": 600, "setecientos": 700, "setecientas": 700, "ochocientos": 800,
            "ochocientas": 800, "novecientos": 900, "novecientas": 900}
SCALES = {"mil": 1000, "millon": 1_000_000, "millones": 1_000_000}
WORDS = set(UNITS) | set(TENS) | set(HUNDREDS) | set(SCALES)


def find_numbers(text: str) -> list:
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        if toks[i] not in WORDS or (toks[i] in ("un", "una") and (i + 1 >= len(toks) or toks[i + 1] not in SCALES)):
            i += 1  # "un"/"una" alone is an article, not the number one
            continue
        start, total, current = i, 0, 0
        while i < len(toks):
            t = toks[i]
            if (t in UNITS or t in TENS) and current % 100 or t in HUNDREDS and current:
                break  # "ochenta y cinco ochenta y ocho" is two numbers (a self-correction), not 173
            if t in UNITS:
                current += UNITS[t]
            elif t in TENS:
                current += TENS[t]
                if i + 2 < len(toks) and toks[i + 1] == "y" and toks[i + 2] in UNITS and UNITS[toks[i + 2]] < 10:
                    current += UNITS[toks[i + 2]]
                    i += 2
            elif t in HUNDREDS:
                current += HUNDREDS[t]
            elif t in SCALES:
                total += max(current, 1) * SCALES[t]
                current = 0
            else:
                break
            i += 1
        value = total + current
        if toks[i:i + 2] == ["y", "medio"] or toks[i:i + 3] in (["anos", "y", "medio"], ["años", "y", "medio"]):
            value += 0.5
            i += 2 if toks[i] == "y" else 3
        out.append((value, start, i))
    return out


def to_words_set(value) -> str:
    """The normalized words a value is spoken as (used for literal verification)."""
    from text_norm import normalize
    if isinstance(value, float) and not value.is_integer():
        return normalize(f"{int(value)}") + " y medio"
    return normalize(str(int(value)))
