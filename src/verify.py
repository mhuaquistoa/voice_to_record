"""Literal verification: a field value is kept only if it can be found in the transcript itself.

An extractor (regex or LLM) can produce a value that was never said: an LLM can "fix" a misheard
number, or fill a field from its own knowledge. Rule: every value must be traceable to the words
of the transcript; otherwise it is replaced by SIN_VERIFICAR (the raw value stays available).
  numbers      their spoken form appears ("6500" -> "seis mil quinientos"; 4.5 -> "cuatro ... y medio")
  free text    >= 80 % of its tokens appear in the transcript
  lists        each element verified separately; unverified elements are dropped
  categories   the category word or an accepted synonym appears (C1 -> avanzado, POS -> tarjeta)
"""
from extract_regex import LEVELS, PAYMENTS, PRODUCT_PATTERNS, TECH_ALIASES
from numbers_es import find_numbers
from text_norm import normalize

SIN_VERIFICAR = "sin_verificar"


def _keep_missing(fields: dict, verified: dict) -> dict:
    """A value the extractor did not find stays None ("not said"), which is not the same as unverified."""
    return {k: (None if fields[k] is None or fields[k] == [] else v) for k, v in verified.items()}


def _number_said(value, toks) -> bool:
    if value is None:
        return False
    said = [v for v, _, _ in find_numbers(" ".join(toks))]
    if value in said:
        return True
    if isinstance(value, float) and value - int(value) == 0.5:  # "dos años y medio": 2 said, then "y medio"
        return int(value) in said and "medio" in toks
    return False  # any other decimal (e.g. "85.88" glued from "85... 88") was never said


def _text_said(value, text_padded, toks) -> bool:
    words = normalize(str(value or "")).split()
    if not words:
        return False
    vocab = set(toks)
    return sum(w in vocab for w in words) / len(words) >= 0.8


def _tech_said(term, text_padded) -> bool:
    if f" {normalize(term)} " in text_padded:
        return True
    return any(canon == term and f" {alias} " in text_padded for alias, canon in TECH_ALIASES.items())


def verify_interview(fields: dict, transcript: str) -> dict:
    toks = normalize(transcript).split()
    padded = f" {' '.join(toks)} "
    out = {}
    out["anios_experiencia"] = fields["anios_experiencia"] if _number_said(fields["anios_experiencia"], toks) else SIN_VERIFICAR
    out["pretension_salarial"] = fields["pretension_salarial"] if _number_said(fields["pretension_salarial"], toks) else SIN_VERIFICAR
    out["ultimo_puesto"] = fields["ultimo_puesto"] if _text_said(fields["ultimo_puesto"], padded, toks) else SIN_VERIFICAR
    techs = [t for t in fields["tecnologias"] or [] if _tech_said(t, padded)]
    out["tecnologias"] = techs if techs else SIN_VERIFICAR
    disp = fields["disponibilidad"]
    out["disponibilidad"] = disp if disp and _text_said(disp.get("texto"), padded, toks) else SIN_VERIFICAR
    level = fields["nivel_ingles"]
    out["nivel_ingles"] = level if level and any(LEVELS.get(t) == level for t in toks) else SIN_VERIFICAR
    return _keep_missing(fields, out)


def _product_said(product, text) -> bool:
    import re
    return any(p == product and re.search(r"\b" + pat + r"\b", text) for pat, p in PRODUCT_PATTERNS)


def verify_call(fields: dict, transcript: str) -> dict:
    text = normalize(transcript)
    toks, padded = text.split(), f" {text} "
    out = {}
    items = [i for i in fields["items"] if _product_said(i["producto"], text)]
    out["items"] = items if items else SIN_VERIFICAR
    promo = fields["promocion"]
    out["promocion"] = promo if promo and (promo["descuento"] == 0 or _number_said(promo["descuento"], toks)
                                           or "mitad" in toks or "regalo" in toks or "regalamos" in toks) else SIN_VERIFICAR
    out["direccion_entrega"] = fields["direccion_entrega"] if _text_said(fields["direccion_entrega"], padded, toks) else SIN_VERIFICAR
    pay = fields["metodo_pago"]
    out["metodo_pago"] = pay if pay and any(PAYMENTS.get(t) == pay for t in toks) else SIN_VERIFICAR
    out["total_dicho"] = fields["total_dicho"] if _number_said(fields["total_dicho"], toks) else SIN_VERIFICAR
    return _keep_missing(fields, out)
