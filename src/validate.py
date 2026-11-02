"""Arithmetic validation of an order and the diagnosis rule.

total_calculado = sum(cantidad * precio del menú) - descuento. If it differs from total_dicho the
order is flagged. The diagnosis says WHO probably made the mistake:
  "error_asr"       the evidence is weak: the spoken total failed literal verification, or Whisper
                    was unsure of the words that carry it (mean word probability < threshold), or an
                    item could not be verified. A human should listen before charging.
  "error_hablante"  the transcript is confident and complete, yet the arithmetic does not close:
                    the seller (or customer) said a wrong total. Charge the calculated total.
It also lists prices said in the call that differ from the menu ("pepperoni familiar 45").
"""
from config import menu
from extract_regex import PRODUCT_PATTERNS
from numbers_es import find_numbers
from text_norm import normalize
from verify import SIN_VERIFICAR

CONFIDENCE_THRESHOLD = 0.6


def total_confidence(words: list) -> float | None:
    """Mean Whisper probability of the number words right after the last 'total'."""
    if not words:
        return None
    norm = [normalize(w["w"]) for w in words]
    idx = [i for i, w in enumerate(norm) if w == "total"]
    if not idx:
        return None
    def number_probs(span, stop_at_soles=True):
        probs = []
        for w, n in span:
            if n in ("soles", "en") and not stop_at_soles:
                continue
            if n == "soles":
                break
            if find_numbers(n) or n == "y" or any(ch.isdigit() for ch in w["w"]):
                probs.append(w["p"])
            elif probs:
                break
        return probs

    i = idx[-1]
    probs = number_probs(zip(words[i + 1:i + 10], norm[i + 1:i + 10]))
    if not probs:  # "serían ochenta y dos soles en total": the number comes before
        probs = number_probs(reversed(list(zip(words[max(0, i - 8):i], norm[max(0, i - 8):i]))), stop_at_soles=False)
    return round(sum(probs) / len(probs), 3) if probs else None


def said_prices(transcript: str) -> list:
    """Prices spoken right after a product name that differ from the menu."""
    import re
    text, prices, out = normalize(transcript), menu(), []
    for pat, product in PRODUCT_PATTERNS:
        for m in re.finditer(r"\b" + pat + r"\b", text):
            after = text[m.end(): m.end() + 60].split()[:6]
            first_number = find_numbers(" ".join(after))
            direct = first_number and first_number[0][1] == 0 and not any(  # "dos brownies" is a quantity
                re.match(pat, " ".join(after[first_number[0][2]:])) for pat, _ in PRODUCT_PATTERNS)
            if after[:1] and (after[0] in ("esta", "es", "a", "le", "sale", "les") or direct):
                nums = [v for v, a, _ in first_number if a <= 3 and v < 200]
                if nums and nums[0] != prices[product]:
                    out.append({"producto": product, "dicho": nums[0], "menu": prices[product]})
    return out


def validate(verified: dict, transcript: str, words: list = None) -> dict:
    items = verified["items"] if verified["items"] not in (SIN_VERIFICAR, None) else []
    promo = verified["promocion"] if verified["promocion"] != SIN_VERIFICAR else None
    prices = menu()
    subtotal = sum(i["cantidad"] * prices[i["producto"]] for i in items)
    calculated = subtotal - (promo["descuento"] if promo else 0)
    said = verified["total_dicho"]
    conf = total_confidence(words)
    result = {"subtotal": subtotal, "descuento": promo["descuento"] if promo else 0, "total_calculado": calculated,
              "total_dicho": said, "confianza_total_asr": conf, "precios_distintos_al_menu": said_prices(transcript)}
    if said == SIN_VERIFICAR or said is None:
        result.update(estado="sin_total", diagnostico="error_asr",
                      motivo="El total dicho no se pudo leer en la transcripción.")
    elif abs(said - calculated) < 0.01:
        result.update(estado="cuadra", diagnostico=None, motivo="La suma coincide con el total dicho.")
    else:
        weak = (conf is not None and conf < CONFIDENCE_THRESHOLD) or verified["items"] == SIN_VERIFICAR \
            or verified["promocion"] == SIN_VERIFICAR
        result.update(estado="no_cuadra", diagnostico="error_asr" if weak else "error_hablante",
                      motivo=(f"Whisper no está seguro del total (confianza {conf}) o falta verificar ítems: escuchar el audio."
                              if weak else f"La transcripción es confiable: se dijo {said} pero la suma da {calculated}."))
    return result
