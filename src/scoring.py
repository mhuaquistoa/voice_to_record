"""Record-layer metrics: each extracted field against the ground truth.

A field marked sin_verificar counts as NOT correct (the record a user receives has no value).
  numbers            exact match; absolute error reported separately
  ultimo_puesto      string similarity >= 0.8 after normalization ("Ingeniero" vs "Ingeniera" passes)
  tecnologias        set F1 over canonical names
  disponibilidad     the resolved date equals the gold date
  nivel_ingles, pago exact category
  items              exact multiset of (product, quantity); item F1 reported too
  direccion          the street name and every number of the gold address, and no other number
"""
import difflib

from extract_regex import TECH_ALIASES
from numbers_es import find_numbers
from text_norm import normalize
from verify import SIN_VERIFICAR

INTERVIEW_FIELDS = ["anios_experiencia", "ultimo_puesto", "tecnologias", "pretension_salarial", "disponibilidad", "nivel_ingles"]
CALL_FIELDS = ["items", "promocion", "direccion_entrega", "metodo_pago", "total_dicho"]
CANON = {normalize(k): v for k, v in TECH_ALIASES.items()}
STREET_WORDS = {"avenida", "jiron", "calle", "urbanizacion", "dpto", "departamento", "torre", "manzana", "lote",
                "santa", "cercado"}


def _canon_tech(t: str) -> str:
    n = normalize(t)
    return normalize(CANON.get(n, t))


def _f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _address_ok(pred, gold) -> bool:
    if not pred:
        return False
    p, g = normalize(str(pred)).replace("giron", "jiron"), normalize(gold)
    pn = [v for v, _, _ in find_numbers(p)]
    gn = [v for v, _, _ in find_numbers(g)]
    street = next(w for w in g.split() if len(w) > 3 and w not in STREET_WORDS and not find_numbers(w))
    return sorted(set(pn)) == sorted(set(gn)) and street in p.split()


def score_interview(rec: dict, gold: dict) -> dict:
    s = {}
    for f in ("anios_experiencia", "pretension_salarial"):
        v = rec[f]
        s[f] = v != SIN_VERIFICAR and v is not None and abs(float(v) - gold[f]) < 1e-6
        s[f + "_abs_error"] = abs(float(v) - gold[f]) if isinstance(v, (int, float)) else None
    role = rec["ultimo_puesto"]
    s["ultimo_puesto"] = role not in (None, SIN_VERIFICAR) and difflib.SequenceMatcher(
        None, normalize(role), normalize(gold["ultimo_puesto"])).ratio() >= 0.8
    techs = rec["tecnologias"] if isinstance(rec["tecnologias"], list) else []
    s["tecnologias_f1"] = _f1({_canon_tech(t) for t in techs}, {_canon_tech(t) for t in gold["tecnologias"]})
    s["tecnologias"] = s["tecnologias_f1"] == 1.0
    d = rec["disponibilidad"]
    s["disponibilidad"] = isinstance(d, dict) and d.get("fecha") == gold["disponibilidad"]["fecha"]
    s["nivel_ingles"] = rec["nivel_ingles"] == gold["nivel_ingles"]
    return s


def score_call(rec: dict, gold: dict) -> dict:
    s = {}
    items = rec["items"] if isinstance(rec["items"], list) else []
    pred = {(i["producto"], i["cantidad"]) for i in items}
    gset = {(i["producto"], i["cantidad"]) for i in gold["items"]}
    s["items_f1"] = _f1(pred, gset)
    s["items"] = pred == gset
    promo = rec["promocion"]
    s["promocion"] = promo not in (None, SIN_VERIFICAR) and abs(promo["descuento"] - gold["promocion"]["descuento"]) < 0.01
    s["direccion_entrega"] = rec["direccion_entrega"] != SIN_VERIFICAR and _address_ok(rec["direccion_entrega"], gold["direccion_entrega"])
    s["metodo_pago"] = rec["metodo_pago"] == gold["metodo_pago"]
    t = rec["total_dicho"]
    s["total_dicho"] = isinstance(t, (int, float)) and abs(t - gold["total_dicho"]) < 0.01
    s["total_dicho_abs_error"] = abs(t - gold["total_dicho"]) if isinstance(t, (int, float)) else None
    return s


def validation_outcome(validation: dict, gold: dict) -> str:
    """caught / missed / false_alarm / correct_pass for the arithmetic check."""
    flagged = validation["estado"] != "cuadra"
    if not gold["total_correcto"]:
        return "caught" if flagged else "missed"
    return "false_alarm" if flagged else "correct_pass"
