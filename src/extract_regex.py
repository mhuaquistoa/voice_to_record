"""Rule-based extraction (the baseline the LLM must beat). Works on the normalized transcript.

Each rule is written for the phrasing a person actually uses, not for these ten clips' wording:
keyword windows, "the last value wins" for self-corrections, and exclusion of current salary.
"""
import re
from datetime import date, timedelta

from config import load_config, menu
from numbers_es import find_numbers
from text_norm import normalize

LEVELS = {"basico": "basico", "intermedio": "intermedio", "avanzado": "avanzado", "fluido": "avanzado",
          "nativo": "avanzado", "a1": "basico", "a2": "basico", "b1": "intermedio", "b2": "intermedio",
          "c1": "avanzado", "c2": "avanzado"}
TECH_ALIASES = {"amazon web services": "AWS", "postgres": "PostgreSQL", "postgre sql": "PostgreSQL",
                "power bi": "Power BI", "spring boot": "Spring Boot"}
NOT_TECH = {"UX"}  # in the ASR glossary (a hard word to hear) but it is a discipline, not a tool
MONTHS = {m: i + 1 for i, m in enumerate("enero febrero marzo abril mayo junio julio agosto setiembre octubre noviembre diciembre".split())}
MONTHS["septiembre"] = 9


def _window(toks, i, before, after):
    return toks[max(0, i - before): i + after]


def _ref_date() -> date:
    return date.fromisoformat(load_config()["extraction"]["reference_date"])


def _add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    return date(d.year + m // 12, m % 12 + 1, 1)


# ---------------------------------------------------------------- interview
def years_experience(toks):
    cands = []
    for value, a, b in find_numbers(" ".join(toks)):
        after = toks[b:b + 1]
        if after and after[0] in ("anos", "años", "ano", "año") or toks[b - 1] == "medio":
            ctx = _window(toks, a, 6, 8)
            score = any(k in ctx for k in ("experiencia", "total", "llevo", "trabajando", "son"))
            cands.append((score, a, value))
    if not cands:
        return None
    scored = [c for c in cands if c[0]] or cands
    return scored[-1][2]  # last keyword-backed mention wins: "casi tres... no, dos años y medio"


def salary(toks):
    keys = ("expectativa", "pretension", "esperas", "buscando", "pensando", "diria", "exacto", "pido", "ganar")
    cands = []
    for value, a, b in find_numbers(" ".join(toks)):
        if value < 500 or 1900 <= value <= 2100 and "en" in toks[max(0, a - 2):a]:
            continue  # small numbers and years ("en dos mil veinte")
        before = toks[max(0, a - 8):a]
        if "gano" in before and not any(k in before for k in keys):
            continue  # current salary, not the expectation
        if "soles" in toks[b:b + 2] or any(k in before for k in keys):
            cands.append(value)
    return cands[-1] if cands else None


def last_role(text):
    m = re.search(r"(?:ultimo|actual) puesto (?:es|fue|era) (?:de |como )?(.+?)(?= en (?:una|un|el|la)\b| y\b| me\b| somos\b|$)", text)
    if not m:
        return None
    return " ".join(m.group(1).split()[:6])


def technologies(text):
    found, padded = [], f" {text} "
    for alias, canonical in TECH_ALIASES.items():
        if f" {alias} " in padded and canonical not in found:
            found.append(canonical)
    for term in load_config()["glossary"]["interview"]:
        if term in NOT_TECH:
            continue
        if f" {normalize(term)} " in padded and term not in found:
            found.append(term)
    return found


def availability(text):
    ref = _ref_date()
    m = re.search(r"\b(de inmediato|inmediatamente|disponibilidad inmediata|desde ya|desde manana)\b", text)
    if m:
        return {"texto": m.group(0), "fecha": ref.isoformat()}
    m = re.search(r"a partir del? (\w+(?: y \w+)?) de (" + "|".join(MONTHS) + r")", text)
    if m:
        day = [v for v, _, _ in find_numbers(m.group(1))]
        if day:
            d = date(ref.year, MONTHS[m.group(2)], int(day[0]))
            d = d if d >= ref else date(ref.year + 1, d.month, d.day)
            return {"texto": m.group(0), "fecha": d.isoformat()}
    m = re.search(r"(?:inicios?|principios?) del? (?:proximo|siguiente) mes|el (?:proximo|siguiente) mes", text)
    if m:
        return {"texto": m.group(0), "fecha": _add_months(ref, 1).isoformat()}
    m = re.search(r"(?:empezar|incorporarme|disponible|comenzar)\w* en (un|una|\w+) (semanas?|mes|meses|dias?)", text)
    if m:
        n = int(([v for v, _, _ in find_numbers(m.group(1))] or [1])[0])
        unit = m.group(2)
        if unit.startswith("semana"):
            d = ref + timedelta(weeks=n)
        elif unit.startswith("dia"):
            d = ref + timedelta(days=n)
        else:
            k = ref.month - 1 + n
            d = date(ref.year + k // 12, k % 12 + 1, min(ref.day, 28))
        return {"texto": m.group(0), "fecha": d.isoformat()}
    return None


def english(toks):
    pos = [i for i, t in enumerate(toks) if t == "ingles"]
    best = None
    for i, t in enumerate(toks):
        if t in LEVELS and pos:
            dist = min(abs(i - p) for p in pos)
            if dist <= 25 and (best is None or dist < best[0]):
                best = (dist, LEVELS[t])
    return best[1] if best else None


def extract_interview(transcript: str) -> dict:
    text = normalize(transcript)
    toks = text.split()
    return {"anios_experiencia": years_experience(toks), "ultimo_puesto": last_role(text),
            "tecnologias": technologies(text), "pretension_salarial": salary(toks),
            "disponibilidad": availability(text), "nivel_ingles": english(toks)}


# ---------------------------------------------------------------- call
PRODUCT_PATTERNS = [
    (r"americanas? medianas?", "americana mediana"), (r"americanas? familiar(?:es)?", "americana familiar"),
    (r"pep+eroni familiar(?:es)?", "pepperoni familiar"), (r"hawaianas? medianas?", "hawaiana mediana"),
    (r"gaseosas?", "gaseosa 1.5 L"), (r"pan al ajo", "pan al ajo"), (r"alitas", "alitas x6"), (r"brownies?", "brownie"),
]
REMOVE = ("quita", "quitame", "quito", "retiro", "retira", "sin", "cambia", "cambiame", "saca")
ARTICLES_ONE = ("un", "una", "unas", "unos")
PAYMENTS = {"yape": "yape", "plin": "plin", "efectivo": "efectivo", "tarjeta": "tarjeta", "pos": "tarjeta"}


def _product_events(text):
    events = []
    for pattern, product in PRODUCT_PATTERNS:
        for m in re.finditer(r"\b" + pattern + r"\b", text):
            events.append((m.start(), m.end(), product))
    events.sort()
    return [e for i, e in enumerate(events) if not any(o[0] <= e[0] < o[1] for o in events[:i])]


def order_items(text):
    prices, qty = menu(), {}
    for start, _, product in _product_events(text):
        before = text[:start].split()[-5:]
        if any(w in REMOVE for w in before[-4:]) and not ("por" in before and before.index("por") > max(
                i for i, w in enumerate(before) if w in REMOVE)):
            qty[product] = 0
            continue
        nums = [v for v, a, b in find_numbers(" ".join(before)) if b >= len(before) - 1 and v < 20]
        if nums:
            qty[product] = int(nums[-1])
        elif before and before[-1] in ARTICLES_ONE or before[-2:-1] and before[-2] in ARTICLES_ONE and before[-1] == "pizza":
            qty[product] = 1
    return [{"producto": p, "cantidad": q, "precio_unitario": prices[p]} for p, q in qty.items() if q > 0]


def promotion(text, items):
    prices = menu()
    m = re.search(r"(\w+(?: y \w+)?) soles de descuento", text)
    if m and find_numbers(m.group(1)):
        return {"descripcion": m.group(0), "descuento": find_numbers(m.group(1))[-1][0]}
    if "mitad de precio" in text:
        pizzas = sorted(i["precio_unitario"] for i in items for _ in range(i["cantidad"]) if "gaseosa" not in i["producto"]
                        and i["producto"] not in ("pan al ajo", "alitas x6", "brownie"))
        return {"descripcion": "segunda pizza a mitad de precio", "descuento": pizzas[0] / 2 if len(pizzas) >= 2 else 0}
    m = re.search(r"(?:regalamos|de regalo|gratis) (?:un |una )?(pan al ajo|gaseosa|brownie)", text) or \
        re.search(r"(pan al ajo|gaseosa|brownie) de regalo", text)
    if m:
        key = {"gaseosa": "gaseosa 1.5 L"}.get(m.group(1), m.group(1))
        return {"descripcion": f"{m.group(1)} de regalo", "descuento": prices[key]}
    if re.search(r"delivery (?:es )?gratis|envio gratis", text):
        return {"descripcion": "delivery gratis", "descuento": 0}
    return None


def address(text):
    start = r"\b(avenida|av|jiron|giron|jr|calle|urbanizacion|pasaje)\b"  # "girón" is how ASR spells "jirón"
    m = re.search(start + r"(.+?)(?= le repito| alguna| algun| es una?\b| hay que| correcto| perfecto| ok\b|"
                  r" (?:avenida|calle|jiron|giron|urbanizacion)\b|$)", text)
    return (m.group(1) + m.group(2)).strip() if m else None


def payment(toks):
    for i, t in enumerate(toks):
        if t in ("pagar", "pago", "paga", "pagas"):
            for u in toks[i + 1:i + 12]:
                if u in PAYMENTS:
                    return PAYMENTS[u]
    return next((PAYMENTS[t] for t in toks if t in PAYMENTS), None)


def total_said(toks):
    best = None
    for i, t in enumerate(toks):
        if t == "total":
            ahead = toks[i:i + 15]
            ahead = ahead[:ahead.index("soles")] if "soles" in ahead else ahead[:10]
            nums = find_numbers(" ".join(toks[max(0, i - 6):i] + ahead))
            if nums:
                best = nums[-1][0]  # "ochenta y cinco... ochenta y ocho": the last value wins
    return best


def extract_call(transcript: str) -> dict:
    text = normalize(transcript)
    toks = text.split()
    items = order_items(text)
    return {"items": items, "promocion": promotion(text, items), "direccion_entrega": address(text),
            "metodo_pago": payment(toks), "total_dicho": total_said(toks)}
