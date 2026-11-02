"""LLM extraction with a local instruction model (default Qwen2.5-3B-Instruct, config: extraction.llm_model).

Runs on the GPU, so no API key or account is needed. Greedy decoding (deterministic), JSON-only
answer. The model's raw answer is post-processed: numbers in words are converted, products are
snapped to the menu and prices always come from the menu (the model never invents prices).
"""
import difflib
import gc
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import extract_regex
from config import load_config, menu
from numbers_es import find_numbers
from text_norm import normalize

_LLM = {}

INTERVIEW_PROMPT = """Eres un asistente de selección de personal. Lee la transcripción de una entrevista (puede tener errores de reconocimiento de voz) y extrae los datos del CANDIDATO.

Reglas:
- Si el candidato se corrige ("tres años... no, dos años y medio"), usa el valor corregido final.
- pretension_salarial es lo que PIDE para el nuevo puesto, no lo que gana hoy. Si da un rango y luego un número exacto, usa el número exacto. Número en soles, sin texto.
- anios_experiencia: total de años de experiencia (número; "y medio" = .5).
- ultimo_puesto: solo el cargo, sin la empresa ("Contador senior", no "Contador senior en un banco").
- tecnologias: herramientas, lenguajes o programas que USA (lista, escritos como se escriben normalmente).
- disponibilidad: copia SOLO la frase corta que dice cuándo puede empezar ("en tres semanas", "de inmediato", "a partir del 2 de marzo").
- nivel_ingles: uno de "basico", "intermedio", "avanzado" (C1/C2 = avanzado, B1/B2 = intermedio).
- Si un dato no aparece, usa null. No inventes.

Ejemplo (otra entrevista):
Transcripción: "...Trabajé dos años en una clínica y luego un año y medio en un hospital, en total tres años y medio. Mi último puesto fue enfermera jefe de turno en el Hospital Central. Uso el sistema SIGES y Excel. Hoy gano tres mil, pero busco cuatro mil quinientos soles. Mi inglés es B1. Podría empezar en tres semanas..."
Respuesta: {{"anios_experiencia": 3.5, "ultimo_puesto": "Enfermera jefe de turno", "tecnologias": ["SIGES", "Excel"], "pretension_salarial": 4500, "disponibilidad": "en tres semanas", "nivel_ingles": "intermedio"}}

Responde SOLO con este JSON:
{{"anios_experiencia": número, "ultimo_puesto": "texto", "tecnologias": ["..."], "pretension_salarial": número, "disponibilidad": "frase", "nivel_ingles": "..."}}

Transcripción:
{transcript}"""

CALL_PROMPT = """Eres un asistente de una pizzería. Lee la transcripción de una llamada de pedido (puede tener errores de reconocimiento de voz) y extrae el PEDIDO FINAL.

Menú (usa exactamente estos nombres): {menu}

Reglas:
- Solo el pedido final: si el cliente cambia, quita o reemplaza algo, aplica el cambio. Lo que solo se preguntó y no se pidió NO va.
- Las BEBIDAS también son ítems: revisa la transcripción completa y no olvides ninguna gaseosa ni su cantidad final.
- Un producto de regalo va en items y su valor va como descuento de la promoción.
- descuento: soles descontados por la promoción (0 si no hay). "Segunda pizza a mitad de precio" = mitad del precio de una pizza.
- direccion_entrega: la dirección que da el CLIENTE (no la repetición del vendedor si difiere).
- metodo_pago: uno de "yape", "plin", "efectivo", "tarjeta".
- total_dicho: el total en soles que DICE el vendedor (si se corrige, el último). Cópialo aunque la suma no cuadre.
- Si un dato no aparece, usa null. No inventes.

Ejemplo (otra llamada, otro menú):
Transcripción: "...Quiero dos sánguches de pollo y una limonada. ¿Y los postres cuánto están? Ocho soles. No, mejor no. Ah, y que sean tres limonadas. Hoy el envío es gratis. ¿Dirección? Calle Las Flores ciento veinte. Le repito, Calle Las Flores ciento veintiséis. ¿Cómo paga? Con Plin. Serían sesenta soles..."
Respuesta: {{"items": [{{"producto": "sánguche de pollo", "cantidad": 2}}, {{"producto": "limonada", "cantidad": 3}}], "promocion": {{"descripcion": "envío gratis", "descuento": 0}}, "direccion_entrega": "Calle Las Flores 120", "metodo_pago": "plin", "total_dicho": 60}}

Responde SOLO con este JSON:
{{"items": [{{"producto": "nombre del menú", "cantidad": número}}], "promocion": {{"descripcion": "texto", "descuento": número}}, "direccion_entrega": "texto", "metodo_pago": "...", "total_dicho": número}}

Transcripción:
{transcript}"""


_OVERRIDE = {"model": None}


def set_model(name: str = None) -> None:
    """Use another LLM than config.yaml (experiments compare 3B and 7B)."""
    _OVERRIDE["model"] = name


def _load():
    name = _OVERRIDE["model"] or load_config()["extraction"]["llm_model"]
    if name not in _LLM:
        unload()
        tok = AutoTokenizer.from_pretrained(name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and "7B" in name:  # 7B in bf16 is ~15 GB: load it 4-bit (~5.5 GB) to fit 8 GB
            from transformers import BitsAndBytesConfig
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map="cuda").eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                name, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
        _LLM[name] = (tok, model, device)
    return _LLM[name]


def unload() -> None:
    _LLM.clear()
    gc.collect()  # HF models hold reference cycles: without this the VRAM is not returned
    torch.cuda.empty_cache()


def _generate(prompt: str) -> str:
    tok, model, device = _load()
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=500, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"LLM did not return JSON: {text[:200]}")
    s = m.group(0)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # typical small-model slip: a brace closed too early -> {...}}, "key": ...}  - parse both halves
        obj, end = json.JSONDecoder().raw_decode(s)
        rest = s[end:].strip().lstrip("}").strip()
        if rest.startswith(","):
            obj.update(json.loads("{" + rest[1:]))
        return obj


def _number(v):
    """Accept 6500, "6500", "6.500", "seis mil quinientos"."""
    if v is None or isinstance(v, (int, float)):
        return v
    nums = find_numbers(normalize(str(v)))
    return nums[-1][0] if nums else None


def _snap_product(name: str):
    options = list(menu())
    hit = difflib.get_close_matches(normalize(name or ""), [normalize(o) for o in options], n=1, cutoff=0.5)
    return options[[normalize(o) for o in options].index(hit[0])] if hit else None


def _availability(phrase):
    """The LLM copies the phrase; the date arithmetic is done by code (small LLMs get dates wrong)."""
    if not phrase:
        return None
    phrase = phrase if isinstance(phrase, str) else phrase.get("texto") or ""
    text = normalize(phrase)
    parsed = next(filter(None, (extract_regex.availability(t) for t in (text, "empezar " + text, "empezar en " + text))), None)
    return {"texto": phrase, "fecha": parsed["fecha"] if parsed else None}


def extract_interview(transcript: str) -> dict:
    raw = _parse_json(_generate(INTERVIEW_PROMPT.format(transcript=transcript)))
    level = normalize(str(raw.get("nivel_ingles") or "")) or None
    role = raw.get("ultimo_puesto")
    if role:
        role = re.split(r"\s+en\s+(?:una?|el|la)\s+", role)[0]
    return {"anios_experiencia": _number(raw.get("anios_experiencia")),
            "ultimo_puesto": role,
            "tecnologias": [t for t in raw.get("tecnologias") or [] if t],
            "pretension_salarial": _number(raw.get("pretension_salarial")),
            "disponibilidad": _availability(raw.get("disponibilidad")),
            "nivel_ingles": extract_regex.LEVELS.get(level, level)}


def extract_call(transcript: str) -> dict:
    prices = menu()
    raw = _parse_json(_generate(CALL_PROMPT.format(menu=", ".join(f"{k} (S/ {v})" for k, v in prices.items()),
                                                    transcript=transcript)))
    items = {}
    for it in raw.get("items") or []:
        product, qty = _snap_product(it.get("producto")), _number(it.get("cantidad")) or 1
        if product:
            items[product] = items.get(product, 0) + int(qty)
    promo = raw.get("promocion") or None
    if promo:
        promo = {"descripcion": promo.get("descripcion"), "descuento": _number(promo.get("descuento")) or 0}
        desc = normalize(promo["descripcion"] or "")
        gift = next((p for p in items if ("regalo" in desc or "gratis" in desc) and normalize(p).split()[0] in desc), None)
        if gift and not promo["descuento"]:
            promo["descuento"] = prices[gift]  # a gifted item is discounted at its menu price, decided by code
    pay = normalize(str(raw.get("metodo_pago") or "")) or None
    return {"items": [{"producto": p, "cantidad": q, "precio_unitario": prices[p]} for p, q in items.items() if q > 0],
            "promocion": promo, "direccion_entrega": raw.get("direccion_entrega"),
            "metodo_pago": pay, "total_dicho": _number(raw.get("total_dicho"))}
