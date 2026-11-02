"""From Voice to Record - Streamlit front end. All the work is done by src/engine.py.
Run: streamlit run app.py
"""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import engine  # noqa: E402
from config import ground_truth, load_config, path  # noqa: E402
from degrade import SR, load, make_levels  # noqa: E402
from verify import SIN_VERIFICAR  # noqa: E402

st.set_page_config(page_title="From Voice to Record", page_icon="🎙️", layout="wide")
cfg = load_config()
TABLES, FIGURES = ROOT / "output" / "tables", ROOT / "output" / "figures"
LABELS = {"anios_experiencia": "Años de experiencia", "ultimo_puesto": "Último puesto", "tecnologias": "Tecnologías",
          "pretension_salarial": "Pretensión salarial (S/)", "disponibilidad": "Disponibilidad",
          "nivel_ingles": "Nivel de inglés", "items": "Ítems", "promocion": "Promoción",
          "direccion_entrega": "Dirección de entrega", "metodo_pago": "Método de pago", "total_dicho": "Total dicho (S/)"}


def show_value(v):
    if v == SIN_VERIFICAR:
        return "⚠️ sin_verificar"
    if isinstance(v, dict) and "fecha" in v:
        return f"{v.get('texto')} → {v.get('fecha')}"
    if isinstance(v, dict):
        return f"{v.get('descripcion')} (−S/ {v.get('descuento')})"
    if isinstance(v, list) and v and isinstance(v[0], dict):
        return ", ".join(f"{i['cantidad']} × {i['producto']} (S/ {i['precio_unitario']})" for i in v)
    if isinstance(v, list):
        return ", ".join(v)
    return "—" if v is None else str(v)


st.title("🎙️ From Voice to Record")
st.caption("Entrevistas → ficha del candidato · Llamadas de venta → pedido validado. Whisper + extracción + verificación literal.")
tab_run, tab_results, tab_corpus = st.tabs(["Procesar audio", "Resultados de los experimentos", "Corpus"])

with tab_run:
    left, right = st.columns([1, 2])
    with left:
        source = st.radio("Audio", ["Clip del corpus", "Subir archivo"], horizontal=True)
        if source == "Clip del corpus":
            clip = st.selectbox("Clip", sorted(p.stem for p in path("audio").glob("*.m4a")))
            audio_path, task = path("audio") / f"{clip}.m4a", clip.split("_")[0]
            st.audio(str(audio_path))
        else:
            up = st.file_uploader("Archivo de audio", type=["m4a", "mp3", "wav", "ogg", "opus", "mp4"])
            task = st.radio("Tipo", ["interview", "call"], format_func={"interview": "Entrevista", "call": "Llamada"}.get, horizontal=True)
            clip, audio_path = None, None
            if up:
                tmp = Path(tempfile.gettempdir()) / up.name
                tmp.write_bytes(up.getvalue())
                audio_path = tmp
                st.audio(up)
        level = st.selectbox("Degradar antes de transcribir", cfg["degradation"]["levels"],
                             help="L0 original · L1 nota de voz (Opus) · L2 canal telefónico 8 kHz · L3 L1 + ruido")
        with st.expander("Opciones (por defecto: config.yaml)"):
            models = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]
            model = st.selectbox("Modelo Whisper", models, index=models.index(cfg["asr"]["model"]))
            device = st.selectbox("Dispositivo", ["cuda", "cpu"], index=["cuda", "cpu"].index(cfg["asr"]["device"]))
            compute = st.selectbox("Precisión", ["float16", "int8_float16", "int8"],
                                   index=["float16", "int8_float16", "int8"].index(cfg["asr"]["compute_type"]))
            glossary = st.checkbox("Usar glosario", value=cfg["asr"]["use_glossary"])
            method = st.radio("Extractor", ["llm", "regex"], index=["llm", "regex"].index(cfg["extraction"]["method"]), horizontal=True)
        go = st.button("Procesar", type="primary", disabled=audio_path is None)

    with right:
        if go:
            audio = load(audio_path)
            if level != "L0":
                y, sr, snr = make_levels(audio, cfg["seed"])[level]
                audio = y if sr == SR else resample_poly(y, SR, sr).astype("float32")  # Whisper wants 16 kHz
                st.info(f"Audio degradado a {level}" + (f" (SNR medido {snr} dB)" if snr else ""))
            with st.spinner("Transcribiendo y extrayendo…"):
                res = engine.process(audio, task, model, device, compute, glossary, method)
            st.session_state["res"], st.session_state["clip"] = res, clip
        res = st.session_state.get("res")
        if res:
            if res["error"]:
                st.error(res["error"])
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Dispositivo real", res["device"].upper())
                c2.metric("Transcripción", f"{res['elapsed']['asr']:.1f} s")
                c3.metric("Extracción", f"{res['elapsed']['extraction']:.1f} s")
                c4.metric("Total", f"{res['elapsed']['total']:.1f} s")
                if res["device"] == "cuda":
                    st.caption(f"{res['asr']['gpu_name']} · VRAM modelo {res['asr']['vram_model_mb']} MB · pico extra {res['asr']['vram_peak_extra_mb']} MB · RTF {res['asr']['rtf']}")
                gold = {**ground_truth("interview"), **ground_truth("call")}.get(st.session_state.get("clip"))
                rows = [{"Campo": LABELS[k], "Extraído": show_value(res["fields"][k]), "Registro verificado": show_value(v),
                         **({"Respuesta correcta": show_value(gold[k])} if gold else {})} for k, v in res["verified"].items()]
                st.subheader("Ficha" if task == "interview" else "Pedido")
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                val = res.get("validation")
                if val:
                    st.subheader("Validación aritmética")
                    a, b, c = st.columns(3)
                    a.metric("Total calculado", f"S/ {val['total_calculado']}", help=f"subtotal {val['subtotal']} − descuento {val['descuento']}")
                    b.metric("Total dicho", f"S/ {val['total_dicho']}")
                    c.metric("Confianza ASR del total", val["confianza_total_asr"] if val["confianza_total_asr"] is not None else "—")
                    if val["estado"] == "cuadra":
                        st.success(val["motivo"])
                    elif val["diagnostico"] == "error_hablante":
                        st.error(f"No cuadra · error del hablante · {val['motivo']}")
                    else:
                        st.warning(f"No cuadra · posible error de transcripción · {val['motivo']}")
                    for p in val["precios_distintos_al_menu"]:
                        st.warning(f"Se dijo S/ {p['dicho']} para {p['producto']}; el menú dice S/ {p['menu']}.")
                with st.expander("Transcripción"):
                    st.write(res["transcript"])
                with st.expander("JSON completo"):
                    st.json({k: v for k, v in res.items() if k not in ("words", "traceback")})

with tab_results:
    if not TABLES.exists():
        st.info("Corre los experimentos primero (ver README).")
    else:
        for title, name, fig in [("Modelos, dispositivo y precisión (L0)", "asr_benchmark", "asr_wer_vs_speed.png"),
                                 ("GPU vs CPU", "gpu_vs_cpu", None), ("Glosario (IC 95 % bootstrap)", "glossary_ci", None),
                                 ("Escalera de degradación: transcripción", "ladder_wer", "ladder_wer.png"),
                                 ("Fichas de entrevistas", "records_interview", "records_by_level.png"),
                                 ("Pedidos de llamadas", "records_call", None),
                                 ("Validación aritmética por nivel", "validation_by_level", None),
                                 ("Diagnóstico de cada total que no cuadra", "validation_diagnosis", None)]:
            f = TABLES / f"{name}.csv"
            if f.exists():
                st.subheader(title)
                if fig and (FIGURES / fig).exists():
                    st.image(str(FIGURES / fig))
                st.dataframe(pd.read_csv(f), hide_index=True, width="stretch")

with tab_corpus:
    st.markdown("Corpus propio: 5 entrevistas y 5 llamadas grabadas con un compañero el 2026-11-01. "
                "Datos inventados; ambos hablantes dieron su consentimiento. Sin audio sintético.")
    st.dataframe(pd.read_csv(path("manifest")), hide_index=True, width="stretch")
    st.json(json.loads((path("ground_truth") / "calls.json").read_text(encoding="utf-8"))["_menu"])
