"""Tables (markdown + csv) and figures for the README and the app.
    python experiments/report.py
Output: output/tables/*.md|csv, output/figures/*.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import load_config  # noqa: E402
from scoring import CALL_FIELDS, INTERVIEW_FIELDS  # noqa: E402

M, T, F = ROOT / "output" / "metrics", ROOT / "output" / "tables", ROOT / "output" / "figures"


def save(df: pd.DataFrame, name: str, floatfmt=".3f") -> None:
    df.to_csv(T / f"{name}.csv", index=False)
    (T / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=floatfmt), encoding="utf-8")


def bootstrap_ci(diffs, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pooled(g: pd.DataFrame) -> pd.Series:
    """Corpus-level WER (errors / reference words), plus mean speed."""
    err = g["substitutions"] + g["deletions"] + g["insertions"]
    return pd.Series({"WER": err.sum() / g["ref_words"].sum(), "CER (media)": g["cer"].mean(),
                      "WER entrevistas": err[g.task == "interview"].sum() / g.loc[g.task == "interview", "ref_words"].sum(),
                      "WER llamadas": err[g.task == "call"].sum() / g.loc[g.task == "call", "ref_words"].sum(),
                      "recall de términos": g["term_recall"].mean(), "segundos/clip": g["elapsed_seconds"].mean(),
                      "RTF": g["rtf"].mean(), "VRAM modelo MB": g["vram_model_mb"].max(),
                      "VRAM pico extra MB": g["vram_peak_extra_mb"].max(), "dispositivo real": "/".join(sorted(g["device"].unique()))})


def asr_tables(runs: pd.DataFrame) -> None:
    l0 = runs[runs.level == "L0"]
    bench = l0.groupby(["model", "device_requested", "compute_type", "glossary"]).apply(pooled, include_groups=False).reset_index()
    bench = bench.sort_values("WER")
    save(bench, "asr_benchmark")

    fig, ax = plt.subplots(figsize=(8, 5))
    plain = bench[~bench.glossary & bench.compute_type.isin(["float16", "int8"])
                  & ~((bench.device_requested == "cuda") & (bench.compute_type == "int8"))]  # one point per model+device
    for dev, marker in (("cuda", "o"), ("cpu", "s")):
        d = plain[plain.device_requested == dev]
        ax.scatter(d["RTF"], d["WER"], marker=marker, s=60, label="GPU float16" if dev == "cuda" else "CPU int8")
        for _, r in d.iterrows():
            ax.annotate(r.model, (r.RTF, r.WER), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set(xscale="log", xlabel="RTF (segundos de cómputo / segundo de audio, log)", ylabel="WER (L0)",
           title="Calidad vs velocidad de transcripción")
    ax.legend()
    fig.tight_layout()
    fig.savefig(F / "asr_wer_vs_speed.png", dpi=150)
    plt.close(fig)

    # glossary: paired per-clip differences with a bootstrap 95 % CI
    rows = []
    for model in ("large-v3", "large-v3-turbo"):
        a = l0[(l0.model == model) & (l0.compute_type == "float16") & (l0.device_requested == "cuda")]
        on, off = a[a.glossary].set_index("clip_id"), a[~a.glossary].set_index("clip_id")
        common = on.index.intersection(off.index)
        if len(common) == 0:
            continue
        for metric in ("wer", "term_recall"):
            d = (on.loc[common, metric] - off.loc[common, metric]).dropna()
            mean, lo, hi = bootstrap_ci(d.values)
            rows.append({"modelo": model, "métrica": metric, "sin glosario": off.loc[common, metric].mean(),
                         "con glosario": on.loc[common, metric].mean(), "diferencia media": mean,
                         "IC 95% bajo": lo, "IC 95% alto": hi, "clips": len(d),
                         "conclusión": "mejora" if (hi < 0 if metric == "wer" else lo > 0) else
                         "empeora" if (lo > 0 if metric == "wer" else hi < 0) else "no concluyente"})
    save(pd.DataFrame(rows), "glossary_ci")

    # GPU vs CPU on the same model
    rows = []
    for model in ("small", "large-v3-turbo", "large-v3"):
        g = bench[(bench.model == model) & (bench.device_requested == "cuda") & (bench.compute_type == "float16") & (~bench.glossary)]
        c = bench[(bench.model == model) & (bench.device_requested == "cpu")]
        if len(g) and len(c):
            rows.append({"modelo": model, "GPU float16 s/clip": g["segundos/clip"].iloc[0], "CPU int8 s/clip": c["segundos/clip"].iloc[0],
                         "aceleración GPU": c["segundos/clip"].iloc[0] / g["segundos/clip"].iloc[0],
                         "WER GPU": g["WER"].iloc[0], "WER CPU": c["WER"].iloc[0]})
    save(pd.DataFrame(rows), "gpu_vs_cpu", floatfmt=".2f")


def ladder_tables(runs: pd.DataFrame) -> None:
    a = load_config()["asr"]
    name = f"{a['model']}_{a['device']}_{a['compute_type']}" + ("_glossary" if a["use_glossary"] else "")
    lad = runs[runs.run == name].groupby("level").apply(pooled, include_groups=False).reset_index()
    lad = lad[["level", "WER", "WER entrevistas", "WER llamadas", "CER (media)", "recall de términos"]]
    save(lad, "ladder_wer")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lad["level"], lad["WER entrevistas"], "o-", label="entrevistas")
    ax.plot(lad["level"], lad["WER llamadas"], "s-", label="llamadas")
    ax.set(xlabel="nivel de degradación", ylabel="WER", title=f"Escalera de degradación ({name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(F / "ladder_wer.png", dpi=150)
    plt.close(fig)


def record_tables(rec: pd.DataFrame, val: pd.DataFrame) -> None:
    rec = rec[rec.get("error").isna()] if "error" in rec else rec
    rec = rec.assign(nivel=rec.level.replace({"reference": "texto perfecto"}))
    order = ["texto perfecto", "L0", "L1", "L2", "L3"]
    rows = []
    for (method, nivel, task), g in rec.groupby(["method", "nivel", "task"]):
        fields = INTERVIEW_FIELDS if task == "interview" else CALL_FIELDS
        row = {"método": method, "nivel": nivel, "tarea": task,
               "exactitud de campos": g["fields_correct"].sum() / g["fields_total"].sum(),
               "antes de verificar": g["fields_correct_before_verification"].sum() / g["fields_total"].sum(),
               "campos sin_verificar": int(g["fields_unverified"].sum())}
        row.update({f: g[f"ok_{f}"].mean() for f in fields})
        if task == "interview":
            row["F1 tecnologías"] = g["ok_tecnologias_f1"].mean()
            row["error abs. salario"] = g["ok_pretension_salarial_abs_error"].mean()
        else:
            row["F1 ítems"] = g["ok_items_f1"].mean()
        rows.append(row)
    t = pd.DataFrame(rows)
    t["_o"] = t["nivel"].map({k: i for i, k in enumerate(order)})
    t = t.sort_values(["tarea", "método", "_o"]).drop(columns="_o")
    for task in ("interview", "call"):
        save(t[t.tarea == task].dropna(axis=1, how="all"), f"records_{task}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, task, title in zip(axes, ("interview", "call"), ("Entrevistas (6 campos)", "Llamadas (5 campos)")):
        for method, style in (("regex", "o--"), ("llm-3b", "s-"), ("llm-7b", "^-")):
            d = t[(t.tarea == task) & (t["método"] == method)]
            if len(d):
                ax.plot(d["nivel"], d["exactitud de campos"], style, label=method)
        ax.set(title=title, xlabel="entrada", ylim=(0, 1.05))
        ax.legend()
    axes[0].set_ylabel("exactitud de campos (verificados)")
    fig.tight_layout()
    fig.savefig(F / "records_by_level.png", dpi=150)
    plt.close(fig)

    v = val.assign(nivel=val.level.replace({"reference": "texto perfecto"}))
    vt = v.pivot_table(index=["method", "nivel"], columns="outcome", values="clip_id", aggfunc="count", fill_value=0).reset_index()
    for col in ("caught", "missed", "false_alarm", "correct_pass"):
        if col not in vt:
            vt[col] = 0
    vt["_o"] = vt["nivel"].map({k: i for i, k in enumerate(order)})
    vt = vt.sort_values(["method", "_o"]).drop(columns="_o")
    vt = vt.rename(columns={"method": "método", "caught": "atrapados", "missed": "escapados",
                            "false_alarm": "falsas alarmas", "correct_pass": "correctos sin alarma"})
    save(vt[["método", "nivel", "atrapados", "escapados", "falsas alarmas", "correctos sin alarma"]], "validation_by_level", floatfmt=".0f")
    diag = v[v.estado != "cuadra"].groupby(["method", "nivel", "clip_id"])[["estado", "diagnostico", "total_dicho",
                                                                            "total_calculado", "confianza_total_asr"]].first().reset_index()
    save(diag, "validation_diagnosis", floatfmt=".2f")


def main() -> None:
    T.mkdir(parents=True, exist_ok=True)
    F.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(M / "asr_runs.csv")
    asr_tables(runs)
    ladder_tables(runs)
    if (M / "records.csv").exists():
        record_tables(pd.read_csv(M / "records.csv"), pd.read_csv(M / "validation.csv"))
    print("tables:", sorted(p.name for p in T.glob("*.md")))


if __name__ == "__main__":
    main()
