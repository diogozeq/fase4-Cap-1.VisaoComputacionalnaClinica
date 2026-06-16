# Gera resultados/dashboard_data.json com tudo que o portal web precisa,
# a partir do manifesto + metricas + fairness. So usa csv/json (sem TF/pandas).
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "resultados"
MANI = ROOT / "dados" / "processed" / "data_manifest.csv"


def load_manifest_stats():
    rows = list(csv.DictReader(MANI.open(encoding="utf-8")))
    splits = defaultdict(lambda: {"imgs": 0, "pacientes": set(), "Cardiomegaly": 0, "No Finding": 0})
    gender = defaultdict(lambda: {"Cardiomegaly": 0, "No Finding": 0})
    view = defaultdict(lambda: {"Cardiomegaly": 0, "No Finding": 0})
    ages = []
    for r in rows:
        s = r["split"]
        lbl = r.get("target_label") or ("Cardiomegaly" if str(r.get("target")) == "1" else "No Finding")
        splits[s]["imgs"] += 1
        splits[s]["pacientes"].add(r["Patient ID"])
        splits[s][lbl] += 1
        gender[r.get("Patient Gender", "?")][lbl] += 1
        view[r.get("View Position", "?")][lbl] += 1
        try:
            ages.append(float(r.get("Patient Age", "")))
        except Exception:
            pass
    out_splits = {}
    for s, d in splits.items():
        out_splits[s] = {"imgs": d["imgs"], "pacientes": len(d["pacientes"]),
                         "Cardiomegaly": d["Cardiomegaly"], "No Finding": d["No Finding"]}
    return {
        "total_imgs": len(rows),
        "total_pacientes": len({r["Patient ID"] for r in rows}),
        "splits": out_splits,
        "genero": dict(gender),
        "view": dict(view),
        "idade_min": round(min(ages), 0) if ages else None,
        "idade_max": round(max(ages), 0) if ages else None,
        "idade_media": round(sum(ages) / len(ages), 1) if ages else None,
    }


def load_csv(path):
    return list(csv.DictReader(Path(path).open(encoding="utf-8"))) if Path(path).exists() else []


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def list_plots():
    def g(*pats):
        out = []
        for p in pats:
            out += sorted([f.name for f in RES.glob(p)])
        return out
    return {
        "confusao": g("confusion_*.png"),
        "roc": g("roc_*.png"),
        "pr": g("pr_*.png"),
        "treino": g("history_*.png"),
        "gradcam": g("gradcam_*.png"),
        "fairness": g("fairness_genero.png", "fairness_idade.png"),
    }


data = {
    "projeto": {
        "nome": "CardioIA — Fase 4",
        "subtitulo": "Triagem de Cardiomegalia em raio-X de tórax (Visão Computacional)",
        "aluno": "Diogo Zequini", "rm": "565535", "turma": "2TIAOA",
        "dataset": "NIH Chest X-ray14 — Cardiomegaly vs No Finding",
        "disclaimer": ("Protótipo acadêmico (FIAP). NÃO é dispositivo médico, NÃO foi validado "
                       "clinicamente e NÃO deve ser usado para diagnóstico real."),
    },
    "metricas": load_json(RES / "metricas.json") or [],
    "fairness": load_csv(RES / "fairness_metrics.csv"),
    "fairness_gaps": (load_json(RES / "fairness_resumo.json") or {}).get("gaps", {}),
    "dataset": load_manifest_stats(),
    "plots": list_plots(),
}

(RES / "dashboard_data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
print("OK -> resultados/dashboard_data.json")
print("Splits:", data["dataset"]["splits"])
print("Plots:", {k: len(v) for k, v in data["plots"].items()})
