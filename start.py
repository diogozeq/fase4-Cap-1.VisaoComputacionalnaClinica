#!/usr/bin/env python3
"""
============================================================
  START SUPREMO — CardioIA Fase 4 (Visão Computacional)
============================================================
Um comando para usar de verdade tudo o que foi construído:

    python start.py

O que ele faz, em ordem:
  1) Garante um ambiente compatível (Python 3.10–3.12) num venv isolado (.venv),
     instalando as dependências na primeira execução.
  2) Mostra os RESULTADOS (tabela comparativa dos modelos) — "os cálculos".
  3) Faz um AUTOTESTE de inferência real nas imagens de teste (dados/raw/sample).
  4) Sobe a INTERFACE WEB (Flask) e abre o navegador em http://localhost:5000,
     onde você envia um raio-X e recebe classe + score + Grad-CAM.

Opções:
  python start.py --test       só ambiente + métricas + autoteste (não sobe a interface)
  python start.py --no-test    pula o autoteste e vai direto para a interface
  python start.py --port 8000  troca a porta da interface
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "app" / "backend"
MODEL = ROOT / "modelos" / "transfer_mobilenetv2.keras"
CONFIG = ROOT / "modelos" / "preprocess_config.json"
REQ = BACKEND / "requirements.txt"
SAMPLES = ROOT / "dados" / "raw" / "sample"
METRICS_CSV = ROOT / "resultados" / "metricas_comparativo.csv"

IS_WIN = os.name == "nt"
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")
PY_OK = (3, 10) <= sys.version_info[:2] <= (3, 12)

C = {"b": "\033[1m", "g": "\033[92m", "y": "\033[93m", "r": "\033[91m", "c": "\033[96m", "0": "\033[0m"}
if IS_WIN:
    os.system("")  # habilita ANSI no terminal do Windows
# Garante UTF-8 no console (Windows usa cp1252 por padrão e quebra em → ✓ ≈)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


def say(msg, color="0"):
    print(f"{C.get(color, '')}{msg}{C['0']}")


def banner():
    say("\n" + "=" * 60, "c")
    say("   CardioIA — Fase 4 · START SUPREMO", "b")
    say("   Triagem de Cardiomegalia em raio-X (Visão Computacional)", "c")
    say("=" * 60, "c")
    say("  Protótipo acadêmico (FIAP). NÃO é dispositivo médico.", "y")
    say("=" * 60 + "\n", "c")


# ---------------------------------------------------------------------------
# 1) Ambiente: venv compatível + dependências
# ---------------------------------------------------------------------------
def in_target_env() -> bool:
    try:
        return Path(sys.executable).resolve() == VENV_PY.resolve()
    except Exception:
        return False


def find_compatible_python() -> str | None:
    """Procura um Python 3.10–3.12 (py launcher, PATH e caminhos conhecidos no Windows)."""
    probes = []
    if IS_WIN:
        for v in ("3.12", "3.11", "3.10"):
            probes.append(["py", f"-{v}", "-c", "import sys;print(sys.executable)"])
    for v in ("python3.12", "python3.11", "python3.10"):
        probes.append([v, "-c", "import sys;print(sys.executable)"])
    for p in probes:
        try:
            out = subprocess.check_output(p, text=True, stderr=subprocess.DEVNULL).strip()
            if out and Path(out).exists():
                return out
        except Exception:
            continue
    if IS_WIN:
        base = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python"
        for name in ("Python312", "Python311", "Python310"):
            exe = base / name / "python.exe"
            if exe.exists():
                return str(exe)
    return None


def deps_present(python_exe: str) -> bool:
    code = "import tensorflow, flask, flask_cors, cv2, PIL, matplotlib, numpy"
    return subprocess.call([python_exe, "-c", code],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def bootstrap():
    """Cria/usa o .venv compatível e re-executa este script dentro dele."""
    say("[1/4] Preparando o ambiente...", "b")
    base = sys.executable if PY_OK else find_compatible_python()
    if base is None:
        say("  ✗ Não encontrei Python 3.10–3.12 (necessário para o TensorFlow).", "r")
        say("    Instale o Python 3.11 em https://www.python.org/downloads/ e rode de novo.", "y")
        sys.exit(1)

    if not VENV_PY.exists():
        say(f"  • Criando ambiente isolado em .venv (base: {base})", "c")
        subprocess.check_call([base, "-m", "venv", str(VENV)])

    if not deps_present(str(VENV_PY)):
        say("  • Instalando dependências (primeira vez — pode levar alguns minutos)...", "y")
        subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "-q"])
        subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "-q", "-r", str(REQ)])
        say("  ✓ Dependências instaladas.", "g")
    else:
        say("  ✓ Dependências já presentes.", "g")

    # re-executa dentro do venv (mantém os argumentos)
    rc = subprocess.call([str(VENV_PY), str(ROOT / "start.py"), *sys.argv[1:]])
    sys.exit(rc)


# ---------------------------------------------------------------------------
# 2) Resultados / cálculos
# ---------------------------------------------------------------------------
def show_metrics():
    say("\n[2/4] Resultados dos modelos (conjunto de teste):", "b")
    if not METRICS_CSV.exists():
        say("  (resultados/metricas_comparativo.csv não encontrado)", "y")
        return
    import csv
    rows = list(csv.DictReader(METRICS_CSV.open(encoding="utf-8")))
    cols = ["modelo", "accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"]
    head = ["Modelo", "Acur.", "Prec.", "Recall", "F1", "AUC-ROC", "AUC-PR"]
    print("  " + " | ".join(f"{h:>20}" if i == 0 else f"{h:>8}" for i, h in enumerate(head)))
    print("  " + "-" * 86)
    for r in rows:
        vals = []
        for i, c in enumerate(cols):
            v = r.get(c, "")
            try:
                v = f"{float(v):.3f}"
            except Exception:
                pass
            vals.append(f"{v:>20}" if i == 0 else f"{v:>8}")
        print("  " + " | ".join(vals))
    say("  → Transfer Learning (fine-tuning) é o melhor (AUC ≈ 0.77, recall 0.77).", "g")


# ---------------------------------------------------------------------------
# 3) Autoteste de inferência real
# ---------------------------------------------------------------------------
def selftest():
    say("\n[3/4] Autoteste de inferência (imagens reais de teste):", "b")
    if not MODEL.exists():
        say(f"  ✗ Modelo não encontrado em {MODEL}.", "r")
        say("    Treine na Etapa 2 (Kaggle) e coloque o .keras em modelos/.", "y")
        return False
    sys.path.insert(0, str(BACKEND))
    import inference  # usa o mesmo código do backend
    imgs = sorted(SAMPLES.glob("*.png"))
    if not imgs:
        say("  (sem imagens em dados/raw/sample — pulando autoteste)", "y")
        return True
    say("  • Carregando modelo e prevendo... (a primeira carga leva alguns segundos)", "c")
    for img in imgs:
        out = inference.predict_with_gradcam(img.read_bytes())
        esperado = "Cardiomegaly" if "cardiomegaly" in img.name.lower() else "No Finding"
        ok = "✓" if out["classe"] == esperado else "≈"
        gc = "Grad-CAM ok" if out.get("gradcam") else "sem Grad-CAM"
        say(f"    {ok} {img.name[:34]:34s} -> {out['classe']:13s} "
            f"score={out['score_experimental']:.3f}  ({gc})",
            "g" if ok == "✓" else "y")
    return True


# ---------------------------------------------------------------------------
# 4) Interface web
# ---------------------------------------------------------------------------
def launch_ui(port: int):
    say(f"\n[4/4] Subindo a interface web em http://localhost:{port}", "b")
    if not MODEL.exists():
        say("  ✗ Sem modelo treinado — a interface não conseguiria prever. Abortando.", "r")
        return
    sys.path.insert(0, str(BACKEND))
    import app as flask_app

    def open_later():
        time.sleep(2.5)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=open_later, daemon=True).start()
    say("  ✓ Pronto! O navegador vai abrir. Envie um raio-X (ex.: dados/raw/sample).", "g")
    say("    (Ctrl+C para encerrar)\n", "c")
    flask_app.app.run(host="0.0.0.0", port=port, debug=False)


# ---------------------------------------------------------------------------
def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    args = sys.argv[1:]
    port = 5000
    if "--port" in args:
        try:
            port = int(args[args.index("--port") + 1])
        except Exception:
            pass

    banner()

    # Garante o ambiente (a menos que já estejamos no venv ou o usuário force o atual)
    if not in_target_env() and not os.environ.get("CARDIOIA_SKIP_BOOTSTRAP"):
        bootstrap()
        return

    say("[1/4] Ambiente OK (venv ativo).", "g")
    show_metrics()
    if "--no-test" not in args:
        selftest()

    if "--test" in args:
        say("\n✓ Autoteste concluído (modo --test, interface não iniciada).", "g")
        return

    try:
        launch_ui(port)
    except KeyboardInterrupt:
        say("\nEncerrado. Até a próxima!", "c")


if __name__ == "__main__":
    main()
