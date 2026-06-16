"""
app.py — backend Flask do CardioIA Fase 4 (portal multi-telas).

Rotas:
  GET  /                     portal (SPA: Visão Geral, Classificador, Dados, Resultados, Ética, Histórico)
  GET  /api/dashboard        dados agregados (métricas, fairness, dataset, lista de plots)
  GET  /api/history          histórico de predições da sessão
  POST /predict              recebe imagem (multipart 'file') -> {classe, score, gradcam, disclaimer}
  GET  /predict-sample?name= roda a predição numa imagem de exemplo (dados/raw/sample)
  GET  /sample-list          lista as imagens de exemplo
  GET  /sample/<arquivo>     serve uma imagem de exemplo
  GET  /resultados/<arquivo> serve um gráfico de resultados/Grad-CAM
  GET  /health               liveness

Governança: score "experimental" (não clínico), disclaimer, retenção ZERO de uploads.

Executar:  python app.py   (ou, recomendado, `python start.py` na raiz do projeto)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import inference

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = Path(__file__).resolve().parent / "templates"
RES = ROOT / "resultados"
SAMPLES = ROOT / "dados" / "raw" / "sample"
DASHBOARD_JSON = RES / "dashboard_data.json"

app = Flask(__name__)
CORS(app)

ALLOWED = {"png", "jpg", "jpeg", "bmp", "webp"}
HISTORY: list[dict] = []  # histórico em memória (zera ao reiniciar)


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def _registrar(arquivo: str, result: dict, origem: str):
    HISTORY.insert(0, {
        "arquivo": arquivo, "classe": result["classe"],
        "score": result["score_experimental"], "threshold": result["threshold"],
        "origem": origem, "hora": datetime.now().strftime("%H:%M:%S"),
    })
    del HISTORY[50:]


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/")
def index():
    return send_from_directory(TEMPLATES, "index.html")


@app.get("/api/dashboard")
def api_dashboard():
    if DASHBOARD_JSON.exists():
        return app.response_class(DASHBOARD_JSON.read_text(encoding="utf-8"), mimetype="application/json")
    return jsonify({"erro": "dashboard_data.json não gerado. Rode scripts/gerar_dashboard_data.py."}), 503


@app.get("/api/history")
def api_history():
    return jsonify(HISTORY)


@app.get("/sample-list")
def sample_list():
    if not SAMPLES.exists():
        return jsonify([])
    return jsonify(sorted(p.name for p in SAMPLES.glob("*.png")))


@app.get("/sample/<path:arquivo>")
def serve_sample(arquivo):
    return send_from_directory(SAMPLES, arquivo)


@app.get("/resultados/<path:arquivo>")
def serve_resultado(arquivo):
    return send_from_directory(RES, arquivo)


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"erro": "Envie uma imagem no campo 'file'."}), 400
    file = request.files["file"]
    if not file.filename or not _allowed(file.filename):
        return jsonify({"erro": "Formato inválido. Use PNG/JPG/JPEG/BMP."}), 400
    image_bytes = file.read()  # em memória; nunca salvo em disco (retenção zero)
    try:
        result = inference.predict_with_gradcam(image_bytes)
    except FileNotFoundError as exc:
        return jsonify({"erro": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Falha na predição")  # log SEM a imagem
        return jsonify({"erro": f"Falha ao processar: {exc}"}), 500
    _registrar(file.filename, result, "upload")
    return jsonify(result)


@app.get("/predict-sample")
def predict_sample():
    name = request.args.get("name", "")
    img = SAMPLES / name
    if not name or not img.exists() or not _allowed(name):
        return jsonify({"erro": "Exemplo inválido."}), 400
    try:
        result = inference.predict_with_gradcam(img.read_bytes())
    except FileNotFoundError as exc:
        return jsonify({"erro": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Falha na predição (sample)")
        return jsonify({"erro": f"Falha ao processar: {exc}"}), 500
    _registrar(name, result, "exemplo")
    result["sample_url"] = f"/sample/{name}"
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
