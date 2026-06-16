"""
inference.py — carga do modelo e predição para o backend Flask.

O modelo treinado (Etapa 2) tem o PRÉ-PROCESSAMENTO EMBUTIDO (camadas Rescaling /
preprocess_input dentro do próprio .keras). Logo, a inferência só precisa:
redimensionar para 224x224 RGB e alimentar o array BRUTO (0-255). Isso garante
paridade total treino ↔ Flask, sem reaplicar normalização (preprocess_in_model=true).

Expõe predict() e predict_with_gradcam(). Grad-CAM é best-effort: se falhar
(ex.: base aninhada), a predição ainda é retornada com gradcam=None.

O .keras NÃO é versionado (.gitignore). Coloque o modelo em modelos/ ou aponte
via variável de ambiente CARDIOIA_MODEL.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("CARDIOIA_MODEL", ROOT / "modelos" / "transfer_mobilenetv2.keras"))
CONFIG_PATH = Path(os.environ.get("CARDIOIA_CONFIG", ROOT / "modelos" / "preprocess_config.json"))

DISCLAIMER = (
    "Protótipo acadêmico (FIAP — CardioIA Fase 4). NÃO é dispositivo médico, NÃO foi "
    "validado clinicamente e NÃO deve ser usado para diagnóstico real. O valor exibido é "
    "um score experimental, não uma probabilidade clínica."
)

_model = None
_cfg = None


def _lazy_load():
    global _model, _cfg
    if _model is not None:
        return
    import tensorflow as tf

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. Baixe o modelo treinado da Etapa 2 para "
            f"modelos/ ou defina a variável de ambiente CARDIOIA_MODEL."
        )
    if CONFIG_PATH.exists():
        _cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    else:
        _cfg = {"img_size": [224, 224], "class_names": ["No Finding", "Cardiomegaly"],
                "preprocess_in_model": True, "last_conv_layer": "out_relu", "threshold": 0.5}
    # O modelo embute uma camada Lambda(preprocess_input); precisa de custom_objects + safe_mode=False
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    _model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects={"preprocess_input": preprocess_input}, safe_mode=False)


def _prep(pil_img: Image.Image):
    """PIL → (array original uint8 224x224, batch bruto float 0-255)."""
    h, w = _cfg["img_size"]
    raw = np.asarray(pil_img.convert("RGB").resize((w, h)), dtype="float32")  # 0-255
    if not _cfg.get("preprocess_in_model", True):
        # caso raro: pré-proc não embutido → aplica aqui conforme model_kind
        import utils_fallback  # não usado no fluxo padrão
    return raw.astype("uint8"), np.expand_dims(raw.copy(), 0)


def predict(image_bytes: bytes) -> dict:
    _lazy_load()
    pil = Image.open(io.BytesIO(image_bytes))
    _, batch = _prep(pil)
    score = float(_model.predict(batch, verbose=0).ravel()[0])
    thr = float(_cfg.get("threshold", 0.5))
    idx = int(score >= thr)
    return {"classe": _cfg["class_names"][idx], "score_experimental": round(score, 4),
            "threshold": thr, "disclaimer": DISCLAIMER}


def _gradcam_parts(model):
    """Localiza o sub-modelo MobileNetV2 e a camada densa final (clf) para Grad-CAM manual."""
    mnv2 = next((l for l in model.layers if hasattr(l, "layers") and "mobilenet" in l.name.lower()), None)
    dense = None
    try:
        dense = model.get_layer("clf")
    except Exception:
        dense = next((l for l in reversed(model.layers) if l.__class__.__name__ == "Dense"), None)
    return mnv2, dense


def predict_with_gradcam(image_bytes: bytes) -> dict:
    _lazy_load()
    import tensorflow as tf

    pil = Image.open(io.BytesIO(image_bytes))
    raw_uint8, batch = _prep(pil)
    score = float(_model.predict(batch, verbose=0).ravel()[0])
    thr = float(_cfg.get("threshold", 0.5))
    idx = int(score >= thr)

    gradcam_b64 = None
    try:
        import cv2
        import matplotlib.cm as cm
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        # Método manual (robusto a base aninhada): chama o backbone dentro do GradientTape.
        mnv2, dense = _gradcam_parts(_model)
        if mnv2 is None or dense is None:
            raise ValueError("backbone/clf não localizados para Grad-CAM")
        x = preprocess_input(tf.identity(batch.astype("float32")))
        with tf.GradientTape() as tape:
            conv_out = mnv2(x, training=False)
            tape.watch(conv_out)
            pooled = tf.reduce_mean(conv_out, axis=(1, 2))
            channel = dense(pooled)[:, 0]
        grads = tape.gradient(channel, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.squeeze(conv_out[0] @ weights[..., tf.newaxis])
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = cv2.resize(heatmap.numpy(), tuple(_cfg["img_size"][::-1]))
        colored = (cm.jet(heatmap)[..., :3] * 255).astype("float32")
        overlay = np.uint8(0.4 * colored + 0.6 * raw_uint8)
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        gradcam_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:  # noqa: BLE001
        print(f"[gradcam] indisponível: {exc}")

    return {"classe": _cfg["class_names"][idx], "score_experimental": round(score, 4),
            "threshold": thr, "gradcam": gradcam_b64, "disclaimer": DISCLAIMER}
