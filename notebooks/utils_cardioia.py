"""
utils_cardioia.py — utilitários compartilhados da Fase 4 (CardioIA / Visão Computacional).

Reúso entre os notebooks 1-4 e o backend Flask. Centraliza as decisões que o Conselho
marcou como críticas/altas:

- Seeds globais (reprodutibilidade).
- Pré-processamento TRAVADO por modelo, salvo/lido de modelos/preprocess_config.json,
  para garantir paridade treino ↔ inferência (Flask).
- Avaliação com as métricas da rubrica (acurácia, matriz de confusão, precisão, recall,
  F1) PRIMEIRO + AUC-ROC e AUC-PR como rigor complementar (classe desbalanceada).
- Grad-CAM PARAMETRIZADO por camada (a última conv difere entre VGG/ResNet/MobileNet/CNN-zero).

Sem dependência de dados: pode ser importado em qualquer lugar.
Testado com TensorFlow 2.16 (Keras 3) / scikit-learn 1.5.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np

SEED = 565535  # RM do aluno — seed única do projeto

# Camada conv-final por arquitetura (necessária para o Grad-CAM correto).
LAST_CONV_LAYER = {
    "mobilenetv2": "out_relu",        # ativação final do MobileNetV2 (mesma do treino/inferência)
    "vgg16": "block5_conv3",          # tf.keras.applications.VGG16
    "resnet50": "conv5_block3_out",   # tf.keras.applications.ResNet50
    # Para a CNN do zero, nomeie a última Conv2D como "last_conv" ao construir o modelo.
    "cnn_scratch": "last_conv",
}

IMG_SIZE = (224, 224)
CLASS_NAMES = ["No Finding", "Cardiomegaly"]  # índice 0 = negativo, 1 = positivo


# --------------------------------------------------------------------------------------
# Reprodutibilidade
# --------------------------------------------------------------------------------------
def set_global_seed(seed: int = SEED) -> None:
    """Fixa todas as fontes de aleatoriedade. Chamar no início de cada notebook."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except Exception:  # tensorflow pode não estar importado em alguns contextos
        pass


# --------------------------------------------------------------------------------------
# Pré-processamento travado por modelo (paridade treino ↔ Flask)
# --------------------------------------------------------------------------------------
def get_preprocess_fn(model_kind: str):
    """
    Retorna a função de pré-processamento correta para o modelo.
    'cnn_scratch' usa rescale 1/255; transfer learning usa o preprocess_input do backbone.
    """
    kind = model_kind.lower()
    if kind == "cnn_scratch":
        return lambda arr: arr.astype("float32") / 255.0
    if kind == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        return lambda arr: preprocess_input(arr.astype("float32"))
    if kind == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input

        return lambda arr: preprocess_input(arr.astype("float32"))
    if kind == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input

        return lambda arr: preprocess_input(arr.astype("float32"))
    raise ValueError(f"model_kind desconhecido: {model_kind}")


def save_preprocess_config(path: str | Path, model_kind: str, img_size=IMG_SIZE,
                           class_names: Iterable[str] = CLASS_NAMES,
                           threshold: float = 0.5) -> None:
    """Persiste a configuração que o Flask precisa para replicar exatamente o treino."""
    cfg = {
        "model_kind": model_kind.lower(),
        "img_size": list(img_size),
        "class_names": list(class_names),
        "last_conv_layer": LAST_CONV_LAYER.get(model_kind.lower()),
        "threshold": threshold,  # ponto de operação escolhido (ver tune_threshold)
        "seed": SEED,
    }
    Path(path).write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return cfg


def load_preprocess_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_image_array(img_path_or_pil, img_size=IMG_SIZE) -> np.ndarray:
    """Carrega imagem (caminho ou PIL.Image) → np.array RGB (H, W, 3) SEM normalizar."""
    from PIL import Image

    img = img_path_or_pil
    if not hasattr(img, "convert"):
        img = Image.open(img_path_or_pil)
    img = img.convert("RGB").resize(img_size)
    return np.asarray(img, dtype="float32")


# --------------------------------------------------------------------------------------
# Avaliação — métricas da rubrica PRIMEIRO + AUC complementar
# --------------------------------------------------------------------------------------
def evaluate_binary(y_true, y_scores, threshold: float = 0.5, class_names=CLASS_NAMES,
                    title: str = "", save_dir: str | Path | None = None) -> dict:
    """
    Avaliação completa para classificação binária.

    y_true   : array 0/1
    y_scores : probabilidade da classe positiva (sigmoid)
    Retorna dict com todas as métricas e (se save_dir) salva matriz de confusão, ROC e PR.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, ConfusionMatrixDisplay, classification_report,
        roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    )

    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    y_pred = (y_scores >= threshold).astype(int)

    metrics = {
        # --- métricas exigidas pela rubrica (mostrar primeiro) ---
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # --- rigor complementar p/ classe desbalanceada ---
        "auc_roc": float(roc_auc_score(y_true, y_scores)) if len(set(y_true)) > 1 else float("nan"),
        "auc_pr": float(average_precision_score(y_true, y_scores)) if len(set(y_true)) > 1 else float("nan"),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
    }

    print(f"\n=== {title} ===")
    print(f"Acurácia : {metrics['accuracy']:.4f}")
    print(f"Precisão : {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"AUC-ROC  : {metrics['auc_roc']:.4f}   (complementar)")
    print(f"AUC-PR   : {metrics['auc_pr']:.4f}   (complementar; bom p/ desbalanceio)")
    print(f"\n{classification_report(y_true, y_pred, target_names=class_names, zero_division=0)}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        slug = (title or "modelo").lower().replace(" ", "_")

        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues", colorbar=False)
        plt.title(f"Matriz de Confusão — {title}")
        plt.tight_layout()
        plt.savefig(save_dir / f"confusion_{slug}.png", dpi=130)
        plt.close()

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.plot(fpr, tpr, label=f"AUC-ROC={metrics['auc_roc']:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {title}"); plt.legend()
        plt.tight_layout(); plt.savefig(save_dir / f"roc_{slug}.png", dpi=130); plt.close()

        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(rec, prec, label=f"AUC-PR={metrics['auc_pr']:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precisão"); plt.title(f"Precision-Recall — {title}"); plt.legend()
        plt.tight_layout(); plt.savefig(save_dir / f"pr_{slug}.png", dpi=130); plt.close()

    return metrics


def baseline_metrics(y_true) -> dict:
    """Baseline trivial (sempre prevê a classe majoritária) p/ contextualizar."""
    y_true = np.asarray(y_true).astype(int)
    majority = int(round(y_true.mean()))
    y_pred = np.full_like(y_true, majority)
    return {"baseline_accuracy": float((y_pred == y_true).mean()),
            "majority_class": majority}


def tune_threshold(y_true, y_scores, target_recall: float = 0.90) -> float:
    """
    Escolhe o maior threshold que ainda atinge o recall-alvo (ponto de operação clínico:
    em triagem, falso-negativo é pior que falso-positivo). Documentar no relatório.
    """
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(np.asarray(y_true).astype(int), np.asarray(y_scores))
    # precision_recall_curve devolve len(thr) = len(prec)-1
    ok = [t for t, r in zip(thr, rec[:-1]) if r >= target_recall]
    return float(max(ok)) if ok else 0.5


# --------------------------------------------------------------------------------------
# Grad-CAM parametrizado (instrumento de auditoria de shortcut learning)
# --------------------------------------------------------------------------------------
def make_gradcam_heatmap(img_array_preprocessed: np.ndarray, model, last_conv_layer_name: str) -> np.ndarray:
    """
    Gera o heatmap Grad-CAM. `img_array_preprocessed` já deve estar pré-processado e com
    batch dim (1, H, W, 3). `last_conv_layer_name` vem de LAST_CONV_LAYER[model_kind].
    Retorna heatmap normalizado [0,1] no tamanho do mapa convolucional.
    """
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array_preprocessed)
        # saída sigmoid de 1 neurônio (binário)
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_rgb_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Sobrepõe o heatmap (colormap jet) na imagem original RGB uint8. Retorna RGB uint8."""
    import cv2

    h, w = original_rgb_uint8.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.uint8(colored * alpha + original_rgb_uint8 * (1 - alpha))


# --------------------------------------------------------------------------------------
# Verificação anti-vazamento (reutilizável fora da EDA)
# --------------------------------------------------------------------------------------
def assert_no_patient_leakage(manifest_df, patient_col: str = "Patient ID", split_col: str = "split") -> dict:
    """Garante interseção vazia de pacientes entre train/val/test. Levanta AssertionError se vazar."""
    sets = {s: set(g[patient_col].astype(str)) for s, g in manifest_df.groupby(split_col)}
    overlaps = {
        "train_val": sets.get("train", set()) & sets.get("val", set()),
        "train_test": sets.get("train", set()) & sets.get("test", set()),
        "val_test": sets.get("val", set()) & sets.get("test", set()),
    }
    assert not any(overlaps.values()), f"VAZAMENTO de paciente detectado: {overlaps}"
    return {k: 0 for k in overlaps}


DISCLAIMER = (
    "Protótipo acadêmico (FIAP — CardioIA Fase 4). NÃO é dispositivo médico, NÃO foi "
    "validado clinicamente e NÃO deve ser usado para diagnóstico real. O valor exibido é "
    "um score experimental, não uma probabilidade clínica."
)
