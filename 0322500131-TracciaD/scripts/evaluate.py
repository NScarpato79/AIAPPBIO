"""Rigenera metriche e figure dai checkpoint salvati (nessun addestramento).

Uso (dalla cartella del progetto): python scripts/evaluate.py
Richiede i pesi in deploy_app/models/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # nessuna finestra grafica: salva solo i file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from medmnist import PneumoniaMNIST
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy_app"))
from src.inference import get_transform, load_model  # noqa: E402

RESULTS = ROOT / "results"
DATA = ROOT / "data"
CLASSES = ["Normal", "Pneumonia"]
NAMES = {"cnn": "CNN", "vit": "ViT"}


def predict_all(model, dataset, transform, device, batch_size=16):
    """Probabilita' della classe Pneumonia per tutto il test set."""
    probs = []
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = torch.stack([
            torch.as_tensor(transform(np.array(dataset[i][0])))
            for i in range(start, end)
        ]).to(device)
        with torch.no_grad():
            probs.extend(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
        print(f"  {end}/{len(dataset)}", flush=True)
    return np.array(probs)


def metrics(y, p):
    pred = (p > 0.5).astype(int)
    tn, fp, _, _ = confusion_matrix(y, pred).ravel()
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall (Sensitivity)": recall_score(y, pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1-score": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, p),
        "Brier Score": brier_score_loss(y, p),
    }


def bootstrap_ci(y, p, n=1000, seed=42):
    """Intervalli di confidenza al 95% (bootstrap sulle immagini di test)."""
    rng = np.random.default_rng(seed)
    samples = []
    while len(samples) < n:
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            samples.append(metrics(y[idx], p[idx]))
    table = pd.DataFrame(samples)
    return table.quantile([0.025, 0.975]).T.set_axis(["CI_low", "CI_high"], axis=1)


def vit_reshape(tensor):
    # Esclude il token CLS e ricostruisce la griglia 14x14 delle patch
    tensor = tensor[:, 1:, :]
    return tensor.reshape(tensor.size(0), 14, 14, tensor.size(2)).permute(0, 3, 1, 2)


def save_gradcam(key, model, dataset, index, device):
    image = np.array(dataset[index][0])
    tensor = torch.tensor(np.array(get_transform(key)(image))).unsqueeze(0).to(device)
    # Immagine da mostrare: senza la normalizzazione ImageNet
    rgb = torch.as_tensor(get_transform("cnn")(image)).permute(1, 2, 0).numpy()
    rgb = np.clip(rgb, 0, 1).astype(np.float32)

    if key == "cnn":
        cam = GradCAM(model=model, target_layers=[model.features[12]])
    else:
        cam = GradCAM(model=model, target_layers=[model.encoder.layers[-1].ln_1],
                      reshape_transform=vit_reshape)
    heatmap = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(1)])[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Radiografia originale")
    axes[1].imshow(show_cam_on_image(rgb, heatmap, use_rgb=True))
    axes[1].set_title(f"Grad-CAM - {NAMES[key]}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / f"gradcam_{key}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    dataset = PneumoniaMNIST(split="test", root=str(DATA), download=True, as_rgb=True)
    y = dataset.labels.flatten().astype(int)

    models, probs = {}, {}
    for key in NAMES:
        print(f"Valutazione {NAMES[key]}...")
        model, device = load_model(key, models_dir=ROOT / "deploy_app" / "models")
        models[key] = (model, device)
        probs[key] = predict_all(model, dataset, get_transform(key), device)

    table = pd.DataFrame({NAMES[k]: metrics(y, probs[k]) for k in NAMES}).T
    table.to_csv(RESULTS / "model_comparison.csv")
    pd.concat({NAMES[k]: bootstrap_ci(y, probs[k]) for k in NAMES}).to_csv(
        RESULTS / "confidence_intervals.csv")
    pd.DataFrame({"y_true": y, "prob_cnn": probs["cnn"],
                  "prob_vit": probs["vit"]}).to_csv(
        RESULTS / "test_predictions.csv", index=False)

    # Matrici di confusione
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key in zip(axes, NAMES):
        ConfusionMatrixDisplay.from_predictions(
            y, (probs[key] > 0.5).astype(int), display_labels=CLASSES,
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Matrice di confusione - {NAMES[key]}")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Curve di calibrazione
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Calibrazione ideale")
    for key, marker in zip(NAMES, ["o", "s"]):
        observed, predicted = calibration_curve(y, probs[key], n_bins=10)
        ax.plot(predicted, observed, marker=marker, label=NAMES[key])
    ax.set_xlabel("Probabilita' media prevista")
    ax.set_ylabel("Frequenza osservata")
    ax.set_title("Confronto della calibrazione")
    ax.legend()
    fig.savefig(RESULTS / "calibration_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Grad-CAM: primo vero positivo della CNN, stessa immagine per entrambi
    index = int(np.where((y == 1) & (probs["cnn"] > 0.5))[0][0])
    for key, (model, device) in models.items():
        save_gradcam(key, model, dataset, index, device)

    print("\n", table.round(4).to_string())
    print(f"\nImmagini di test: {len(y)} - esempio Grad-CAM: indice {index}")
    print(f"File salvati in {RESULTS}")


if __name__ == "__main__":
    main()
