"""Addestramento della CNN o del ViT-B/16 su PneumoniaMNIST.

Riproduce la configurazione usata nel notebook. I nuovi pesi sono salvati in
retrain/ per non sovrascrivere quelli usati nei risultati.

Uso: python training/train.py --model cnn
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from medmnist import PneumoniaMNIST
from monai.transforms import RandFlip
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy_app"))
from src.inference import get_transform  # noqa: E402
from src.model_definitions import PneumoniaCNN, create_vit  # noqa: E402

SEED = 42
CONFIG = {
    "cnn": {"batch_size": 64, "lr": 1e-4, "optimizer": torch.optim.Adam, "weight_decay": 0.0},
    "vit": {"batch_size": 8, "lr": 1e-5, "optimizer": torch.optim.AdamW, "weight_decay": 1e-4},
}


class PneumoniaDataset(Dataset):
    """Applica preprocessing (e flip in training) alle immagini MedMNIST."""

    def __init__(self, base, model_name, augment=False):
        self.base = base
        self.transform = get_transform(model_name)
        self.flip = RandFlip(prob=0.5, spatial_axis=1) if augment else None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        x = self.transform(np.array(image))
        if self.flip is not None:
            x = self.flip(x)
        return torch.as_tensor(x), int(label[0])


def class_weights(labels):
    """Pesi inversamente proporzionali alla frequenza delle classi."""
    counts = np.bincount(labels, minlength=2)
    return torch.tensor(len(labels) / (2 * counts), dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """Un'epoca di training (se c'e' l'ottimizzatore) o di validazione."""
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, total = 0.0, 0, 0
    use_amp = scaler is not None and scaler.is_enabled()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        loss_sum += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CONFIG, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "retrain")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CONFIG[args.model]
    args.data.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    train, val = (PneumoniaMNIST(split=s, root=str(args.data), download=True, as_rgb=True)
                  for s in ("train", "val"))
    train_loader = DataLoader(PneumoniaDataset(train, args.model, augment=True),
                              batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(PneumoniaDataset(val, args.model),
                            batch_size=cfg["batch_size"])

    model = (PneumoniaCNN() if args.model == "cnn" else create_vit()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train.labels.flatten()).to(device))
    optimizer = cfg["optimizer"](model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    # Precisione mista solo per il ViT su GPU
    scaler = torch.amp.GradScaler("cuda", enabled=args.model == "vit" and device.type == "cuda")

    best, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(history[-1], flush=True)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), args.out / f"best_pneumonia_{args.model}.pth")
        pd.DataFrame(history).to_csv(args.out / f"history_{args.model}.csv", index=False)


if __name__ == "__main__":
    main()
