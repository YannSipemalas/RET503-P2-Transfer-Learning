#!/usr/bin/env python3
import argparse
import csv
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
MODES = ("feature", "partial", "scratch")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MetadataDataset(Dataset):
    def __init__(self, rows, class_to_idx, transform):
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = ROOT / row["path"]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), self.class_to_idx[row["label"]]


def load_metadata(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "validation"]
    if not train_rows or not val_rows:
        raise ValueError("No train/validation split found; run split.py first")
    classes = sorted({row["label"] for row in rows if row.get("split") != "excluded"})
    for split_name, split_rows in (("train", train_rows), ("validation", val_rows)):
        missing = set(classes) - {row["label"] for row in split_rows}
        if missing:
            raise ValueError(f"{split_name} split is missing classes: {sorted(missing)}")
        for row in split_rows:
            if not (ROOT / row["path"]).is_file():
                raise FileNotFoundError(ROOT / row["path"])
    return train_rows, val_rows, {name: index for index, name in enumerate(classes)}


def make_model(mode: str, num_classes: int, pretrained: bool = True):
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose from {MODES}")
    weights = models.ResNet18_Weights.DEFAULT if pretrained and mode != "scratch" else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if mode == "feature":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    elif mode == "partial":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
    return model


def make_optimizer(model, mode):
    if mode == "partial":
        return torch.optim.Adam([
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ])
    return torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return loss_sum / count, correct / count


def train_mode(mode, train_loader, val_loader, class_to_idx, epochs, device, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_model(mode, len(class_to_idx), pretrained=(mode != "scratch")).to(device)
    optimizer = make_optimizer(model, mode)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_accuracy = -1.0
    best_epoch = 0
    model_path = ROOT / "models" / f"resnet18_{mode}.pth"
    log_path = ROOT / "results" / f"{mode}_history.csv"
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = correct = count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
        train_loss, train_accuracy = loss_sum / count, correct / count
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                  "val_loss": val_loss, "val_accuracy": val_accuracy}
        history.append(record)
        print(f"{mode} epoch {epoch:02d}/{epochs}: train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            torch.save({"state_dict": model.state_dict(), "mode": mode, "class_to_idx": class_to_idx,
                        "input_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                        "epoch": epoch, "val_accuracy": val_accuracy}, model_path)

    elapsed = time.perf_counter() - started
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return {"mode": mode, "best_val_accuracy": best_accuracy, "best_epoch": best_epoch,
            "epochs_to_90_percent": next((r["epoch"] for r in history if r["val_accuracy"] >= .9), ""),
            "training_seconds": elapsed, "checkpoint": str(model_path.relative_to(ROOT))}


def update_results(results):
    path = ROOT / "results" / "experiments.csv"
    existing = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = {row["mode"]: row for row in csv.DictReader(handle)}
    for result in results:
        existing[result["mode"]] = result
    columns = ["mode", "best_val_accuracy", "best_epoch", "epochs_to_90_percent", "training_seconds", "checkpoint"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for mode in MODES:
            if mode in existing:
                writer.writerow(existing[mode])


def plot_results():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for mode in MODES:
        path = ROOT / "results" / f"{mode}_history.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            ax.plot([int(r["epoch"]) for r in rows], [float(r["val_accuracy"]) for r in rows], marker="o", label=mode)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set(xlabel="Epoch", ylabel="Validation accuracy", title="ResNet-18 mode comparison")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "accuracy_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--metadata", type=Path, default=ROOT / "metadata.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_rows, val_rows, class_to_idx = load_metadata(args.metadata)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.05),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = MetadataDataset(train_rows, class_to_idx, train_transform)
    val_ds = MetadataDataset(val_rows, class_to_idx, val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    modes = MODES if args.mode == "all" else (args.mode,)
    results = [train_mode(mode, train_loader, val_loader, class_to_idx, args.epochs, device, args.seed)
               for mode in modes]
    update_results(results)
    plot_results()


if __name__ == "__main__":
    main()
