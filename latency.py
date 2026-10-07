#!/usr/bin/env python3
"""Measure batch-one inference latency for a trained ResNet-18 checkpoint."""

import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="image path relative to project root or absolute")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "latency.csv")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be non-negative and iterations must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    image_path = args.image if args.image.is_absolute() else ROOT / args.image
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    preprocessing = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(checkpoint.get("mean", (.485, .456, .406)),
                             checkpoint.get("std", (.229, .224, .225))),
    ])
    with Image.open(image_path) as image:
        tensor = preprocessing(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - start) * 1000)
    result = {"checkpoint": str(checkpoint_path), "device": str(device), "batch_size": 1,
              "warmup_iterations": args.warmup, "measured_iterations": args.iterations,
              "mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples),
              "p95_ms": sorted(samples)[max(0, math.ceil(.95 * len(samples)) - 1)],
              "image": str(image_path)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exists = args.output.exists()
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        if not exists:
            writer.writeheader()
        writer.writerow(result)
    print(f"{device}: median={result['median_ms']:.3f} ms, mean={result['mean_ms']:.3f} ms, p95={result['p95_ms']:.3f} ms")


if __name__ == "__main__":
    main()
