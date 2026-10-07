#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import models, transforms

if len(sys.argv) < 2:
    print("Cara pakai: python predict.py <path_foto.jpg>")
    print("Contoh   : python predict.py box_cokelat/box_cokelat__frame_000000000.jpg")
    sys.exit(1)

image_path = Path(sys.argv[1])
if not image_path.exists():
    print(f"Error: File '{image_path}' tidak ditemukan!")
    sys.exit(1)

checkpoint_path = Path("models/resnet18_partial.pth")
checkpoint = torch.load(checkpoint_path)

class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {idx: name for name, idx in class_to_idx.items()}

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)[0]
    pred_idx = torch.argmax(probabilities).item()
    confidence = probabilities[pred_idx].item() * 100

predicted_label = idx_to_class[pred_idx]

print("=" * 40)
print(f" File Gambar    : {image_path.name}")
print(f" Hasil Prediksi : {predicted_label.upper()}")
print(f" Tingkat Keyakinan (Confidence): {confidence:.2f}%")
print("=" * 40)