"""Model-loading and image preprocessing for Group 07 EcoSort Vision.

The model is the genuine Milestone 4 MobileNetV2 checkpoint. This package does
not download other checkpoints or use an external model service.
"""
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageOps

CHECKPOINT_PATH = Path(__file__).resolve().parent / "mobilenet_v2_best.pth"
EXPECTED_CLASSES = [
    "Cardboard", "Food Organics", "Glass", "Metal",
    "Miscellaneous Trash", "Paper", "Plastic", "Textile Trash", "Vegetation"
]
REVIEW_THRESHOLD = 0.54  # Exploratory validation choice, not a safety guarantee.


def load_model():
    """Load on CPU so Streamlit Community Cloud does not need GPU access."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {CHECKPOINT_PATH.name}. Put the provided checkpoint next to app.py."
        )
    torch.set_num_threads(min(2, torch.get_num_threads()))
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if ckpt.get("model_name") != "mobilenet_v2":
        raise ValueError("The checkpoint is not a MobileNetV2 model.")
    classes = list(ckpt.get("classes", []))
    if classes != EXPECTED_CLASSES:
        raise ValueError("The saved model's nine class labels do not match this app.")
    size = int(ckpt.get("image_size", 224))
    if size != 224:
        raise ValueError(f"Unexpected checkpoint image size: {size}")

    model = models.mobilenet_v2(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, preprocess, classes


def predict(image: Image.Image, model, preprocess, classes):
    """Return a list of 9 (class_name, softmax_probability) pairs, largest first."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    x = preprocess(image).unsqueeze(0)  # CPU tensor, shape [1, 3, 224, 224]
    with torch.inference_mode():
        probabilities = torch.softmax(model(x), dim=1).squeeze(0).tolist()
    return sorted(zip(classes, (float(v) for v in probabilities)), key=lambda x: x[1], reverse=True)
