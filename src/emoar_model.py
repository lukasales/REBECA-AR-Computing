from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms


CLASSES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


def build_emoar_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, 7)
    return model


def load_emoar_model(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = build_emoar_model()

    # Strict loading is intentional.
    # Any architecture mismatch must fail explicitly.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model


def historical_web_transform():
    """
    Reproduces the preprocessing declared in EmoAR web_app/commons.py.

    RGB conversion is explicit here because the model expects
    three input channels.
    """

    return transforms.Compose([
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict(model, image):
    transform = historical_web_transform()

    tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_index = probabilities.max(dim=1)

    index = int(predicted_index.item())

    return {
        "class_index": index,
        "class_name": CLASSES[index],
        "confidence": float(confidence.item()),
        "probabilities": {
            name: float(probabilities[0, i].item())
            for i, name in enumerate(CLASSES)
        },
        "input_shape": list(tensor.shape),
        "logits_shape": list(logits.shape),
    }
