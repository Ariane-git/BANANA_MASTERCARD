"""Outils simples et partagés pour le module de classification des bananes."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


SEED = 42
IMAGE_SIZE = 160
BATCH_SIZE = 32
CLASS_NAMES = ("overripe", "ripe", "rotten", "unripe")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ImageFile.LOAD_TRUNCATED_IMAGES = False


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_root() -> Path:
    return project_root() / "Banana Ripeness Classification Dataset"


def build_transforms(image_size: int = IMAGE_SIZE):
    """L'augmentation est réservée au train ; valid/test ont un prétraitement fixe."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def make_loaders(root: Path | None = None, image_size: int = IMAGE_SIZE, num_workers: int = 0):
    root = Path(root or dataset_root())
    train_transform, evaluation_transform = build_transforms(image_size)
    train_data = datasets.ImageFolder(root / "train", transform=train_transform)
    valid_data = datasets.ImageFolder(root / "valid", transform=evaluation_transform)
    test_data = datasets.ImageFolder(root / "test", transform=evaluation_transform)
    if tuple(train_data.classes) != CLASS_NAMES:
        raise ValueError(f"Classes inattendues dans train : {train_data.classes}")
    for name, dataset in (("valid", valid_data), ("test", test_data)):
        if dataset.classes != train_data.classes:
            raise ValueError(f"Les classes de {name} diffèrent de celles de train.")
    loaders = {
        "train": DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers),
        "valid": DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers),
    }
    return train_data, valid_data, test_data, loaders


class BananaCNN(nn.Module):
    """CNN de base compact : convolutions, normalisation, pooling et dropout."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(192, num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_resnet18(num_classes: int = 4, pretrained: bool = True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _keep_frozen_batchnorm_in_eval(model) -> None:
    """Correction : une couche BatchNorm gelée ne doit pas mettre à jour ses statistiques."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()


def _run_epoch(model, loader, loss_function, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    if is_training:
        _keep_frozen_batchnorm_in_eval(model)
    total_loss = 0.0
    total_items = 0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = loss_function(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_items += labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    return total_loss / max(total_items, 1), correct / max(total_items, 1)


def fit_model(
    model,
    loaders,
    class_weights,
    device,
    model_name: str,
    output_dir: Path,
    epochs: int = 8,
    patience: int = 3,
    fine_tune_resnet: bool = False,
):
    """Entraîne, arrête tôt sur la validation et restaure les meilleurs poids."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    if fine_tune_resnet:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("layer4") or name.startswith("fc")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=2e-4 if fine_tune_resnet else 1e-3,
        weight_decay=1e-4,
    )
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        train_loss, train_accuracy = _run_epoch(model, loaders["train"], criterion, device, optimizer)
        valid_loss, valid_accuracy = _run_epoch(model, loaders["valid"], criterion, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_accuracy": train_accuracy,
                "valid_accuracy": valid_accuracy,
            }
        )
        print(
            f"{model_name} | époque {epoch + 1}/{epochs} | "
            f"loss train {train_loss:.4f} | val {valid_loss:.4f} | "
            f"accuracy val {valid_accuracy:.3f}"
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, output_dir / f"{model_name}_best.pt")
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Arrêt anticipé : la perte de validation ne s'améliore plus.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    return model, history, elapsed


@torch.inference_mode()
def evaluate_model(model, loader, device):
    model.eval()
    actual, predicted = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        actual.extend(labels.numpy().tolist())
        predicted.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "report": classification_report(
            actual, predicted, labels=list(range(len(CLASS_NAMES))),
            target_names=list(CLASS_NAMES), output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=list(range(len(CLASS_NAMES)))
        ).tolist(),
        "y_true": actual,
        "y_pred": predicted,
    }


def save_json(data, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_image_model(project: Path | None = None, device: str = "cpu"):
    project = Path(project or project_root())
    metadata_path = project / "models" / "image_metadata.json"
    weights_path = project / "models" / "image_model.pt"
    if not metadata_path.exists() or not weights_path.exists():
        raise FileNotFoundError("Modèle image absent : exécute d'abord le notebook 01.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["architecture"] == "resnet18":
        model = build_resnet18(len(metadata["classes"]), pretrained=False)
    else:
        model = BananaCNN(len(metadata["classes"]))
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    _, evaluation_transform = build_transforms(metadata["image_size"])
    return model, metadata, evaluation_transform


def predict_pil_image(image: Image.Image, model, metadata, image_transform, device="cpu"):
    image = image.convert("RGB")
    tensor = image_transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return metadata["classes"][int(probabilities.argmax())], probabilities
