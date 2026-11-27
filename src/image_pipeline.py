"""Outils simples et partagés pour le module de classification des bananes."""

from __future__ import annotations

import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


# ---------------------------------------------------------------------------
# Chargement rapide : les images sont lues UNE SEULE FOIS, redimensionnées, puis
# gardées en mémoire (uint8). Chaque époque ne fait plus de décodage JPEG.
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def build_image_cache(root: Path, image_size: int, cache_path: Path) -> dict:
    """Lit tout le dataset, redimensionne comme à l'inférence et sauvegarde un cache."""
    root = Path(root)
    resize = transforms.Resize((image_size, image_size))  # même redimensionnement que build_transforms
    cache = {"classes": list(CLASS_NAMES), "image_size": image_size, "unreadable": []}

    def load_one(path):
        try:
            with Image.open(path) as image:
                image = resize(image.convert("RGB"))
            return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).contiguous()
        except Exception:
            return None

    for split in ("train", "valid", "test"):
        paths, labels = [], []
        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = root / split / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Dossier de classe introuvable : {split}/{class_name}")
            for path in _list_images(class_dir):
                paths.append(path)
                labels.append(label)
        with ThreadPoolExecutor(max_workers=4) as pool:
            tensors = list(pool.map(load_one, paths))
        kept = [index for index, tensor in enumerate(tensors) if tensor is not None]
        cache["unreadable"] += [str(paths[i].relative_to(root)) for i, t in enumerate(tensors) if t is None]
        cache[split] = (
            torch.stack([tensors[i] for i in kept]),
            torch.tensor([labels[i] for i in kept], dtype=torch.long),
        )
        print(f"Cache {split} : {len(kept)} images lues")
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache


def _subset_per_class(images, labels, per_class: int, seed: int = SEED):
    """Garde au plus `per_class` images par classe (tirage reproductible) : test rapide."""
    generator = torch.Generator().manual_seed(seed)
    kept = []
    for class_index in range(len(CLASS_NAMES)):
        indices = (labels == class_index).nonzero().flatten()
        kept.append(indices[torch.randperm(len(indices), generator=generator)[:per_class]])
    kept = torch.cat(kept).sort().values
    return images[kept], labels[kept]


def _augment_batch(batch: torch.Tensor) -> torch.Tensor:
    """Augmentation légère sur un lot dans [0, 1] : miroir, rotation ±10°, luminosité, contraste, saturation."""
    count = batch.size(0)
    flip = torch.rand(count) < 0.5
    batch = torch.where(flip.view(-1, 1, 1, 1), batch.flip(3), batch)
    angle = (torch.rand(count) * 2 - 1) * math.radians(10)
    theta = torch.zeros(count, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1] = torch.cos(angle), -torch.sin(angle)
    theta[:, 1, 0], theta[:, 1, 1] = torch.sin(angle), torch.cos(angle)
    grid = F.affine_grid(theta, list(batch.shape), align_corners=False)
    batch = F.grid_sample(batch, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    brightness = 1 + (torch.rand(count, 1, 1, 1) * 2 - 1) * 0.12
    contrast = 1 + (torch.rand(count, 1, 1, 1) * 2 - 1) * 0.12
    saturation = 1 + (torch.rand(count, 1, 1, 1) * 2 - 1) * 0.08
    gray = (0.299 * batch[:, 0:1] + 0.587 * batch[:, 1:2] + 0.114 * batch[:, 2:3])
    batch = (batch - gray) * saturation + gray
    mean = batch.mean(dim=(1, 2, 3), keepdim=True)
    batch = ((batch - mean) * contrast + mean) * brightness
    return batch.clamp(0, 1)


class CachedDataset:
    """Images en mémoire (uint8) et étiquettes, avec les attributs utiles du notebook."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels
        self.targets = labels.tolist()
        self.classes = list(CLASS_NAMES)

    def __len__(self):
        return len(self.labels)


class CachedLoader:
    """Fournit des lots (images normalisées, étiquettes), comme un DataLoader."""

    def __init__(self, dataset: CachedDataset, batch_size: int, shuffle: bool, augment: bool):
        self.dataset, self.batch_size, self.shuffle, self.augment = dataset, batch_size, shuffle, augment
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        count = len(self.dataset)
        order = torch.randperm(count) if self.shuffle else torch.arange(count)
        for start in range(0, count, self.batch_size):
            indices = order[start:start + self.batch_size]
            batch = self.dataset.images[indices].float() / 255.0
            if self.augment:
                batch = _augment_batch(batch)
            yield (batch - self.mean) / self.std, self.dataset.labels[indices]


def make_cached_loaders(root: Path | None = None, image_size: int = IMAGE_SIZE,
                        cache_dir: Path | None = None, batch_size: int = BATCH_SIZE,
                        subset_per_class: int | None = None):
    """Charge (ou crée) le cache d'images, puis renvoie datasets et loaders prêts à l'emploi."""
    root = Path(root or dataset_root())
    cache_path = Path(cache_dir or project_root() / "cache") / f"images_{image_size}.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"Cache chargé : {cache_path.name}")
    else:
        cache = build_image_cache(root, image_size, cache_path)
    if cache["classes"] != list(CLASS_NAMES):
        raise ValueError("Le cache ne correspond pas aux classes attendues : supprime le dossier cache.")
    train_images, train_labels = cache["train"]
    valid_images, valid_labels = cache["valid"]
    if subset_per_class:
        train_images, train_labels = _subset_per_class(train_images, train_labels, subset_per_class)
        valid_images, valid_labels = _subset_per_class(valid_images, valid_labels, subset_per_class)
    train_data = CachedDataset(train_images, train_labels)
    valid_data = CachedDataset(valid_images, valid_labels)
    test_data = CachedDataset(*cache["test"])
    loaders = {
        "train": CachedLoader(train_data, batch_size, shuffle=True, augment=True),
        "valid": CachedLoader(valid_data, batch_size, shuffle=False, augment=False),
        "test": CachedLoader(test_data, batch_size, shuffle=False, augment=False),
    }
    return train_data, valid_data, test_data, loaders, cache["unreadable"]


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


# ---------------------------------------------------------------------------
# Garde-fou de l'application : vérifie qu'une photo contient bien une banane
# (MobileNetV3 pré-entraîné sur ImageNet) avant de prédire la maturité.
# ---------------------------------------------------------------------------
def load_banana_guard(project: Path | None = None, device: str = "cpu"):
    """Charge MobileNetV3 (ImageNet) et renvoie (modèle, prétraitement, indice « banana », catégories)."""
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    detector = models.mobilenet_v3_large(weights=weights).to(device).eval()
    categories = list(weights.meta["categories"])
    return detector, weights.transforms(), categories.index("banana"), categories


def check_banana_image(image: Image.Image, detector, detector_transform, banana_index: int, categories,
                       top_k: int = 5, min_score: float = 0.05):
    """Renvoie (est_banane, score_banane, catégorie_dominante, rang_banane) ; acceptée si top_k et score minimal."""
    device = next(detector.parameters()).device
    tensor = detector_transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(detector(tensor), dim=1)[0].cpu()
    order = probabilities.argsort(descending=True)
    rank = int((order == banana_index).nonzero().item()) + 1
    score = float(probabilities[banana_index])
    return (rank <= top_k and score >= min_score), score, categories[int(order[0])], rank
