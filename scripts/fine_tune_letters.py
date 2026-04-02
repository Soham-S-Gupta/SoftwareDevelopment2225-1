import argparse
import json
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import timm
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from src.asl_translator.config import load_project_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform(image_size: int, train: bool):
    items = [transforms.Resize((image_size, image_size))]
    if train:
        items.extend(
            [
                transforms.RandomRotation(8),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            ]
        )
    items.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(items)


class FixedLabelImageFolder(Dataset):
    def __init__(self, root: str, class_to_index: dict[str, int], transform=None) -> None:
        self.root = Path(root)
        self.class_to_index = class_to_index
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for class_name, class_index in class_to_index.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for image_path in class_dir.glob("*"):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                self.samples.append((image_path, class_index))
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_datasets(base_dir: str, custom_dir: str, image_size: int, validation_split: float):
    train_transform = build_transform(image_size, train=True)
    eval_transform = build_transform(image_size, train=False)

    base_dataset = datasets.ImageFolder(base_dir)
    class_to_index = {name: idx for idx, name in enumerate(base_dataset.classes)}
    custom_dataset = FixedLabelImageFolder(custom_dir, class_to_index=class_to_index)
    if len(custom_dataset) == 0:
        raise ValueError("No custom letter images were found. Capture images first.")

    combined_targets = base_dataset.targets + custom_dataset.targets
    combined_size = len(base_dataset) + len(custom_dataset)
    train_indices, val_indices = train_test_split(
        list(range(combined_size)),
        test_size=validation_split,
        stratify=combined_targets,
        random_state=42,
    )

    train_base = datasets.ImageFolder(base_dir, transform=train_transform)
    train_custom = FixedLabelImageFolder(custom_dir, class_to_index=class_to_index, transform=train_transform)
    eval_base = datasets.ImageFolder(base_dir, transform=eval_transform)
    eval_custom = FixedLabelImageFolder(custom_dir, class_to_index=class_to_index, transform=eval_transform)

    train_dataset = ConcatDataset([train_base, train_custom])
    val_dataset = ConcatDataset([eval_base, eval_custom])
    return Subset(train_dataset, train_indices), Subset(val_dataset, val_indices), base_dataset.classes


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)
    return total_loss / total_examples, total_correct / total_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_project_config(args.config)
    set_seed(config["training"]["seed"])

    base_dir = Path(config["paths"]["letter_train_dir"])
    custom_dir = Path(config["paths"]["custom_letter_train_dir"])
    base_checkpoint = Path(config["paths"]["letter_checkpoint"])
    custom_checkpoint = Path(config["paths"]["custom_letter_checkpoint"])
    labels_path = Path(config["paths"]["letter_labels"])

    if not base_checkpoint.exists():
        raise FileNotFoundError("Run the 12-epoch base letter training first.")
    if not custom_dir.exists():
        raise FileNotFoundError("Collect your custom letter images first.")

    train_dataset, val_dataset, classes = build_datasets(
        base_dir=str(base_dir),
        custom_dir=str(custom_dir),
        image_size=config["training"]["image_size"],
        validation_split=config["training"]["validation_split"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_state = torch.load(base_checkpoint, map_location=device)
    model = timm.create_model(
        config["training"]["model_name"],
        pretrained=False,
        num_classes=len(classes),
    ).to(device)
    model.load_state_dict(base_state["model_state"])

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate_fine_tune"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_val_acc = 0.0
    custom_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["fine_tune_epochs_letters"]):
        model.train()
        progress = tqdm(train_loader, desc=f"fine tune epoch {epoch + 1}")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"epoch={epoch + 1} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "model_name": config["training"]["model_name"],
                    "image_size": config["training"]["image_size"],
                },
                custom_checkpoint,
            )
            with open(labels_path, "w", encoding="utf-8") as handle:
                json.dump(classes, handle, indent=2)

    print(f"best fine-tuned validation accuracy: {best_val_acc:.4f}")
    print(f"saved fine-tuned checkpoint to {custom_checkpoint}")


if __name__ == "__main__":
    main()
