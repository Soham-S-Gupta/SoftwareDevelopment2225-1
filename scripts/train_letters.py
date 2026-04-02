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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from src.asl_translator.config import load_project_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(train_dir: str, image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = datasets.ImageFolder(train_dir)
    train_indices, val_indices = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.15,
        stratify=full_dataset.targets,
        random_state=42,
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(train_dir, transform=eval_transform)
    return Subset(train_dataset, train_indices), Subset(val_dataset, val_indices), full_dataset.classes


def build_model(model_name: str, num_classes: int, pretrained: bool) -> torch.nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


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

    train_dataset, val_dataset, classes = build_datasets(
        train_dir=config["paths"]["letter_train_dir"],
        image_size=config["training"]["image_size"],
    )

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        model_name=config["training"]["model_name"],
        pretrained=True,
        num_classes=len(classes),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate_letters"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_val_acc = 0.0
    checkpoint_path = Path(config["paths"]["letter_checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path = Path(config["paths"]["letter_labels"])
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["epochs_letters"]):
        model.train()
        progress = tqdm(train_loader, desc=f"letters epoch {epoch + 1}")
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
                checkpoint_path,
            )
            with open(labels_path, "w", encoding="utf-8") as handle:
                json.dump(classes, handle, indent=2)

    print(f"best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
