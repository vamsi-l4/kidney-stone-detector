# train.py
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from sklearn.model_selection import StratifiedShuffleSplit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(data_dir: str, batch_size: int = 32, val_split=0.2, test_split=0.1):
    data_dir = Path(data_dir)
    assert data_dir.exists(), f"Data folder not found: {data_dir}"

    # Base dataset to read labels
    base = ImageFolder(str(data_dir))
    y = base.targets  # class indices per sample

    # Get default weights (ImageNet pretrained) and transforms
    weights = models.MobileNet_V2_Weights.DEFAULT
    preprocess = weights.transforms()

    # Training augmentations + preprocess
    train_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        preprocess,  # includes ToTensor + Normalize
    ])

    eval_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        preprocess,
    ])

    # Stratified Train/Val/Test split
    idx_all = list(range(len(y)))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=42)
    trainval_idx, test_idx = next(sss1.split(idx_all, y))

    y_trainval = [y[i] for i in trainval_idx]
    val_ratio = val_split / (1.0 - test_split)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx_rel, val_idx_rel = next(sss2.split(trainval_idx, y_trainval))

    # Map back to original indices
    train_idx = [trainval_idx[i] for i in train_idx_rel]
    val_idx = [trainval_idx[i] for i in val_idx_rel]

    train_ds_full = ImageFolder(str(data_dir), transform=train_tfms)
    eval_ds_full = ImageFolder(str(data_dir), transform=eval_tfms)

    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(eval_ds_full, val_idx)
    test_ds = Subset(eval_ds_full, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ✅ FIXED
    class_to_idx = train_ds_full.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return train_loader, val_loader, test_loader, idx_to_class


def build_model(num_classes: int):
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    # Freeze feature extractor
    for p in model.features.parameters():
        p.requires_grad = False
    # Replace classifier head
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def eval_model(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='mixed_subset')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, idx_to_class = build_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    num_classes = len(idx_to_class)
    model = build_model(num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val_acc = 0.0
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_model(model, val_loader, criterion)
        print(f"Epoch {epoch:02d}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'state_dict': model.state_dict(), 'idx_to_class': idx_to_class}, models_dir / 'kidney_mobilenet_v2.pt')
            with open(models_dir / 'labels.json', 'w') as f:
                json.dump(idx_to_class, f)
            print(f"✓ Saved best model (val_acc={best_val_acc:.4f})")

    # Load best and evaluate on test set
    ckpt = torch.load(models_dir / 'kidney_mobilenet_v2.pt', map_location=DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    test_loss, test_acc = eval_model(model, test_loader, criterion)
    print(f"\nTest Loss {test_loss:.4f} | Test Acc {test_acc:.4f}")


if __name__ == '__main__':
    main()
