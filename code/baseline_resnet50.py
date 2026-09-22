"""
Step 3 of the locked protocol: Baseline ResNet-50.

Flow (locked order - do not reorder):
  Train -> Validation -> pick best checkpoint (validation only) -> Freeze
  -> Test set (first and only touch) -> report baseline metrics.

Split: 70/15/15 per class, fixed seed, from the 4,000-image balanced
dataset (1,000/class: Buffalo, Elephant, Rhino, Zebra).
  Train      = 700/class (2,800 total) -> also the style-reference pool
               for the main cue-conflict experiment
  Validation = 150/class (600 total)   -> checkpoint selection only
  Test       = 150/class (600 total)   -> touched once, after freeze;
               also the content pool for the main experiment

Model: ResNet-50, ImageNet-pretrained, backbone frozen except layer4,
fc replaced with a 4-class head. Fine-tuning only the last block keeps
CPU training tractable while still adapting high-level features.
"""

import os
import glob
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
DATASET_ROOT = os.environ.get(
    "SHAPE_VS_TEXTURE_DATASET_ROOT",
    r"C:\Users\User\Desktop\Deep Learning CNN Project\archive (1)\animal_computer_vision\Dataset",
)
CLASSES = ["Buffalo", "Elephant", "Rhino", "Zebra"]
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 4
LR = 1e-4
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(OUT_DIR, "resnet50_baseline_best.pt")
SPLIT_PATH = os.path.join(OUT_DIR, "split_indices.json")
RESULTS_PATH = os.path.join(OUT_DIR, "baseline_results.json")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Build fixed 70/15/15 split, per class, seeded
# ---------------------------------------------------------------------------
def build_split():
    split = {"train": [], "val": [], "test": []}
    rng = random.Random(SEED)
    for label, cls in enumerate(CLASSES):
        files = sorted(glob.glob(os.path.join(DATASET_ROOT, cls, "*.jpg")))
        assert len(files) == 1000, f"{cls}: expected 1000, found {len(files)}"
        idx = list(range(len(files)))
        rng.shuffle(idx)
        n_train, n_val = 700, 150
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        for i in train_idx:
            split["train"].append((files[i], label))
        for i in val_idx:
            split["val"].append((files[i], label))
        for i in test_idx:
            split["test"].append((files[i], label))
    return split


split = build_split()
print(f"Train: {len(split['train'])} | Val: {len(split['val'])} | Test: {len(split['test'])}")

with open(SPLIT_PATH, "w") as f:
    json.dump({k: [[p, l] for p, l in v] for k, v in split.items()}, f, indent=2)
print(f"Saved split indices to {SPLIT_PATH}")


# ---------------------------------------------------------------------------
# Dataset / transforms
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class AnimalDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


train_loader = DataLoader(AnimalDataset(split["train"], train_tf),
                           batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(AnimalDataset(split["val"], eval_tf),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(AnimalDataset(split["test"], eval_tf),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# Model: ResNet-50, freeze all but layer4 + fc
# ---------------------------------------------------------------------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
for name, param in model.named_parameters():
    param.requires_grad = name.startswith("layer4") or name.startswith("fc")
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(DEVICE)

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable, lr=LR)
criterion = nn.CrossEntropyLoss()


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1, all_labels, all_preds


# ---------------------------------------------------------------------------
# Train with early stopping on validation loss
# ---------------------------------------------------------------------------
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc, train_f1, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_acc, val_f1, _, _ = run_epoch(val_loader, train=False)
    print(f"Epoch {epoch:2d} | train_loss={train_loss:.4f} acc={train_acc:.3f} | "
          f"val_loss={val_loss:.4f} acc={val_acc:.3f} macro_f1={val_f1:.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"  -> new best checkpoint saved (val_loss={val_loss:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

print(f"\nBest checkpoint: {CHECKPOINT_PATH} (val_loss={best_val_loss:.4f})")


# ---------------------------------------------------------------------------
# Freeze best checkpoint, touch test set for the first and only time
# ---------------------------------------------------------------------------
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

test_loss, test_acc, test_f1, test_labels, test_preds = run_epoch(test_loader, train=False)
report = classification_report(test_labels, test_preds, target_names=CLASSES, output_dict=True)
cm = confusion_matrix(test_labels, test_preds).tolist()

print("\n=== BASELINE TEST RESULTS (frozen checkpoint, touched once) ===")
print(f"Overall accuracy: {test_acc:.4f}")
print(f"Macro F1:         {test_f1:.4f}")
print(classification_report(test_labels, test_preds, target_names=CLASSES))
print("Confusion matrix (rows=true, cols=pred):")
print(np.array(cm))

results = {
    "test_accuracy": test_acc,
    "test_macro_f1": test_f1,
    "per_class_report": report,
    "confusion_matrix": cm,
    "classes": CLASSES,
    "best_val_loss": best_val_loss,
    "seed": SEED,
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {RESULTS_PATH}")
