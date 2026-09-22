"""
Silhouette / edge control experiment (the missing piece of Step 4).

Locked protocol: same 40 test content images per class already used as
content in the main cue-conflict experiment (main_experiment.py), converted
to edge/silhouette maps (Canny-style edges via PIL FIND_EDGES + threshold),
classified with the SAME frozen ResNet-50 checkpoint. Reported as
SUPPORTING evidence only (silhouettes are out-of-distribution relative to
the natural-image training data - see project-plan.md caveat).

Uses the exact same seeded content-pool logic as main_experiment.py so the
160 images here (40/class x 4 classes) are the identical set used as
content in the main experiment - no re-sampling, no cherry-picking.

Output: silhouette_trials.jsonl (one record per image) + prints a summary
(accuracy vs baseline test_accuracy) + saves a handful of example
silhouette images under examples/ for the report.
"""

import os
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Config (must match main_experiment.py exactly for the content pool)
# ---------------------------------------------------------------------------
SEED = 42
DATASET_ROOT = os.environ.get(
    "SHAPE_VS_TEXTURE_DATASET_ROOT",
    r"C:\Users\User\Desktop\Deep Learning CNN Project\archive (1)\animal_computer_vision\Dataset",
)
CLASSES = ["Buffalo", "Elephant", "Rhino", "Zebra"]

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(OUT_DIR, "resnet50_baseline_best.pt")
SPLIT_PATH = os.path.join(OUT_DIR, "split_indices.json")
RESULTS_JSONL = os.path.join(OUT_DIR, "silhouette_trials.jsonl")
BASELINE_RESULTS_PATH = os.path.join(OUT_DIR, "baseline_results.json")
EXAMPLES_DIR = os.path.join(OUT_DIR, "examples")
os.makedirs(EXAMPLES_DIR, exist_ok=True)

CONTENT_PER_CLASS = 40
STYLE_IMG_SIZE = 256
CLS_IMG_SIZE = 224
EDGE_THRESHOLD = 40

PILOT_EXCLUDE = {
    os.path.join(DATASET_ROOT, "Buffalo", "Buffalo_991.jpg"),
    os.path.join(DATASET_ROOT, "Buffalo", "Buffalo_992.jpg"),
    os.path.join(DATASET_ROOT, "Buffalo", "Buffalo_993.jpg"),
    os.path.join(DATASET_ROOT, "Buffalo", "Buffalo_994.jpg"),
    os.path.join(DATASET_ROOT, "Buffalo", "Buffalo_995.jpg"),
    os.path.join(DATASET_ROOT, "Zebra", "Zebra_991.jpg"),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ---------------------------------------------------------------------------
# Rebuild the EXACT same content pool as main_experiment.py (same seeds)
# ---------------------------------------------------------------------------
with open(SPLIT_PATH) as f:
    split_raw = json.load(f)
split = {k: [(p, l) for p, l in v] for k, v in split_raw.items()}


def pool_for_class(split_name, cls_idx, n, seed_offset):
    items = [p for p, l in split[split_name] if l == cls_idx and p not in PILOT_EXCLUDE]
    rng = random.Random(SEED + seed_offset)
    rng.shuffle(items)
    assert len(items) >= n, f"Not enough images in {split_name} for class {cls_idx}"
    return items[:n]


content_pools = {c: pool_for_class("test", i, CONTENT_PER_CLASS, seed_offset=i)
                  for i, c in enumerate(CLASSES)}
print("Content pool sizes:", {c: len(v) for c, v in content_pools.items()}, flush=True)

# ---------------------------------------------------------------------------
# Frozen classifier (identical to main_experiment.py)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
cls_resize = T.Resize((CLS_IMG_SIZE, CLS_IMG_SIZE))
sil_loader = T.Compose([T.Resize((STYLE_IMG_SIZE, STYLE_IMG_SIZE))])


def normalize_vgg(x):
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


print("Loading frozen ResNet-50 checkpoint...", flush=True)
clf = models.resnet50(weights=None)
clf.fc = nn.Linear(clf.fc.in_features, len(CLASSES))
clf.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
clf = clf.to(DEVICE).eval()
for p in clf.parameters():
    p.requires_grad_(False)


def classify_pil(img_pil):
    x = T.ToTensor()(cls_resize(img_pil)).unsqueeze(0).to(DEVICE)
    x = normalize_vgg(x)
    with torch.no_grad():
        logits = clf(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    pred_idx = int(torch.tensor(probs).argmax().item())
    return CLASSES[pred_idx], probs


def make_silhouette(img_pil, threshold=EDGE_THRESHOLD):
    gray = img_pil.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    bw = edges.point(lambda p: 255 if p > threshold else 0)
    return bw.convert("RGB")


# ---------------------------------------------------------------------------
# Run: same 160 content images as the main experiment, edge-ified
# ---------------------------------------------------------------------------
records = []
example_saved = {c: False for c in CLASSES}

for cls in CLASSES:
    for content_path in content_pools[cls]:
        img = Image.open(content_path).convert("RGB")
        img = sil_loader(img)
        sil = make_silhouette(img)
        pred_cls, probs = classify_pil(sil)

        records.append({
            "true_class": cls,
            "content_path": content_path,
            "pred_class": pred_cls,
            "correct": pred_cls == cls,
            "confidence": max(probs),
            "probs": probs,
        })

        if not example_saved[cls]:
            sil.save(os.path.join(EXAMPLES_DIR, f"{cls}_silhouette.png"))
            example_saved[cls] = True

with open(RESULTS_JSONL, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

n_correct = sum(r["correct"] for r in records)
n_total = len(records)
sil_acc = n_correct / n_total
avg_conf = sum(r["confidence"] for r in records) / n_total

print(f"\n=== SILHOUETTE CONTROL RESULTS (n={n_total}) ===")
print(f"Accuracy: {n_correct}/{n_total} = {sil_acc:.3f}")
print(f"Avg confidence: {avg_conf:.3f}")

per_class = {}
for cls in CLASSES:
    cls_records = [r for r in records if r["true_class"] == cls]
    n = len(cls_records)
    c = sum(r["correct"] for r in cls_records)
    per_class[cls] = c / n
    print(f"  {cls}: {c}/{n} = {c/n:.3f}")

if os.path.exists(BASELINE_RESULTS_PATH):
    with open(BASELINE_RESULTS_PATH) as f:
        baseline = json.load(f)
    print(f"\nBaseline (natural images) test_accuracy = {baseline['test_accuracy']:.3f}")
    print(f"Silhouette accuracy                       = {sil_acc:.3f}")
    print(f"Drop = {baseline['test_accuracy'] - sil_acc:.3f}")
    print("(Reminder: silhouettes are out-of-distribution relative to training "
          "data, so this comparison is supporting evidence only per project-plan.md)")

summary = {
    "n": n_total,
    "accuracy": sil_acc,
    "avg_confidence": avg_conf,
    "per_class_accuracy": per_class,
    "edge_threshold": EDGE_THRESHOLD,
}
with open(os.path.join(OUT_DIR, "silhouette_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {RESULTS_JSONL}")
print(f"Saved: silhouette_summary.json")
print(f"Saved example silhouettes (1/class) under: {EXAMPLES_DIR}")
