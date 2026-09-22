"""
Step 4 of the locked protocol: main cue-conflict experiment.

12 directions (shape_class, texture_class), shape_class != texture_class.
Per direction: 40 content images (test split, shape class) x 3 style
images (train split, texture class) = 120 trials. Total 12*120 = 1,440.

Content pool  = TEST split only (40/class, seeded, pilot files excluded)
Style pool    = TRAIN split only (3/class, seeded, pilot files excluded)
Model         = frozen ResNet-50 checkpoint from baseline (resnet50_baseline_best.pt)
Style method  = Gatys (VGG19, LBFGS, steps=150) - locked from pilot

Resumable: each finished trial is appended to cue_conflict_trials.jsonl
immediately; on restart, already-completed trial ids are skipped.

One example stylized image per direction is saved under examples/ for
the report.
"""

import os
import glob
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

# ---------------------------------------------------------------------------
# Config
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
RESULTS_JSONL = os.path.join(OUT_DIR, "cue_conflict_trials.jsonl")
EXAMPLES_DIR = os.path.join(OUT_DIR, "examples")
os.makedirs(EXAMPLES_DIR, exist_ok=True)

CONTENT_PER_CLASS = 40
STYLE_PER_CLASS = 3

STYLE_IMG_SIZE = 256
CLS_IMG_SIZE = 224
GATYS_STEPS = 150
CONTENT_WEIGHT = 1e0
STYLE_WEIGHT = 1e6

# Exact files used during pilot runs - must never appear in real trials
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
# Load frozen split + build content/style pools (seeded, pilot-excluded)
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
style_pools = {c: pool_for_class("train", i, STYLE_PER_CLASS, seed_offset=100 + i)
                for i, c in enumerate(CLASSES)}

print("Content pool sizes:", {c: len(v) for c, v in content_pools.items()}, flush=True)
print("Style pool sizes:  ", {c: len(v) for c, v in style_pools.items()}, flush=True)

DIRECTIONS = [(sc, tc) for sc in CLASSES for tc in CLASSES if sc != tc]
assert len(DIRECTIONS) == 12
TOTAL_TRIALS = len(DIRECTIONS) * CONTENT_PER_CLASS * STYLE_PER_CLASS
print(f"Directions: {len(DIRECTIONS)} | Total trials: {TOTAL_TRIALS}", flush=True)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def trial_id(shape_cls, texture_cls, content_path, style_path):
    return f"{shape_cls}|{texture_cls}|{os.path.basename(content_path)}|{os.path.basename(style_path)}"


done_ids = set()
if os.path.exists(RESULTS_JSONL):
    with open(RESULTS_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done_ids.add(rec["trial_id"])
            except json.JSONDecodeError:
                continue
print(f"Already completed trials found: {len(done_ids)}", flush=True)

results_file = open(RESULTS_JSONL, "a", buffering=1)


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

style_loader = T.Compose([T.Resize((STYLE_IMG_SIZE, STYLE_IMG_SIZE)), T.ToTensor()])
cls_resize = T.Resize((CLS_IMG_SIZE, CLS_IMG_SIZE))


def load_for_style(path):
    img = Image.open(path).convert("RGB")
    return style_loader(img).unsqueeze(0).to(DEVICE)


def normalize_vgg(x):
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def to_pil(tensor):
    img = tensor.detach().cpu().clamp(0, 1).squeeze(0)
    return T.ToPILImage()(img)


# ---------------------------------------------------------------------------
# VGG19 feature extractor (Gatys)
# ---------------------------------------------------------------------------
print("Loading VGG19...", flush=True)
vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()
for p in vgg.parameters():
    p.requires_grad_(False)

CONTENT_LAYERS = {"21": "relu4_1"}
STYLE_LAYERS = {
    "0": "relu1_1", "5": "relu2_1", "10": "relu3_1", "19": "relu4_1", "28": "relu5_1",
}
ALL_LAYERS = {**CONTENT_LAYERS, **STYLE_LAYERS}


def extract_features(x):
    feats = {}
    out = normalize_vgg(x)
    for name, layer in vgg._modules.items():
        out = layer(out)
        if name in ALL_LAYERS:
            feats[ALL_LAYERS[name]] = out
    return feats


def gram_matrix(feat):
    b, c, h, w = feat.shape
    f = feat.view(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / (c * h * w)


def gatys_style_transfer(content_img, style_img, steps=GATYS_STEPS,
                          content_weight=CONTENT_WEIGHT, style_weight=STYLE_WEIGHT):
    content_feats = extract_features(content_img)
    style_feats = extract_features(style_img)
    style_grams = {k: gram_matrix(v) for k, v in style_feats.items() if k in STYLE_LAYERS.values()}

    target = content_img.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([target], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        feats = extract_features(target.clamp(0, 1))
        c_loss = F.mse_loss(feats["relu4_1"], content_feats["relu4_1"])
        s_loss = 0.0
        for name in style_grams:
            s_loss = s_loss + F.mse_loss(gram_matrix(feats[name]), style_grams[name])
        loss = content_weight * c_loss + style_weight * s_loss
        loss.backward()
        return loss

    optimizer.step(closure)
    return target.clamp(0, 1)


# ---------------------------------------------------------------------------
# Frozen ResNet-50 classifier
# ---------------------------------------------------------------------------
print("Loading frozen ResNet-50 checkpoint...", flush=True)
clf = models.resnet50(weights=None)
clf.fc = nn.Linear(clf.fc.in_features, len(CLASSES))
clf.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
clf = clf.to(DEVICE).eval()
for p in clf.parameters():
    p.requires_grad_(False)


def classify(stylized_01):
    # stylized_01: (1,3,H,W) in [0,1], style-loader resolution -> resize to classifier res
    x = cls_resize(stylized_01)
    x = normalize_vgg(x)  # same ImageNet mean/std
    with torch.no_grad():
        logits = clf(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    pred_idx = int(torch.tensor(probs).argmax().item())
    return pred_idx, probs


# ---------------------------------------------------------------------------
# Cache loaded style images (only 12 unique per direction group, small)
# ---------------------------------------------------------------------------
style_cache = {}


def get_style_tensor(path):
    if path not in style_cache:
        style_cache[path] = load_for_style(path)
    return style_cache[path]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
n_done_this_run = 0
t_start = time.time()

for shape_cls, texture_cls in DIRECTIONS:
    saved_example = False
    for content_path in content_pools[shape_cls]:
        content_img = load_for_style(content_path)
        for style_path in style_pools[texture_cls]:
            tid = trial_id(shape_cls, texture_cls, content_path, style_path)
            if tid in done_ids:
                continue

            style_img = get_style_tensor(style_path)
            t0 = time.time()
            stylized = gatys_style_transfer(content_img, style_img)
            gen_time = time.time() - t0

            pred_idx, probs = classify(stylized)
            pred_cls = CLASSES[pred_idx]
            shape_idx = CLASSES.index(shape_cls)
            texture_idx = CLASSES.index(texture_cls)

            record = {
                "trial_id": tid,
                "shape_class": shape_cls,
                "texture_class": texture_cls,
                "content_path": content_path,
                "style_path": style_path,
                "pred_class": pred_cls,
                "pred_is_shape": pred_cls == shape_cls,
                "pred_is_texture": pred_cls == texture_cls,
                "pred_is_other": pred_cls not in (shape_cls, texture_cls),
                "probs": probs,
                "confidence": max(probs),
                "gen_time_sec": gen_time,
            }
            results_file.write(json.dumps(record) + "\n")
            done_ids.add(tid)
            n_done_this_run += 1

            if not saved_example:
                ex_path = os.path.join(EXAMPLES_DIR, f"{shape_cls}_shape__{texture_cls}_texture.png")
                to_pil(stylized).save(ex_path)
                saved_example = True

            if n_done_this_run % 20 == 0:
                elapsed = time.time() - t_start
                rate = elapsed / n_done_this_run
                remaining = TOTAL_TRIALS - len(done_ids)
                eta_h = rate * remaining / 3600
                print(f"[{len(done_ids)}/{TOTAL_TRIALS}] dir={shape_cls}->{texture_cls} "
                      f"last_gen={gen_time:.1f}s | avg={rate:.1f}s/trial | ETA remaining={eta_h:.2f}h",
                      flush=True)

results_file.close()
print("ALL_TRIALS_DONE", flush=True)
print(f"Total trials in file: {len(done_ids)}", flush=True)
