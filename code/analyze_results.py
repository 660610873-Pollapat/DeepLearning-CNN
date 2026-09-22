"""
Quick locked-metric summary of cue_conflict_trials.jsonl:
  Shape Choice   = N(shape)   / (N(shape)+N(texture)+N(other))
  Texture Choice = N(texture) / (N(shape)+N(texture)+N(other))
  Other          = N(other)   / (N(shape)+N(texture)+N(other))
Reported at trial level and content-image level (aggregate 3 style refs
per content image first, then aggregate across the 40 content images).
"""
import json
import os
from collections import defaultdict

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSONL = os.path.join(OUT_DIR, "cue_conflict_trials.jsonl")

trials = []
with open(RESULTS_JSONL) as f:
    for line in f:
        line = line.strip()
        if line:
            trials.append(json.loads(line))

print(f"Total trials loaded: {len(trials)}")

# ---- Trial-level ----
n_shape = sum(t["pred_is_shape"] for t in trials)
n_texture = sum(t["pred_is_texture"] for t in trials)
n_other = sum(t["pred_is_other"] for t in trials)
n_total = len(trials)

print("\n=== TRIAL LEVEL (n={}) ===".format(n_total))
print(f"Shape Choice:   {n_shape/n_total:.3f} ({n_shape})")
print(f"Texture Choice: {n_texture/n_total:.3f} ({n_texture})")
print(f"Other:          {n_other/n_total:.3f} ({n_other})")

avg_conf = sum(t["confidence"] for t in trials) / n_total
print(f"Avg confidence: {avg_conf:.3f}")

# ---- Per-direction breakdown ----
by_dir = defaultdict(list)
for t in trials:
    by_dir[(t["shape_class"], t["texture_class"])].append(t)

print("\n=== PER DIRECTION (shape -> texture) ===")
print(f"{'Direction':<20}{'Shape':>8}{'Texture':>10}{'Other':>8}{'n':>6}")
for (sc, tc), ts in sorted(by_dir.items()):
    n = len(ts)
    s = sum(x["pred_is_shape"] for x in ts) / n
    tx = sum(x["pred_is_texture"] for x in ts) / n
    o = sum(x["pred_is_other"] for x in ts) / n
    print(f"{sc+'->'+tc:<20}{s:>8.3f}{tx:>10.3f}{o:>8.3f}{n:>6}")

# ---- Content-image level (aggregate 3 style refs per content first) ----
by_content = defaultdict(list)
for t in trials:
    key = (t["shape_class"], t["texture_class"], t["content_path"])
    by_content[key].append(t)

content_shape_rates = []
content_texture_rates = []
content_other_rates = []
for key, ts in by_content.items():
    n = len(ts)
    content_shape_rates.append(sum(x["pred_is_shape"] for x in ts) / n)
    content_texture_rates.append(sum(x["pred_is_texture"] for x in ts) / n)
    content_other_rates.append(sum(x["pred_is_other"] for x in ts) / n)

n_content = len(content_shape_rates)
print(f"\n=== CONTENT-IMAGE LEVEL (n={n_content} content images, "
      f"each aggregated over its 3 style refs) ===")
print(f"Mean Shape Choice:   {sum(content_shape_rates)/n_content:.3f}")
print(f"Mean Texture Choice: {sum(content_texture_rates)/n_content:.3f}")
print(f"Mean Other:          {sum(content_other_rates)/n_content:.3f}")

# Save summary
summary = {
    "n_trials": n_total,
    "trial_level": {"shape": n_shape/n_total, "texture": n_texture/n_total, "other": n_other/n_total},
    "avg_confidence": avg_conf,
    "content_level": {
        "shape": sum(content_shape_rates)/n_content,
        "texture": sum(content_texture_rates)/n_content,
        "other": sum(content_other_rates)/n_content,
    },
    "per_direction": {
        f"{sc}->{tc}": {
            "shape": sum(x["pred_is_shape"] for x in ts)/len(ts),
            "texture": sum(x["pred_is_texture"] for x in ts)/len(ts),
            "other": sum(x["pred_is_other"] for x in ts)/len(ts),
            "n": len(ts),
        } for (sc, tc), ts in by_dir.items()
    },
}
with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: results_summary.json")
