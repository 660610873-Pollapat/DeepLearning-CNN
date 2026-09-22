# CNN: Global Shape vs. Local Texture (Cue-Conflict Study)

Code and results for the report `report/shape_vs_texture_report.pdf`.

## 1. Research question

When global shape and local texture provide conflicting class cues, which cue does a
trained ResNet-50 rely on more during classification?

**Prediction (written before the main cue-conflict experiment, not revised afterward):**
the model would rely more strongly on local texture than global shape. The experiment
found the opposite (see Section 3).

## 2. Dataset / model / method

- **Dataset:** [Animal Computer Vision Clean Dataset and Code](https://www.kaggle.com/datasets/emirhanai/animal-computer-vision-clean-dataset-code-cnnai)
  (Emirhan Bulut, Kaggle). 4,000 images, 1,000 each of Buffalo, Elephant, Rhino, Zebra.
  License: CC BY 4.0 — attribution to Emirhan Bulut. **Not included in this repository**
  (download separately from Kaggle; see Section 3 for the expected folder layout).
- **Split:** 70/15/15 per class, fixed seed 42 — Train 2,800 (700/class), Validation 600
  (150/class), Test 600 (150/class). Train also serves as the style-reference pool and
  Test as the content pool for the cue-conflict experiment.
- **Model:** ImageNet-pretrained ResNet-50, fine-tuned with only `layer4` and the final
  classifier head unfrozen. Best checkpoint selected on the validation set only; the test
  set is touched exactly once, after the checkpoint is frozen.
- **Cue-conflict stimuli:** Gatys-style neural style transfer (VGG19 features, Gram-matrix
  style loss, LBFGS optimizer, 150 steps) combines the shape of one class's test image
  with the texture of a different class's train image.
- **Trials:** all 12 ordered class pairs × 40 content images (test split) × 3 style images
  (train split) = **1,440 trials total**. These are not 1,440 independent samples — each
  content image contributes 3 correlated trials; both trial-level and content-image-level
  results are reported (see report, Figure 3 caption).
- **Supporting control:** the same 160 test-content images converted to an edge-based
  representation (`PIL.ImageFilter.FIND_EDGES` + thresholding) and classified with the
  same frozen checkpoint. This does **not** cleanly isolate shape — internal-texture edges
  (e.g. zebra stripes) remain visible — so it is reported as supporting evidence only,
  not a second independent measurement of "shape alone."

Full methodology, results, interpretation, limitations, and AI usage disclosure are in
`report/shape_vs_texture_report.pdf`.

## 3. How to reproduce the reported numbers

### 3a. Fastest path: verify numbers without rerunning anything

Every number in the report is already saved in this repo — no GPU/CPU time required:

- `code/baseline_results.json` — baseline accuracy/F1/confusion matrix (Section: Baseline)
- `code/cue_conflict_trials.jsonl` — all 1,440 raw cue-conflict trial records
- `code/results_summary.json` — trial-level, content-level, and per-direction aggregates
  (produced by `analyze_results.py` from the file above)
- `code/silhouette_trials.jsonl` / `code/silhouette_summary.json` — the 160 edge-based
  control trials and their summary

Open these directly, or re-run just the aggregation step (seconds, no model needed):

```bash
pip install -r requirements.txt
cd code
python analyze_results.py        # reads cue_conflict_trials.jsonl -> results_summary.json
```

### 3b. Full reproduction from scratch (re-trains and re-generates everything)

1. Download the dataset from Kaggle (link above) and arrange it as:
   ```
   <your-path>/animal_computer_vision/Dataset/Buffalo/*.jpg    (1,000 files)
   <your-path>/animal_computer_vision/Dataset/Elephant/*.jpg   (1,000 files)
   <your-path>/animal_computer_vision/Dataset/Rhino/*.jpg      (1,000 files)
   <your-path>/animal_computer_vision/Dataset/Zebra/*.jpg      (1,000 files)
   ```
2. **Set the `SHAPE_VS_TEXTURE_DATASET_ROOT` environment variable** to point at
   `<your-path>/animal_computer_vision/Dataset`. `baseline_resnet50.py`,
   `main_experiment.py`, and `silhouette_control.py` all read this variable and
   fall back to the original machine's path only if it is unset.
3. Install dependencies and run in order from inside `code/`:
   ```bash
   pip install -r requirements.txt
   cd code
   export SHAPE_VS_TEXTURE_DATASET_ROOT="<your-path>/animal_computer_vision/Dataset"  # PowerShell: $env:SHAPE_VS_TEXTURE_DATASET_ROOT = "<your-path>\animal_computer_vision\Dataset"

   python baseline_resnet50.py      # trains baseline, writes split_indices.json,
                                     # resnet50_baseline_best.pt, baseline_results.json
                                     # (checkpoint is NOT included in this repo — see note below)

   python main_experiment.py        # 1,440 cue-conflict trials (resumable; took ~14 hours
                                     # on CPU on the original machine — a GPU will be much
                                     # faster). Writes cue_conflict_trials.jsonl and
                                     # examples/*_shape__*_texture.png

   python analyze_results.py        # aggregates cue_conflict_trials.jsonl -> results_summary.json

   python silhouette_control.py     # edge-based control (160 images, a few minutes),
                                     # writes silhouette_trials.jsonl, silhouette_summary.json
   ```
   All four scripts read/write their output files in their own directory (`code/`), and
   all randomization (split, content/style sampling, model init) uses the fixed seed 42
   defined at the top of each script, so re-running from the same dataset should reproduce
   the same numbers reported in the PDF.

**Note on the checkpoint:** `resnet50_baseline_best.pt` (~94 MB) is not included in this
submission to keep the file size manageable. Re-running `baseline_resnet50.py` with seed 42
on the same dataset regenerates an equivalent checkpoint; the baseline metrics achieved were
test accuracy 98.83%, macro-F1 98.83% (see `code/baseline_results.json`).

## 4. Where the original result files are

| File | Contents |
|---|---|
| `code/split_indices.json` | The exact train/val/test file split used everywhere |
| `code/baseline_results.json` | Baseline accuracy, macro-F1, per-class report, confusion matrix |
| `code/cue_conflict_trials.jsonl` | All 1,440 raw cue-conflict trial records |
| `code/results_summary.json` | Trial-level / content-level / per-direction shape-texture-other rates |
| `code/silhouette_trials.jsonl` | All 160 edge-based control trial records |
| `code/silhouette_summary.json` | Edge-based control accuracy, overall and per class |
| `examples/*_shape__*_texture.png` | One example stylized image per cue-conflict direction (12) |
| `examples/*_silhouette.png` | One example edge-based representation per class (4) |
| `report/shape_vs_texture_report.pdf` | Full report: question, method, results, interpretation, limitations, AI disclosure |

## 5. Important limitations

See the "Limitations" section of the report for the full list. In short: results describe
this specific ResNet-50 + dataset + training procedure + Gatys style-transfer method under
this cue-conflict setup only, and should not be generalized to CNNs broadly; the
style-transfer "texture" condition is a texture-defined/style-defined cue, not perfectly
isolated texture; the 1,440 trials are not fully independent samples; and the edge-based
control does not cleanly isolate shape (internal-texture edges remain visible) and is
out-of-distribution relative to the training data, so it is treated as supporting evidence
only.

## AI usage disclosure

See the "AI Usage Disclosure" section at the end of `report/shape_vs_texture_report.pdf`
for the full, itemized disclosure of where AI assistance was used in this project (and
where it needed correction).
