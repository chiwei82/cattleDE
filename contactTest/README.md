# contactTest — weakly-supervised contact localisation

An experiment that reuses the existing two-stage pipeline's pair crops and binary
interaction labels to predict **where on the image contact happens**, without any
contact annotation and without 3D.

Nothing here modifies data, checkpoints, or config outside this folder. All
inputs are read-only; all outputs land in `contactTest/log/`.

---

## 1. How this attaches to the existing pipeline

```
stage 1   YOLO OBB  ──▶ cow boxes                       (checkpoints/yolo_pseudo.pt)
stage 2   prep/interaction_prep.py
              sample 1 fps → pair up boxes with 0.1 < IoU < 0.8 → drop nested
              → save the UNION crop, un-resized
              → data/interaction/{split}/crops/{video}/frame_XXXXXXXX_pair_XX.jpg
          Label Studio → data/annotated/annotated_interaction_test.csv
          train/interaction_with_image.py → binary interaction classifier

contactTest   same CSV, same crops, same rows
              → binary label pooled from a spatial heatmap instead of a CLS token
              → heatmap is the deliverable
```

The stage-2 classifier answers *whether* a pair interacts. This answers *where*,
from exactly the same supervision.

---

## 2. Why it can work with only a binary label

The label is per **pair**, not per image, which is much stronger than it sounds:
each crop is its own bag, so there is no ambiguity about which two animals the
label refers to. That reduces the problem to single-level multiple-instance
learning, and the bag is further restricted to the region where the two cows
could actually touch.

The one structural commitment that makes it work:

```
patch tokens → dense logits z → mask to R → masked LSE over R → BCE(pair label)
```

There is **no CLS-token classifier and no global pooling branch**. If one
existed, the network could satisfy the label from herd density or background and
leave the heatmap arbitrary. Because the pooled score is a soft maximum over R,
the only way to raise it is to raise the response somewhere the two cows meet —
so the heatmap is the mechanism, not a post-hoc explanation.

Two properties of the data do the rest of the work:

- **Negatives supervise densely.** Pushing the pooled logit down pushes *every*
  pixel in R down, since LSE upper-bounds the max. No extra loss term needed.
- **The negatives are hard.** The `IoU > 0.1` pair filter means every negative is
  two cows overlapping but *not* interacting. That is precisely what teaches the
  model that proximity is not contact. Random distant pairs would have taught it
  nothing.

`R` is the intersection of the two dilated cow supports, minus letterbox padding.
Supports come from the detector boxes by default; drop SAM masks in and set
`data.mask_dir` to tighten it (see §6).

---

## 3. Layout

```
contactTest/
├── config.yaml                      self-contained; never reads global_config.yaml
├── src/
│   ├── data.py                      CSV → records → letterbox → region R
│   ├── model.py                     ViT backbone + dense head + masked LSE pooling
│   ├── losses.py                    MIL BCE + area-normalised sparsity + TV
│   └── utils.py                     config + AUC (torch-free, so the gate runs anywhere)
├── diagnostics/
│   ├── geometry_baseline.py         A — is the label solvable from boxes alone?
│   ├── background_leak.py           B — is it solvable from background alone?
│   └── split_leakage.py             C — near-duplicate overlap across splits
├── train_contact.py
├── infer_contact.py                 overlays + the annotation-free deletion test
└── log/                             all outputs (created on first run)
```

Run everything from the repository root:

```bash
python -m contactTest.diagnostics.geometry_baseline
python -m contactTest.diagnostics.split_leakage
python -m contactTest.diagnostics.background_leak
python -m contactTest.train_contact
python -m contactTest.infer_contact --split test --positives-only --limit 40
python -m contactTest.infer_contact --split test --deletion-test
```

Diagnostics A and C need only numpy/opencv/sklearn and run on a laptop. B and the
training need torch.

---

## 4. Data facts this code was built against

Measured from `data/annotated/annotated_interaction_test.csv` (10 453 rows, of
which 4 224 carry a usable label):

| split | rows | negative | positive |
|---|---|---|---|
| train | 2 064 | 1 824 | 240 |
| val   | 778   | 681   | 97 |
| test  | 1 382 | 1 335 | 47 |

Handled explicitly in `src/data.py`:

- The crop is the merged box cut from the frame **un-resized**, and
  `prep.safe_crop_bgr` clips it at the frame border. In ~12 % of rows the crop is
  therefore smaller than `merged_bbox_xyxy`, so relative boxes are clamped to the
  real crop size rather than trusted.
- Crop origin is `(max(0, merged_x1), max(0, merged_y1))`.
- Letterbox padding is excluded from `R`; it carries no evidence.
- The box intersection is never empty, guaranteed by the `IoU > 0.1` filter
  (verified over 600 crops: 0 fallbacks, R occupies 20–82 % of the canvas,
  median 35 %).

---

## 5. Diagnostic results already obtained

Both torch-free gates have been run on the real data.

**A — geometry baseline: AUC 0.405.** Box layout, crop dimensions and IoU carry
essentially no transferable signal about the label. This is the green light: the
task genuinely requires appearance, so a heatmap model has a reason to look at
pixels. (Below 0.5 also indicates the train and test videos differ in geometry
distribution.)

**C — split leakage: clean, and the split is doing real work.**

| split policy | test crops with a >0.95 near-duplicate in train |
|---|---|
| video-disjoint (what the pipeline does) | **0.0 %** |
| random split (counterfactual) | 37.1 % |

`interaction_prep.assign_videos_622` assigns whole videos to a split, so no
video straddles a boundary and there is no near-duplicate leakage. A random split
would have leaked badly — worth one line in the write-up, since it is a
methodological choice the pipeline gets right by construction.

**B — background leak: not yet run** (needs torch).

---

## 6. Known limitations — read before drawing conclusions

**Positives come from only two training videos.** Of the six labelled videos, all
240 training positives come from two (156 + 84), val's 97 from one, test's 47
from two. The model can plausibly latch onto video-specific appearance rather
than contact, and there are too few videos for leave-one-video-out validation.
Labelling positives in more videos is the single highest-value thing to do next.

**`interest` is probably not contact.** `label_v2` splits the 384 positives into
social grooming (279), interest (97), sniffing (4), mount (4). "Interest" denotes
attention towards another animal and need not involve physical touching. It is
**70 % of the test positives** (33 of 47), which means the test split is mostly
measuring something other than contact.

Default behaviour keeps parity with `train/interaction_with_image.py` and treats
all of them as positive. To train a stricter contact model:

```yaml
labels:
  exclude_positive_v2: ["interest"]
```

That drops test positives from 47 to 14. Neither choice is good; the honest fix
is more annotation. Whichever is used, state it explicitly in the write-up — the
counts are printed at startup.

**AUC does not measure localisation.** Nothing in this dataset supervises where
contact is, so pair-classification AUC is a sanity check only. Two ways to say
something real about localisation:

1. **Deletion test** (`--deletion-test`, no annotation needed). Blank the top 20 %
   of the heatmap and re-score; compare against blanking the same area at random
   and at the region's lowest response. If the model's own region causes a much
   larger drop, the heatmap marks the evidence the classifier relies on. Reported
   as `mean_drop_model` vs `mean_drop_random`.
2. **A small hand-annotated set.** 200–500 crops with a single clicked contact
   point supports a pointing-game accuracy — the standard weakly-supervised
   localisation metric. This is the only way to report a localisation number.

**`R` from boxes is coarse.** For two axis-aligned rectangles the dilated
intersection is itself a rectangle, occupying a median 35 % of the canvas. Real
instance masks would shrink the search space considerably. To use them, write
`<mask_dir>/<same relative path as the crop>.npz` containing `mi` and `mj`
(uint8, same H×W as the crop) and set `data.mask_dir`; the dataset picks them up
automatically and falls back to boxes wherever a file is absent.

---

## 7. Tuning notes

| knob | effect |
|---|---|
| `model.tau_start` → `tau_end` | pooling anneals mean-like → max-like; too fast and only one pixel ever gets gradient |
| `loss.lambda_sparsity` | without it the heatmap saturates all of `R`, since the pooling is indifferent to how much is lit |
| `loss.lambda_tv` | one connected blob instead of scattered pixels; raise if predictions look speckled |
| `region.dilate_px` | larger admits more of the shortcut space, smaller risks excluding true contact |
| `model.backbone` | `timm` reuses the repo's ViT and can warm-start from `checkpoints/action.ckpt`; `dinov2` gives sharper spatial features but downloads weights via torch.hub |
| `model.image_size` | 224 → 16×16 patch grid. 448 quadruples heatmap resolution and localises better once the pipeline is proven |
