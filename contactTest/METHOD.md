# How the contact region is computed, and what depth adds

Reference for the green region in panel 3 of `visualize_sam_confusion.py`, the
four alternative definitions in `sam_contact_region.py`, and the three depth
gates. Written against the code as it stands; the formulas below are the ones
actually evaluated, not a simplification of them.

---

## 1. The problem this is solving

The task is per-pixel: for each frame, which pixels are where two cattle are
touching. The only supervision available at scale is one bit per pair — "do
these two interact" — with no contact location and no 3D. So the region is not
learned. It is **constructed** from geometry, and then measured against 634
hand-clicked points on 81 crops.

Everything below is therefore a hypothesis about where contact is, and the
measurement in section 5 is what decides between them.

---

## 2. The inputs

| | source | what it gives | what it does not |
|---|---|---|---|
| boxes | YOLO, upstream | the two animals' extents; the centre of a box is on an animal | nothing about shape |
| `mi`, `mj` | SAM, `precompute_masks.py` | which pixels belong to which animal — **instance identity** | no depth; misled by coat pattern and by floor |
| `depth` | Depth Anything V2, `precompute_depth.py` | per-pixel range of the **visible** surface | no identity: two touching cattle are one smooth blob |

The last row of that table is the reason depth can never replace SAM here.
Depth loses the ability to tell the two animals apart exactly when they touch,
which is the case of interest. SAM supplies identity, depth qualifies it.

Depth is stored raw, with the robust 2nd/98th percentiles alongside. The
relative checkpoints predict **inverse** depth (larger = nearer) defined only up
to an unknown scale and shift, so no absolute distance survives and every
comparison is made **within one crop**, as a fraction of that crop's spread.
`farness()` normalises the direction once so the metric checkpoints, where
larger means further, do not need separate handling everywhere else.

---

## 3. The green region: four definitions

All four are computed in `contact_readings()`. `di`, `dj` are distance
transforms — for every pixel, the distance to the nearest pixel of that mask.

```
overlap   mi & mj
          Where the two silhouettes literally coincide. Sparse, and in an
          overhead view it means occlusion at least as often as contact.

gap       (di + dj) <= touch_px
          The strip between the two surfaces, thresholded on the true local
          separation rather than on an arbitrary radius. Includes floor visible
          in the gap when the animals are close but not touching.

surface   dilate( boundary(mi) within touch_px of j  OR
                  boundary(mj) within touch_px of i , strip_px )
          Contact happens ON a body surface, so this keeps skin rather than the
          air between. Closest to "which part of the animal is touching".

dilated   dilate(mi, r) & dilate(mj, r)          <- panel 3's green outline
          r has no physical meaning, which is exactly why the other three are
          worth measuring against it.
```

`dilated` at `r = 22` is the operating point everything is currently reported
at. It was chosen by sweeping r on the annotated crops — one scalar on a
monotonic trade-off curve, so there is little room for it to be overfitted, but
it was chosen on the same data it is reported on and that should be stated.

**What none of them can do.** All four are statements about two silhouettes in
the image plane. Two failure modes follow directly and cannot be tuned away by
changing `r`, because the information needed is not in the silhouettes:

- **floor** — two animals standing near each other have their masks dilated into
  the ground between them, so the band runs along the pen surface; feet meeting
  the floor are caught the same way
- **occlusion** — one animal passing behind another overlaps in projection while
  being a metre away

---

## 4. What depth adds

Depth enters at two independent points. They can be combined.

### 4a. Before SAM — as prompts (`precompute_masks.py --prompt-source`)

SAM accepts **negative points** natively, which is the mechanism used. For each
box: a positive point at the centre, positive points at pixels near the centre's
own depth, negative points on anything further than that (ground), and negative
points inside the other animal's box (which stops one mask swallowing both, and
needs no depth).

Two facts anchor this, and **only** these two are assumed:

- **the box centre is on an animal**, because the detector put the box there
- **anything further than the animals is ground**, because the ceiling camera is
  angled downward

Nothing is claimed about what lies *nearer* than the cattle. On this camera the
nearest surfaces are often railings, feed barriers and pipework, not animals —
so a rule of the form "the near mode of the depth histogram is the cattle" is
wrong here, and an earlier version that used one has been removed.

`depth_image` is the other variant: run SAM on the colourised depth map instead
of the photograph. The motivation is that the observed failure is a texture
failure — on a Holstein, the edge between a black patch and a white patch is
stronger than the edge between the animal and the floor, so a mask comes back as
one patch of hide. A depth map has no coat pattern at all. The cost is that it
also has no eyes, ears or legs, so SAM has less to recognise as an object and
may merge two animals standing at the same range.

### 4b. After the band — as gates (`depth_stats()`)

A gate is one line: `region = region & (statistic <= tolerance)`. It can only
ever **remove** pixels. Three statistics, each a full-resolution map in units of
the crop's depth spread:

```
body   min over animals of |depth - ref_k| / spread
       ref_k = median depth of a small disc at box k's centre

       Distance from cattle depth. Removes the floor, and feet where they meet
       it. TWO-SIDED on purpose: rejecting only what is further would keep the
       railings this camera sees nearer than a cow's back, and those are not
       contact either. Taking the smaller distance to either animal keeps it
       correct when one cow stands nearer than the other.

       Anchored on the box centres — not on the masks, which would inherit
       whatever the segmentation got wrong, and not on the near mode, which on
       this camera is as likely to be hardware as animal.

step   |grad depth| / (8 * spread)          [Sobel; the 8 undoes its gain]

       Contact is where two surfaces MEET, so depth runs continuously across
       the junction; occlusion is where one passes in front of another, which
       puts a STEP in the depth map. The size of that step is the signal, and
       it exists precisely because the model makes no attempt to smooth
       occlusion away.

       NOTE THE UNITS. This is a gradient — spread per pixel — while the other
       two are plain depth differences. They are not on the same scale, and a
       tolerance that is tight for `body` is wide open for `step`. Depth
       Anything's output is upsampled bicubically, so an occlusion edge is
       smeared over several pixels and a 0.5-spread step reads as roughly
       0.06 per pixel. `step` only starts acting somewhere around 0.01-0.05.

pair   |depth at i's nearest exclusive pixel - the same for j| / spread

       Whether the two animals are at the same range at all. Each side is read
       from the part of that animal the OTHER mask does not cover. This is
       essential rather than fussy: inside the intersection both masks contain
       the pixel, so sampling either one returns the same single value and the
       difference is identically zero — an occluding pair would score as
       perfectly agreeing, and the gate would be blind exactly where it is
       needed. A depth map only ever holds the FRONT surface, so the occluded
       animal's range is not observable inside the intersection and has to be
       carried in from where that animal is visible.

       Unlike `step`, this fires over the whole INTERIOR of an occluding
       overlap, not only at its edge. That is the one thing the other two miss.
```

Depth Anything V2 performs no amodal completion: it emits the range of the
visible surface and has no notion that anything is hidden. All three readings
are plain lookups into that one map — nothing tries to recover an occluded
surface, because nothing can.

---

## 5. Why these things are measured, and not others

`score_contact.py` reports five numbers per configuration. Each exists to catch
a specific way of being fooled.

**hit rate** — share of the 634 clicked points inside the region.
Alone it is maximised by a region covering the whole crop.

**area / share of crop** — median and mean.
Alone it is maximised by a region covering nothing. Neither is ever reported
without the other. (The two tables differ: the reading table quotes the median,
the depth table the mean, because a gate strong enough to empty the band on more
than half the crops drives a median to zero and makes lift infinite.)

**lift** = hit rate / area share.
How much more concentrated contact is inside the region than chance.

**selectivity** = (share of hits kept) / (share of area kept).
**The only column that justifies a gate.** A gate deleting pixels at random
keeps hits in the same proportion as area, giving 1.00. Above 1 means the
discarded pixels were disproportionately not contact. Below 1 means the gate is
cutting away the right pixels — for `body` that would mean genuinely low
contacts are being lost along with the floor, which no depth map can separate,
since a head lowered to another animal's leg is at floor depth by definition.

This column exists because of a specific earlier failure: three plausible
metrics (region size, component count, share inside the band) all rewarded a
region for simply being **bigger**, and a mask-selection rule "improved" all
three by degenerating towards colouring everything. Selectivity is immune to
that, because it is a ratio of what was kept to what was thrown away.

**fires on none** — share of the 18 crops marked "no contact" where the region
is non-empty anyway.
This is the control. It is 100% for the ungated band at every radius tried,
which is structural: pairs were selected by box IoU > 0.1, so the two boxes
always overlap and the dilated intersection is never empty. **This is why the
band answers "where could contact be" and not "is there contact".** A gate that
drives this down while holding selectivity above 1 would have turned the
candidate region into a detector.

**tolerance table (@0/5/10/20 px)** — hit rate as the click is allowed some
slack. A reading whose hit rate climbs steeply from 0 to 10 px was being charged
for human click precision rather than for being in the wrong place. `dilated`
goes 77 → 81 → 85 → 90%, a flat curve, so the ~10% that still miss at 20 px are
genuine mislocalisation, not annotation noise.

**Precision, recall, IoU and mAP are not reported.** Clicked points have no
area, so `|band ∩ GT| / |band|` is undefined in any useful sense; and there is no
confidence ranking and no instances, so an mAP sweep would be dominated by the
arbitrary choice of band width. Clicks answer "is it in the right place"; area
metrics need an area to compare against.

---

## 6. Measured so far

81 crops, 634 clicked points, 18 "no contact" controls, `dilated` at r = 22,
train split only. Area figures here are means, as in the depth table.

```
gate       tol   hit rate     area   of crop   lift   selectivity   fires on none
none         -        77%   15449px     7.9%    10x          1.00            100%
body      0.20        63%   10378px     5.3%    12x          1.21            100%
step      0.20        77%   15369px     7.9%    10x          1.01            100%
pair      0.20        70%   10011px     5.2%    13x          1.40            100%
all       0.20        60%    7377px     3.9%    15x          1.61             94%
```

- `pair` is the best single gate: it gives up 9% of hits to remove 35% of area.
- `body` helps less than `pair`, which is worth noting because the floor was the
  target and occlusion was not — in this band, occlusion is the larger
  contaminant.
- `step` did nothing at this tolerance, for the units reason in section 4b. It
  is untested, not disproved, until it is swept around 0.01-0.05.
- `fires on none` has barely moved. The band is still a candidate region.

**One tolerance is not a result.** These are single-point readings at 0.20; the
sweep that would locate each gate's knee has not been run.

---

## 7. Standing caveats

- **Train split only.** The 99 annotated crops come from **2 videos**, both
  morning colour footage. The dataset has **15**, spanning 08:00 to 22:10, and
  one of them (22:10) is pure greyscale night footage — measured saturation
  exactly 0.0. SAM and Depth Anything have not been checked there at all.
- **`r = 22` was chosen on the data it is reported on.** One scalar on a
  monotonic curve, so the risk is small, but it should be stated.
- **The 634 points are not 634 independent samples.** Seven clicks on one crop
  share two animals and one segmentation. The effective sample size is nearer
  the 81 crops, so a confidence interval should be computed at crop level:
  roughly ±9% rather than the ±3% that treating points as independent implies.
