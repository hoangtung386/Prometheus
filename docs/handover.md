# Handover

Read this, then [`architecture.md`](architecture.md), then
[`phan-tich-tissue-va-ke-hoach.md`](phan-tich-tissue-va-ke-hoach.md).

## What this project is

A PUMA challenge entry. Two tasks, ranked independently by the challenge and combined as a
mean of ranks:

- **tissue segmentation** — 5 scored classes, official metric = micro Dice;
- **nuclei detection** — 10 classes (Track 2), official metric = macro F1 with 15-pixel
  centroid matching.

## Where it actually stands

| Task | Current | Winner | Verdict |
|---|---:|---:|---|
| Tissue micro Dice | 52.90 | 78.23 | below the challenge baseline (55.48); **this is where the work is** |
| Nuclei macro F1 (10-class) | 21.72 | 26.56 (2nd place) | already competitive; **do not invest here** |

The tissue number decomposes into three classes at winning level (tumor 89.19, stroma 81.00,
epidermis 90.52) and two that were never learned (necrosis 0.00, blood_vessel 3.78). Fixing
only those two would give 77.69 without touching the rest.

That framing is the single most useful thing to carry into the work. The full evidence —
per-class tables, the training log showing validation Dice still rising at the final epoch,
and the leaderboard comparison — is in
[`phan-tich-tissue-va-ke-hoach.md`](phan-tich-tissue-va-ke-hoach.md).

## First hour

```bash
uv sync --extra dev
uv run pytest -q                                  # 128 tests, ~4s, CPU only
uv run prometheus audit --data-root /path/to/puma  # do this before any experiment
```

The audit answers three questions that decide what to work on, and that the metrics alone
cannot distinguish from a modelling limitation:

1. `rasterization.interior_ring_count` — if non-zero, the annotations use interior rings,
   and any rasterizer that drops them produces wrong masks.
2. `rasterization.per_class.<name>.pixel_delta_percent` — how much area each class gains
   over a naive file-order rasterization. A large delta on `necrosis` or `blood_vessel`
   means the old training masks barely contained the class, and no amount of training would
   have recovered it.
3. `resolution` — whether the images are the native 1024x1024 at 40x the challenge test set
   uses, and whether the GeoJSON coordinates agree with them. `coordinates_exceed_image:
   true` means every mask is misaligned; stop and fix that first.

`rasterization.per_class.<name>.images_containing` sizes the rare-class problem: it is the
input to both the oversampling ratio and fold stratification.

## What changed in this refactor

Behaviour-preserving:

- Removed dead code: the YOLO adapter, `DecoderBlock`, five unused binary losses, four
  unused geometry helpers, three unused rasterization helpers, four unused label maps, and a
  dead metric-formatting path. Also stale `__pycache__` trees for packages deleted long ago
  (`legacy/`, `nn/`, `training/`) and two empty agent scaffolding directories.
- Consolidated the taxonomy into `domain/labels.py`. It previously existed twice, which made
  `io` depend on `data` and left two class orderings free to drift. Nucleus indices are now
  zero-based directly, removing an off-by-one dance at the call site.
- Flattened single-module packages, merged `blocks/` and `utils/` into `layers.py`, and
  renamed the two modules both called `evaluator.py` (`metrics/segmentation.py`,
  `engine/validation.py`).
- Split `trainer.py` into `trainer.py` / `ema.py` / `schedule.py`, and `cli/main.py` into
  parser and command handlers.
- `nuclei_detection_metrics` returns a typed dataclass instead of `dict[str, object]`.

Behaviour-changing — these move metrics, review them as such:

| Change | Effect | Rationale |
|---|---|---|
| Tissue rasterization keeps interior rings and paints by class priority | training masks change | doc section 3.5 |
| `SegmentationEvaluator` reports official micro Dice, per class, every epoch | validation numbers drop to the honest value | section 3.4 |
| Class weights: square-root inverse frequency, normalised over scored classes, background pinned to 0.1 | rebalances CE and Dice | section 3.3 |
| `MultiClassDiceLoss(ignore_absent=False)` by default | hallucinated rare classes are now penalised | section 3.3 |
| `PrometheusPredictor` applies 8-view dihedral TTA by default | submission accuracy up, inference 8x slower | section 3.11 |
| `decode_nuclei` rejects a multi-channel center map | a version-1 checkpoint now fails loudly | contract 5 |

Nothing in the roadmap's steps 4, 5 or 7 (patch training, rare-class oversampling,
albumentations) has been implemented. Those are the next work items and they depend on the
audit output.

## What to do next

Ranked by points per unit of effort. Full reasoning and expected gains per step are in
section 6 of the analysis document.

1. Run the audit. Act on what it says before anything else.
2. **Patch-based training** (section 3.1). 11 optimizer steps per epoch is the binding
   constraint; validation Dice had not plateaued at the final epoch of any fold. Random
   512x512 crops with 8-16 patches per image per epoch decouples step count from dataset
   size and fits on a 10GB GPU.
3. **Rare-class oversampling** (section 3.2). Force a fraction of every batch to be centred
   on a uniformly chosen foreground class. This is nnU-Net's `oversample_foreground_percent`
   and it is what makes rare classes learnable at all.
4. **Stronger augmentation** (section 3.8): random scale, free rotation, HSV/RGB shift,
   elastic. Prefer `albumentations` over extending the hand-written pipeline.
5. **5-fold ensemble + logit-bias tuning on out-of-fold predictions** (section 3.11).
6. Only then consider the architecture work: a pathology foundation model encoder and a
   dedicated full-resolution blood-vessel model (sections 3.9 and 5).

**Gate:** if out-of-fold micro Dice is below 72 after step 5, do not start step 6. Something
in the data pipeline is still wrong and a bigger encoder will not find it.

## Things that will surprise you

- `average_dice` in older `metrics.json` files is meaningless: it returns 1.0 for a class
  absent from both prediction and ground truth, so it reported 75.83 where the official
  metric says 52.90. Only `micro_dice_tissue.average_micro_dice` is comparable to the
  leaderboard.
- The preliminary test set is 10 regions. `epidermis` appears in exactly one of them, so its
  90.52 is one image's score. `necrosis` appears in two. Build the local 5-fold out-of-fold
  harness before trusting any comparison between experiments.
- Training and submission use *different* checkpoints from *different* folds
  (`best_primary.ckpt` for nuclei, `best_tissue.ckpt` for tissue), loaded sequentially. This
  is why the shared encoder buys nothing at inference time.
- `PUMA-track2-submit` (sibling repository) vendors `src/prometheus` at a pinned commit. It
  needs re-vendoring after this refactor.
- The GPU on the local workstation is a 10GB RTX 3080; the recorded 5-fold run used a
  ~80GB Colab GPU at 61GB peak. Patch-based training brings the work back onto local
  hardware, which shortens the experiment loop far more than the GPU difference costs.

## Open questions for the project owner

1. Are the images on disk native 1024x1024, or a 512x512 downsample? The audit answers this;
   if they are downsampled, the training data must be regenerated.
2. Is there Hugging Face gated access to `paige-ai/Virchow2`? Required for the winner's
   approach. Fallbacks: `MahmoodLab/UNI2-h`, `bioptimus/H-optimus-1`, `owkin/phikon-v2`.
3. Are the 5120x5120 context images available locally?
4. Which challenge phase is still open? This sets whether the 3-4 week architecture track is
   worth starting.
