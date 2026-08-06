# Contributing

## Setup

```bash
uv sync --extra dev
uv run pre-commit install
```

## Quality gates

Run all four before committing. There is no CI: this is a Colab training project, so the
gate is local.

```bash
uv run ruff check src tests      # lint
uv run ruff format --check src tests
uv run mypy                      # types, src/prometheus
uv run pytest -q                 # tests, CPU only, no downloads
```

`ruff` and `mypy` configuration lives in `pyproject.toml`. There is no separate
`setup.cfg`, `.flake8` or `mypy.ini`.

## Conventions

**Style.** PEP 8 via `ruff format`, 120 columns. `ruff` enforces pyflakes, pycodestyle,
import sorting, naming, pyupgrade, bugbear, comprehensions, pathlib, a pylint subset and
pytest style. Four rules are ignored, each with a reason recorded in `pyproject.toml`; add
to that list only with a comparable reason.

**Naming.** Spell things out: `class_probabilities`, not `cls_p`. The exception is the
PyTorch ecosystem convention `from torch.nn import functional as F`.

**Types.** Annotate public functions. `from __future__ import annotations` at the top of
every module. `mypy` runs with `check_untyped_defs`, so untyped helpers are still checked.

**`__all__`.** Every module declares it. Package `__init__.py` files re-export the public
surface and nothing else.

**Docstrings.** Google style. Say *why*, not *what* — the code already says what. A
docstring that restates the signature is worse than none. Comments earn their place by
recording a decision, a constraint or a measurement, and the empirical ones cite
`docs/phan-tich-tissue-va-ke-hoach.md` so the number can be re-derived.

**Imports.** Relative within the package (`from ..domain import ...`). This keeps
`src/prometheus` relocatable, which matters because the submission runtime vendors it
verbatim.

**Layering.** Respect the table in [`docs/architecture.md`](docs/architecture.md).
`domain`, `layers` and `config` are leaves and must not import from the rest of the package.

## Tests

Tests mirror the package layout: `src/prometheus/metrics/segmentation.py` is tested by
`tests/unit/metrics/test_segmentation.py`.

- **One behaviour per test**, named for the behaviour, not the function.
  `test_absent_class_scores_zero_not_one` beats `test_micro_dice_2`.
- **Assert the property, not an incidental number.** Where a magic number is unavoidable,
  name it or derive it in the test.
- **A test for a fixed bug states what the bug was**, in a comment, with the mechanism.
  `tests/unit/data/test_rasterize.py`, `tests/unit/metrics/test_segmentation.py` and
  `tests/unit/engine/test_class_weight_cache.py` are the model to copy.
- **Never download anything.** The suite is CPU-only and offline; `build_model(pretrained=True)`
  is the notebook's job.
- Markers: `integration`, `slow`, `gpu`. `--strict-markers` is on, so an unregistered marker
  is an error.

### The notebook is covered too

`notebooks/train.ipynb` is the supported training workstation, so
`tests/contracts/test_notebook_contract.py` checks it statically: every `prometheus` symbol
it imports must exist, and no module-level name may be used before an earlier cell binds it.
Both checks exist because both failures have happened, and on Colab they surface minutes
into a GPU session. Run the suite after editing a cell.

## Changes that need extra care

**Class order.** `TISSUE_TRAIN_ORDER` and `NUCLEUS_TRAIN_ORDER` in `domain/labels.py` are a
checkpoint contract. Changing them invalidates every trained model.
`PrometheusNet.architecture_version` must be bumped, and `assert_checkpoint_compatible` will
then reject old weights — which is the intended behaviour, not an obstacle to route around.

**`PrometheusModelConfig`.** Stored in every checkpoint and compared field-by-field on load.
Adding a field invalidates existing checkpoints.

**Anything that changes the training masks or the class-weight policy.** The class-weight
cache is keyed on a signature that includes `class_weight_power` and the tissue paint
priority; extend `PrometheusTrainer._class_weight_signature` when you add another input, or
a run directory that outlives the change will silently reuse stale weights.

**The submission runtime.** `PUMA-track2-submit` vendors `src/prometheus` at a pinned
commit. Module moves are fine, but note them so that repository can be re-vendored
deliberately rather than discovered broken.

## Commits

One logical change per commit. Separate a behaviour-preserving refactor from a
behaviour-changing fix, even when they touch the same file — a mixed diff cannot be
reviewed. Say in the message what changed and why; if a metric moves, put the before and
after numbers in the body.
