"""Write the submission tissue mask, remapping training indices to challenge values."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..domain import TISSUE_CLASS_NAMES, TISSUE_SUBMISSION_VALUE, TissueClass

__all__ = ["write_tissue_tiff"]

_SMIN_SAMPLE_VALUE = 340
_SMAX_SAMPLE_VALUE = 341


def write_tissue_tiff(mask: np.ndarray, path: str | Path) -> None:
    """Write a ``(H, W)`` training-index mask as the submission TIFF.

    Training indices and submission values differ (see
    :data:`prometheus.domain.TISSUE_SUBMISSION_VALUE`), so the remap is explicit rather
    than implied by channel order. ``SMinSampleValue``/``SMaxSampleValue`` are written
    because the challenge validator requires them.
    """
    source = np.asarray(mask)
    output = np.zeros(source.shape, dtype=np.uint8)
    for train_index, class_name in enumerate(TISSUE_CLASS_NAMES):
        output[source == train_index] = TISSUE_SUBMISSION_VALUE[TissueClass(class_name)]

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    foreground = output[output > 0]
    minimum = int(foreground.min()) if foreground.size else 0
    maximum = int(foreground.max()) if foreground.size else 0

    import tifffile  # noqa: PLC0415 - keeps `import prometheus` free of the TIFF stack

    tifffile.imwrite(
        destination,
        output,
        photometric="minisblack",
        resolution=(1.0, 1.0),
        resolutionunit="NONE",
        metadata=None,
        extratags=[
            (_SMIN_SAMPLE_VALUE, "H", 1, minimum, False),
            (_SMAX_SAMPLE_VALUE, "H", 1, maximum, False),
        ],
    )
