"""Write class-index tissue masks using explicit challenge value remapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.puma import TISSUE_CLASSES
from ..domain import TISSUE_SUBMISSION_VALUE, TissueClass


def write_tissue_tiff(mask: np.ndarray, path: str | Path) -> None:
    import tifffile

    source = np.asarray(mask)
    output = np.zeros(source.shape, dtype=np.uint8)
    for train_index, class_name in enumerate(TISSUE_CLASSES):
        output[source == train_index] = TISSUE_SUBMISSION_VALUE[TissueClass(class_name)]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    foreground = output[output > 0]
    minimum = int(foreground.min()) if foreground.size else 0
    maximum = int(foreground.max()) if foreground.size else 0
    tifffile.imwrite(
        destination,
        output,
        photometric="minisblack",
        resolution=(1.0, 1.0),
        resolutionunit="NONE",
        metadata=None,
        extratags=[
            (340, "H", 1, minimum, False),  # SMinSampleValue
            (341, "H", 1, maximum, False),  # SMaxSampleValue
        ],
    )
