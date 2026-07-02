"""Lightweight visualization helpers built on public prediction contracts."""

from __future__ import annotations

import numpy as np
import torch

from ..data.puma import NUCLEI_CLASSES, TISSUE_CLASSES
from ..domain import MultitaskBatch


def _display_image(image: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor into a contrast-safe RGB preview."""
    array = image.detach().float().cpu().permute(1, 2, 0).numpy()
    preview = np.zeros_like(array)
    for channel in range(array.shape[-1]):
        values = array[..., channel]
        low, high = np.percentile(values, (1, 99))
        preview[..., channel] = np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)
    return preview


def visualize_multitask_batch(
    batch: MultitaskBatch,
    index: int = 0,
    *,
    show_boxes: bool = True,
) -> None:
    """Visualize the current instance-aware image, tissue mask, and nuclei targets."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not 0 <= index < len(batch.metadata):
        raise IndexError(f"Batch index {index} is outside [0, {len(batch.metadata)})")

    image = _display_image(batch.images[index])
    tissue = batch.tissue.mask[index].detach().cpu().numpy()
    nuclei = batch.nuclei[index]
    centroids = nuclei.centroids.detach().cpu().numpy()
    labels = nuclei.labels.detach().cpu().numpy()
    boxes = nuclei.boxes.detach().cpu().numpy()

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image)
    axes[0].set_title(f"Normalized input: {batch.metadata[index].sample_id}")
    tissue_plot = axes[1].imshow(tissue, cmap="tab10", vmin=0, vmax=len(TISSUE_CLASSES) - 1)
    axes[1].set_title("Tissue target")
    figure.colorbar(tissue_plot, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(image)
    if len(centroids):
        axes[2].scatter(
            centroids[:, 0],
            centroids[:, 1],
            c=labels,
            cmap="tab10",
            vmin=0,
            vmax=len(NUCLEI_CLASSES) - 1,
            s=14,
            marker="x",
            linewidths=1,
        )
        if show_boxes:
            color_map = plt.get_cmap("tab10")
            for box, label in zip(boxes, labels, strict=True):
                x_min, y_min, x_max, y_max = box
                axes[2].add_patch(
                    Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        fill=False,
                        edgecolor=color_map(int(label) % 10),
                        linewidth=0.6,
                        alpha=0.7,
                    )
                )
    axes[2].set_title(f"Nuclei instances: {len(centroids)}")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    plt.show()
