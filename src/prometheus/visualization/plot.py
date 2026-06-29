"""Lightweight visualization helpers built on public prediction contracts."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data.puma import NUCLEI_CLASSES, TISSUE_CLASSES


@torch.no_grad()
def predict_sample(
    model: nn.Module,
    image_tensor: torch.Tensor,
    model_type: str,
    device: torch.device,
) -> np.ndarray | dict[str, np.ndarray]:
    model.eval()
    model_output = model(image_tensor.unsqueeze(0).to(device))
    if model_type == "UNetTissue":
        return model_output.argmax(dim=1).squeeze(0).cpu().numpy()
    tissue_logits, nuclei_logits, _ = model_output
    return {
        "tissue": tissue_logits.argmax(dim=1).squeeze(0).cpu().numpy(),
        "nuclei": nuclei_logits.argmax(dim=1).squeeze(0).cpu().numpy(),
    }


def _index_mask(mask):
    return mask if mask.ndim == 2 else mask.argmax(dim=0)


def visualize_sample(dataset, idx: int = 0) -> None:
    import matplotlib.pyplot as plt

    image, targets = dataset[idx]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("Input")
    axes[1].imshow(_index_mask(targets["tissue"]), cmap="tab10", vmin=0, vmax=len(TISSUE_CLASSES) - 1)
    axes[1].set_title("Tissue")
    axes[2].imshow(_index_mask(targets["nuclei"]), cmap="tab20", vmin=0, vmax=len(NUCLEI_CLASSES) - 1)
    axes[2].set_title("Nuclei")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    plt.show()


def show_prediction(
    model: nn.Module,
    dataset,
    idx: int = 0,
    model_type: str = "DualUNet",
    device: torch.device | None = None,
) -> None:
    import matplotlib.pyplot as plt

    selected_device = device or torch.device("cpu")
    image, targets = dataset[idx]
    prediction = predict_sample(model, image, model_type, selected_device)
    figure, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(image.permute(1, 2, 0))
    axes[0, 0].set_title("Input")
    axes[0, 1].imshow(_index_mask(targets["tissue"]), cmap="tab10")
    axes[0, 1].set_title("Tissue GT")
    tissue_prediction = prediction if model_type == "UNetTissue" else prediction["tissue"]
    axes[0, 2].imshow(tissue_prediction, cmap="tab10")
    axes[0, 2].set_title("Tissue prediction")
    if model_type != "UNetTissue":
        axes[1, 1].imshow(_index_mask(targets["nuclei"]), cmap="tab20")
        axes[1, 1].set_title("Nuclei GT")
        axes[1, 2].imshow(prediction["nuclei"], cmap="tab20")
        axes[1, 2].set_title("Nuclei prediction")
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    plt.show()
