from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def predict_sample(
    model: nn.Module,
    image_tensor: torch.Tensor,
    model_type: str,
    device: torch.device,
) -> Union[np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    inp = image_tensor.unsqueeze(0).to(device)
    if model_type == "UNetTissue":
        logits = model(inp)
        return logits.argmax(dim=1).squeeze(0).cpu().numpy()
    else:
        t_logits, n_logits, _ = model(inp)
        return {
            "tissue": t_logits.argmax(dim=1).squeeze(0).cpu().numpy(),
            "nuclei": n_logits.argmax(dim=1).squeeze(0).cpu().numpy(),
        }


def visualize_sample(dataset, idx: int = 0) -> None:
    import matplotlib.pyplot as plt

    img, targets = dataset[idx]
    tissue_mask = targets["tissue"]
    nuclei_mask = targets["nuclei"]

    tissue_overlay = tissue_mask.argmax(dim=0).numpy()
    nuclei_overlay = nuclei_mask.argmax(dim=0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img.permute(1, 2, 0))
    axes[0].set_title("Input (normalized)")
    axes[1].imshow(tissue_overlay, cmap="tab10", vmin=0, vmax=5)
    axes[1].set_title("Tissue (6 classes)")
    axes[2].imshow(nuclei_overlay, cmap="tab10", vmin=0, vmax=10)
    axes[2].set_title("Nuclei (10 classes)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_prediction(
    model: nn.Module,
    dataset,
    idx: int = 0,
    model_type: str = "DualUNet",
    device: torch.device = torch.device("cpu"),
) -> None:
    import matplotlib.pyplot as plt

    img, targets = dataset[idx]
    pred = predict_sample(model, img, model_type, device)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(img.permute(1, 2, 0))
    axes[0, 0].set_title("Input")

    if model_type == "UNetTissue":
        axes[0, 1].imshow(targets["tissue"].argmax(dim=0), cmap="tab10", vmin=0, vmax=5)
        axes[0, 1].set_title("Tissue GT")
        axes[0, 2].imshow(pred, cmap="tab10", vmin=0, vmax=5)
        axes[0, 2].set_title("Tissue Pred")
    else:
        axes[0, 1].imshow(targets["tissue"].argmax(dim=0), cmap="tab10", vmin=0, vmax=5)
        axes[0, 1].set_title("Tissue GT")
        axes[0, 2].imshow(pred["tissue"], cmap="tab10", vmin=0, vmax=5)
        axes[0, 2].set_title("Tissue Pred")
        axes[1, 1].imshow(targets["nuclei"].argmax(dim=0), cmap="tab10", vmin=0, vmax=10)
        axes[1, 1].set_title("Nuclei GT")
        axes[1, 2].imshow(pred["nuclei"], cmap="tab10", vmin=0, vmax=10)
        axes[1, 2].set_title("Nuclei Pred")
        axes[1, 0].axis("off")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
