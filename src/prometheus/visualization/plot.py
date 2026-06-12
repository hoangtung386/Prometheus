from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn as nn

from ..data.puma_dataset import NUCLEI_CLASSES, TISSUE_CLASSES


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

    tissue_overlay = tissue_mask.numpy() if tissue_mask.ndim == 2 else tissue_mask.argmax(dim=0).numpy()
    nuclei_overlay = nuclei_mask.numpy() if nuclei_mask.ndim == 2 else nuclei_mask.argmax(dim=0).numpy()

    n_tissue = len(TISSUE_CLASSES)
    n_nuclei = len(NUCLEI_CLASSES)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img.permute(1, 2, 0))
    axes[0].set_title("Input (normalized)")
    axes[1].imshow(tissue_overlay, cmap="tab10", vmin=0, vmax=n_tissue - 1)
    axes[1].set_title(f"Tissue ({n_tissue} classes)")
    axes[2].imshow(nuclei_overlay, cmap="tab20", vmin=0, vmax=n_nuclei - 1)
    axes[2].set_title(f"Nuclei ({n_nuclei} classes)")
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
    print_stats: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from ..training import dice_score

    img, targets = dataset[idx]
    pred = predict_sample(model, img, model_type, device)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(img.permute(1, 2, 0))
    axes[0, 0].set_title("Input")

    def _to_index(mask):
        return mask if mask.ndim == 2 else mask.argmax(dim=0)

    n_tissue = len(TISSUE_CLASSES)
    n_nuclei = len(NUCLEI_CLASSES)
    t_vmax = n_tissue - 1
    n_vmax = n_nuclei - 1

    if model_type == "UNetTissue":
        axes[0, 1].imshow(_to_index(targets["tissue"]), cmap="tab10", vmin=0, vmax=t_vmax)
        axes[0, 1].set_title("Tissue GT")
        axes[0, 2].imshow(pred, cmap="tab10", vmin=0, vmax=t_vmax)
        axes[0, 2].set_title("Tissue Pred")
        if print_stats:
            logits = torch.nn.functional.one_hot(torch.from_numpy(pred).long(), num_classes=n_tissue)
            logits = logits.permute(2, 0, 1).unsqueeze(0).float()
            score = dice_score(logits, targets["tissue"].unsqueeze(0)).item()
            print(f"Tissue present-fg Dice: {score:.4f}")
            print(f"Tissue GT counts: {dict(zip(*np.unique(_to_index(targets['tissue']).numpy(), return_counts=True)))}")
            print(f"Tissue Pred counts: {dict(zip(*np.unique(pred, return_counts=True)))}")
    else:
        axes[0, 1].imshow(_to_index(targets["tissue"]), cmap="tab10", vmin=0, vmax=t_vmax)
        axes[0, 1].set_title("Tissue GT")
        axes[0, 2].imshow(pred["tissue"], cmap="tab10", vmin=0, vmax=t_vmax)
        axes[0, 2].set_title("Tissue Pred")
        axes[1, 1].imshow(_to_index(targets["nuclei"]), cmap="tab20", vmin=0, vmax=n_vmax)
        axes[1, 1].set_title("Nuclei GT")
        axes[1, 2].imshow(pred["nuclei"], cmap="tab20", vmin=0, vmax=n_vmax)
        axes[1, 2].set_title("Nuclei Pred")
        axes[1, 0].axis("off")
        if print_stats:
            t_logits = torch.nn.functional.one_hot(torch.from_numpy(pred["tissue"]).long(), num_classes=n_tissue)
            t_logits = t_logits.permute(2, 0, 1).unsqueeze(0).float()
            n_logits = torch.nn.functional.one_hot(torch.from_numpy(pred["nuclei"]).long(), num_classes=n_nuclei)
            n_logits = n_logits.permute(2, 0, 1).unsqueeze(0).float()
            t_score = dice_score(t_logits, targets["tissue"].unsqueeze(0)).item()
            n_score = dice_score(n_logits, targets["nuclei"].unsqueeze(0)).item()
            print(f"Tissue present-fg Dice: {t_score:.4f}")
            print(f"Nuclei present-fg Dice: {n_score:.4f}")
            print(f"Tissue GT counts: {dict(zip(*np.unique(_to_index(targets['tissue']).numpy(), return_counts=True)))}")
            print(f"Tissue Pred counts: {dict(zip(*np.unique(pred['tissue'], return_counts=True)))}")
            print(f"Nuclei GT counts: {dict(zip(*np.unique(_to_index(targets['nuclei']).numpy(), return_counts=True)))}")
            print(f"Nuclei Pred counts: {dict(zip(*np.unique(pred['nuclei'], return_counts=True)))}")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
