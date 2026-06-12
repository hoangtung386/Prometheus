from __future__ import annotations

import torch


class SegmentationEvaluator:
    """Accumulates per-class TP/FP/FN/TN across batches and computes metrics.

    Metrics (per class):
        Dice, IoU (Jaccard), Sensitivity (Recall), Specificity, Precision, Accuracy.

    Usage:
        evaluator = SegmentationEvaluator(num_classes=6, class_names=TISSUE_CLASSES)
        for batch in loader:
            evaluator.update(logits, targets)
        print(evaluator.summary("Tissue"))
    """

    def __init__(
        self,
        num_classes: int,
        class_names: list[str] | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.class_names = class_names or [f"c{i}" for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
        self.tn = torch.zeros(self.num_classes)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch.

        Args:
            logits: (B, C, H, W) raw logits from model.
            targets: (B, H, W) class-index long tensor.
        """
        preds = logits.argmax(dim=1)
        for c in range(self.num_classes):
            p = preds == c
            t = targets == c
            self.tp[c] += (p & t).sum().item()
            self.fp[c] += (p & ~t).sum().item()
            self.fn[c] += (~p & t).sum().item()
            self.tn[c] += (~p & ~t).sum().item()

    def compute(self) -> dict[str, torch.Tensor]:
        eps = 1e-8
        dice = (2 * self.tp + eps) / (2 * self.tp + self.fp + self.fn + eps)
        iou = (self.tp + eps) / (self.tp + self.fp + self.fn + eps)
        sensitivity = (self.tp + eps) / (self.tp + self.fn + eps)
        specificity = (self.tn + eps) / (self.tn + self.fp + eps)
        precision = (self.tp + eps) / (self.tp + self.fp + eps)
        accuracy = (self.tp + self.tn + eps) / (self.tp + self.tn + self.fp + self.fn + eps)
        return {
            "dice": dice,
            "iou": iou,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "accuracy": accuracy,
        }

    def _format_row(self, name: str, dice: float, iou: float, sens: float, spec: float, prec: float) -> str:
        return f"    {name:<25} {dice:.4f}   {iou:.4f}   {sens:.4f}   {spec:.4f}   {prec:.4f}"

    def _format_mean_row(self, label: str, dice: float, iou: float) -> str:
        return f"    {label:<25} {dice:.4f}   {iou:.4f}"

    def summary(self, modality_name: str = "Tissue") -> str:
        metrics = self.compute()
        header = f"    {'Class':<25} {'Dice':<8} {'IoU':<8} {'Sens':<8} {'Spec':<8} {'Prec':<8}"
        sep = "    " + "-" * 69
        lines = [f"  [{modality_name}] Per-class metrics:", header, sep]
        for c in range(self.num_classes):
            name = self.class_names[c] if c < len(self.class_names) else f"c{c}"
            lines.append(self._format_row(
                name,
                metrics["dice"][c].item(),
                metrics["iou"][c].item(),
                metrics["sensitivity"][c].item(),
                metrics["specificity"][c].item(),
                metrics["precision"][c].item(),
            ))
        lines.append(sep)
        fg_dice = metrics["dice"][1:].mean().item() if self.num_classes > 1 else 0.0
        fg_iou = metrics["iou"][1:].mean().item() if self.num_classes > 1 else 0.0
        all_dice = metrics["dice"].mean().item()
        all_iou = metrics["iou"].mean().item()
        lines.append(self._format_mean_row("Mean (foreground)", fg_dice, fg_iou))
        lines.append(self._format_mean_row("Mean (all classes)", all_dice, all_iou))
        return "\n".join(lines)

    def log_dict(self, modality_name: str = "tissue") -> dict[str, float]:
        log: dict[str, float] = {}
        metrics = self.compute()
        for c in range(self.num_classes):
            name = self.class_names[c] if c < len(self.class_names) else f"c{c}"
            log[f"{modality_name}/Dice/{name}"] = metrics["dice"][c].item()
            log[f"{modality_name}/IoU/{name}"] = metrics["iou"][c].item()
            log[f"{modality_name}/Sens/{name}"] = metrics["sensitivity"][c].item()
            log[f"{modality_name}/Spec/{name}"] = metrics["specificity"][c].item()
            log[f"{modality_name}/Prec/{name}"] = metrics["precision"][c].item()
        log[f"{modality_name}/Dice/mean_fg"] = metrics["dice"][1:].mean().item() if self.num_classes > 1 else 0.0
        log[f"{modality_name}/IoU/mean_fg"] = metrics["iou"][1:].mean().item() if self.num_classes > 1 else 0.0
        return log
