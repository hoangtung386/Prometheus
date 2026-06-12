from __future__ import annotations

import math

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
        self.target_count = torch.zeros(self.num_classes)
        self.pred_count = torch.zeros(self.num_classes)

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
            self.target_count[c] += t.sum().item()
            self.pred_count[c] += p.sum().item()

    def compute(self) -> dict[str, torch.Tensor]:
        eps = 1e-8
        dice_den = 2 * self.tp + self.fp + self.fn
        iou_den = self.tp + self.fp + self.fn
        sens_den = self.tp + self.fn
        prec_den = self.tp + self.fp

        dice = torch.full_like(self.tp, torch.nan)
        iou = torch.full_like(self.tp, torch.nan)
        sensitivity = torch.full_like(self.tp, torch.nan)
        precision = torch.full_like(self.tp, torch.nan)

        dice_mask = dice_den > 0
        iou_mask = iou_den > 0
        sens_mask = sens_den > 0
        prec_mask = prec_den > 0
        dice[dice_mask] = (2 * self.tp[dice_mask]) / (dice_den[dice_mask] + eps)
        iou[iou_mask] = self.tp[iou_mask] / (iou_den[iou_mask] + eps)
        sensitivity[sens_mask] = self.tp[sens_mask] / (sens_den[sens_mask] + eps)
        precision[prec_mask] = self.tp[prec_mask] / (prec_den[prec_mask] + eps)

        specificity = (self.tn + eps) / (self.tn + self.fp + eps)
        accuracy = (self.tp + self.tn + eps) / (self.tp + self.tn + self.fp + self.fn + eps)
        return {
            "dice": dice,
            "iou": iou,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "accuracy": accuracy,
            "target_count": self.target_count.clone(),
            "pred_count": self.pred_count.clone(),
        }

    @staticmethod
    def _nanmean(x: torch.Tensor) -> torch.Tensor:
        valid = ~torch.isnan(x)
        if not valid.any():
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return x[valid].mean()

    @staticmethod
    def _fmt(value: float) -> str:
        return "nan" if math.isnan(value) else f"{value:.4f}"

    def _format_row(self, name: str, dice: float, iou: float, sens: float, spec: float, prec: float) -> str:
        return (
            f"    {name:<25} {self._fmt(dice):<8} "
            f"{self._fmt(iou):<8} {self._fmt(sens):<8} "
            f"{self._fmt(spec):<8} {self._fmt(prec):<8}"
        )

    def _format_mean_row(self, label: str, dice: float, iou: float) -> str:
        return f"    {label:<25} {self._fmt(dice):<8} {self._fmt(iou):<8}"

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
        fg_dice = self._nanmean(metrics["dice"][1:]).item() if self.num_classes > 1 else 0.0
        fg_iou = self._nanmean(metrics["iou"][1:]).item() if self.num_classes > 1 else 0.0
        all_dice = self._nanmean(metrics["dice"]).item()
        all_iou = self._nanmean(metrics["iou"]).item()
        lines.append(self._format_mean_row("Mean (present fg)", fg_dice, fg_iou))
        lines.append(self._format_mean_row("Mean (present all)", all_dice, all_iou))
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
            log[f"{modality_name}/Support/target/{name}"] = metrics["target_count"][c].item()
            log[f"{modality_name}/Support/pred/{name}"] = metrics["pred_count"][c].item()
        log[f"{modality_name}/Dice/mean_present_fg"] = self._nanmean(metrics["dice"][1:]).item() if self.num_classes > 1 else 0.0
        log[f"{modality_name}/IoU/mean_present_fg"] = self._nanmean(metrics["iou"][1:]).item() if self.num_classes > 1 else 0.0
        log[f"{modality_name}/Dice/mean_present_all"] = self._nanmean(metrics["dice"]).item()
        log[f"{modality_name}/IoU/mean_present_all"] = self._nanmean(metrics["iou"]).item()
        return log
