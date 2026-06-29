"""Optional Ultralytics adapter isolated from Prometheus domain contracts."""

from __future__ import annotations

from ...domain import Detection, NucleusClass


class YoloNucleiDetector:
    def __init__(self, model_or_weights) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError("YOLO support is optional; install Prometheus with the 'yolo' extra") from error
        self.model = YOLO(model_or_weights)

    def predict(self, images, confidence: float = 0.25, iou: float = 0.7):
        results = self.model.predict(images, conf=confidence, iou=iou, verbose=False)
        batch = []
        classes = list(NucleusClass)
        for result in results:
            detections = []
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                class_index = int(box.cls.item())
                detections.append(
                    Detection(
                        centroid=((x_min + x_max) / 2.0, (y_min + y_max) / 2.0),
                        label=classes[class_index],
                        confidence=float(box.conf.item()),
                        box_xyxy=(x_min, y_min, x_max, y_max),
                    )
                )
            batch.append(detections)
        return batch


__all__ = ["YoloNucleiDetector"]
