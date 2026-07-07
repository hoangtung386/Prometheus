import torch

from prometheus.inference import decode_nuclei
from prometheus.models import MultitaskOutput


def _output(center_logit: float, class_logits: list[float]) -> MultitaskOutput:
    return MultitaskOutput(
        tissue_logits=torch.zeros(1, 6, 4, 4),
        nuclei_center_logits=torch.tensor([[[[center_logit]]]]),
        nuclei_class_logits=torch.tensor(class_logits).reshape(1, -1, 1, 1),
        nuclei_offsets=torch.zeros(1, 2, 1, 1),
        nuclei_sizes=torch.ones(1, 2, 1, 1),
        auxiliary={},
    )


def test_decoder_thresholds_combined_center_and_class_confidence() -> None:
    # center=0.90 but a flat ten-way classifier gives class confidence=0.10.
    output = _output(2.1972246, [0.0] * 10)
    assert decode_nuclei(output, threshold=0.25, max_detections=1) == [[]]


def test_decoder_emits_one_detection_from_class_agnostic_center() -> None:
    output = _output(4.0, [8.0] + [0.0] * 9)
    detections = decode_nuclei(output, threshold=0.25, max_detections=1)[0]
    assert len(detections) == 1
    assert detections[0].confidence > 0.9
