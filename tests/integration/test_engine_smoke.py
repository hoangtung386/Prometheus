import torch

from prometheus.config import EngineConfig, PathsConfig, ProjectConfig, PrometheusModelConfig
from prometheus.domain import ImageMeta, MultitaskBatch, NucleiTarget, TissueTarget
from prometheus.engine import PrometheusTrainer, load_engine_checkpoint
from prometheus.models import PrometheusNet


def test_one_epoch_checkpoint_smoke(tmp_path) -> None:
    model_config = PrometheusModelConfig(
        encoder_dims=[8, 16, 32, 64],
        encoder_depths=[1, 1, 1, 1],
        tissue_decoder_depths=[1, 1, 1],
    )
    config = ProjectConfig(
        model=model_config,
        trainer=EngineConfig(epochs=1, batch_size=1, num_workers=0, amp=False, warmup_epochs=0),
        paths=PathsConfig(run_dir=str(tmp_path)),
    )
    meta = ImageMeta("sample", (32, 32), (32, 32), (32, 32), (1.0, 1.0), (0, 0))
    batch = MultitaskBatch(
        images=torch.randn(1, 3, 32, 32),
        tissue=TissueTarget(torch.randint(0, 6, (1, 32, 32))),
        nuclei=[
            NucleiTarget(
                centroids=torch.tensor([[12.0, 12.0]]),
                labels=torch.tensor([1]),
                boxes=torch.tensor([[10.0, 10.0, 14.0, 14.0]]),
            )
        ],
        metadata=[meta],
    )
    trainer = PrometheusTrainer(PrometheusNet(model_config), [batch], [batch], config, torch.device("cpu"))
    metrics = trainer.fit()
    assert "nuclei/macro_f1_summed" in metrics
    assert (tmp_path / "last.ckpt").is_file()
    assert (tmp_path / "resolved_config.json").is_file()
    assert len((tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert load_engine_checkpoint(tmp_path / "last.ckpt")["epoch"] == 0
