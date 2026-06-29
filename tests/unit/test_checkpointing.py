import torch
from torch.utils.data import DataLoader

from prometheus.config import ModelConfig, TrainingConfig
from prometheus.training import Trainer, TrainState, load_checkpoint, save_checkpoint


def test_versioned_checkpoint_round_trip(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        model_name="tiny",
        config=ModelConfig(encoder_dims=[2, 4, 8, 16]),
        train_state=TrainState(epoch=3, global_step=12),
    )
    payload = load_checkpoint(path)
    assert payload["schema_version"] == 1
    assert payload["model_name"] == "tiny"
    assert payload["train_state"]["global_step"] == 12


def test_legacy_trainer_checkpoint_is_normalized(tmp_path) -> None:
    path = tmp_path / "legacy.pt"
    model = torch.nn.Linear(2, 1)
    torch.save(
        {
            "epoch": 4,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
            "best_dice": 0.7,
        },
        path,
    )
    payload = load_checkpoint(path)
    assert payload["legacy"] is True
    assert payload["model_state"].keys() == model.state_dict().keys()
    assert payload["train_state"]["epoch"] == 4


def test_trainer_writes_the_versioned_checkpoint_schema(tmp_path) -> None:
    model = torch.nn.Conv2d(3, 6, kernel_size=1)
    dummy = [
        (
            torch.zeros(3, 4, 4),
            {
                "tissue": torch.zeros(4, 4, dtype=torch.long),
                "nuclei": torch.zeros(4, 4, dtype=torch.long),
            },
        )
    ]
    loader = DataLoader(dummy, batch_size=1)
    config = TrainingConfig(
        model_type="UNetTissue",
        epochs=1,
        use_class_weights=False,
        log_dir=str(tmp_path / "logs"),
        ckpt_dir=str(tmp_path / "checkpoints"),
    )
    trainer = Trainer(model, loader, config, device=torch.device("cpu"), test_loader=loader)
    trainer._save_checkpoint(0, "test.pt")
    payload = load_checkpoint(tmp_path / "checkpoints" / "test.pt")
    assert payload["schema_version"] == 1
    assert payload["model_name"] == "UNetTissue"
