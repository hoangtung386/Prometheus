from prometheus.cli.main import build_parser
from prometheus.config import load_experiment_config


def test_cli_exposes_all_workflows() -> None:
    parser = build_parser()
    assert parser.parse_args(["audit", "--data-root", "/tmp/data"]).command == "audit"
    assert parser.parse_args(["train", "--config", "run.toml"]).command == "train"
    assert (
        parser.parse_args(
            [
                "evaluate",
                "--config",
                "run.toml",
                "--checkpoint",
                "model.pt",
            ]
        ).command
        == "evaluate"
    )
    assert (
        parser.parse_args(
            [
                "predict",
                "--config",
                "run.toml",
                "--checkpoint",
                "model.pt",
                "--input",
                "image.tif",
                "--output",
                "output",
            ]
        ).command
        == "predict"
    )


def test_experiment_config_loads_and_validates(tmp_path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(
        """
[model]
encoder_dims = [8, 16, 32, 64]
encoder_depths = [1, 1, 1, 1]
n_heads = 8

[training]
epochs = 2

[data]
root = "/tmp/puma"
validation_fraction = 0.2

[evaluation]
track = "track1"
""",
        encoding="utf-8",
    )
    config = load_experiment_config(path)
    assert config.model.encoder_dims[-1] == 64
    assert config.training.epochs == 2
    assert config.evaluation.track == "track1"
