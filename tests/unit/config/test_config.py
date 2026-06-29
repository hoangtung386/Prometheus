from prometheus.config import ModelConfig, TrainingConfig


def test_default_config() -> None:
    cfg = ModelConfig()
    assert cfg.in_chans == 3
    assert cfg.num_classes == 1
    assert cfg.encoder_dims == [96, 192, 384, 768]
    assert cfg.encoder_depths == [3, 3, 9, 3]
    assert cfg.drop_path_rate == 0.1
    assert cfg.n_heads == 8
    assert cfg.d_ff == 3072
    assert cfg.d_expert == 256
    assert cfg.window_size == 8
    assert cfg.num_transformer_blocks == 6
    assert cfg.num_experts == 16
    assert cfg.moe_top_k == 2


def test_custom_config() -> None:
    cfg = ModelConfig(in_chans=4, num_classes=5, n_heads=16, num_experts=256)
    assert cfg.in_chans == 4
    assert cfg.num_classes == 5
    assert cfg.n_heads == 16
    assert cfg.num_experts == 256


def test_immutable_default() -> None:
    cfg1 = ModelConfig()
    cfg2 = ModelConfig()
    cfg1.encoder_dims.append(999)
    assert 999 not in cfg2.encoder_dims


def test_default_training_config() -> None:
    cfg = TrainingConfig()
    assert cfg.batch_size == 4
    assert cfg.epochs == 100
    assert cfg.lr == 1e-4
    assert cfg.weight_decay == 1e-2
    assert cfg.betas == (0.9, 0.999)
    assert cfg.eps == 1e-8
    assert cfg.warmup_epochs == 5
    assert cfg.gradient_clip_norm == 1.0
    assert cfg.scheduler_min_lr == 1e-6
    assert cfg.num_workers == 4
    assert cfg.amp is True
    assert cfg.seed == 42
    assert cfg.use_class_weights is True
    assert cfg.class_weight_power == 0.5
    assert cfg.early_stopping_patience is None
    assert cfg.early_stopping_monitor == "combined"
    assert cfg.tissue_context_warmup_epochs == 0


def test_custom_training_config() -> None:
    cfg = TrainingConfig(batch_size=8, epochs=50, lr=1e-3, weight_decay=5e-4)
    assert cfg.batch_size == 8
    assert cfg.epochs == 50
    assert cfg.lr == 1e-3
    assert cfg.weight_decay == 5e-4
