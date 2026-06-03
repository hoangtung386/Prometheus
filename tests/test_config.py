from prometheus.config import ModelConfig, DEFAULT_CONFIG


def test_default_config() -> None:
    cfg = ModelConfig()
    assert cfg.in_chans == 3
    assert cfg.num_classes == 1
    assert cfg.encoder_dims == [96, 192, 384, 768]
    assert cfg.encoder_depths == [3, 3, 9, 3]
    assert cfg.drop_path_rate == 0.1
    assert cfg.D == 2


def test_custom_config() -> None:
    cfg = ModelConfig(in_chans=4, num_classes=5, D=3)
    assert cfg.in_chans == 4
    assert cfg.num_classes == 5
    assert cfg.D == 3


def test_immutable_default() -> None:
    cfg1 = ModelConfig()
    cfg2 = ModelConfig()
    cfg1.encoder_dims.append(999)
    assert 999 not in cfg2.encoder_dims


def test_default_config_singleton() -> None:
    assert DEFAULT_CONFIG.in_chans == 3
    assert DEFAULT_CONFIG.num_classes == 1
