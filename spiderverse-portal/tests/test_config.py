import pytest
import yaml

from spiderverse_portal.config import (
    DEFAULT_CONFIG_PATH,
    AppConfig,
    ConfigError,
    config_from_dict,
    inference_size,
    load_config,
    save_config,
)


def test_defaults_validate():
    AppConfig().validate()


def test_shipped_default_yaml_matches_dataclass_defaults():
    """configs/default.yaml must stay in sync (regenerate with tools/write_default_config.py)."""
    assert load_config(DEFAULT_CONFIG_PATH, include_user=False).to_dict() == AppConfig().to_dict()


def test_override_and_type_coercion():
    cfg = config_from_dict({"camera": {"width": 1280, "height": 720}, "comic": {"saturation": 2}})
    assert cfg.camera.width == 1280 and isinstance(cfg.comic.saturation, float)
    assert cfg.comic.shadow_tint == AppConfig().comic.shadow_tint


def test_tuple_fields_load_from_lists():
    cfg = config_from_dict({"comic": {"shadow_tint": [0.1, 0.2, 0.3]}})
    assert cfg.comic.shadow_tint == (0.1, 0.2, 0.3)
    with pytest.raises(ConfigError):
        config_from_dict({"comic": {"shadow_tint": [0.1, 0.2]}})


@pytest.mark.parametrize(
    "data",
    [
        {"camera": {"widht": 640}},
        {"nope": {}},
        {"camera": {"width": "wide"}},
        {"camera": {"mirror": "yes"}},
        {"portal": {"stay_score": 0.9, "enter_score": 0.5}},
        {"stylization": {"neural": {"engine": "tpu"}}},
        {"comic": {"color_levels": 20}},
    ],
)
def test_invalid_configs_rejected(data):
    with pytest.raises(ConfigError):
        config_from_dict(data)


def test_save_and_reload_roundtrip(tmp_path):
    cfg = AppConfig()
    cfg.comic.halftone_strength = 0.123
    cfg.stylization.neural.model = "celeba_distill"
    path = save_config(cfg, tmp_path / "user.yaml")
    data = yaml.safe_load(path.read_text())
    assert data["comic"]["halftone_strength"] == 0.123
    again = load_config(path, include_user=False)
    assert again.to_dict() == cfg.to_dict()


def test_missing_explicit_config_is_an_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_inference_size_keeps_aspect_and_multiple_of_4():
    cfg = AppConfig()
    assert inference_size(cfg, 640, 480) == (320, 240)
    assert inference_size(cfg, 1280, 720) == (320, 180)
    cfg.stylization.neural.inference_width = 257
    w, h = inference_size(cfg, 640, 480)
    assert w % 4 == 0 and h % 4 == 0
