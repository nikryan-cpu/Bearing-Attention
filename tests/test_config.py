from bearing_attention.config import DEFAULT_CONFIG, load_config


def test_paths_are_relative_to_config_file(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("paths:\n  raw: data/raw\n", encoding="utf-8")
    assert load_config(cfg_file)["paths"]["raw"] == tmp_path / "data" / "raw"


def test_default_config_has_both_datasets():
    cfg = load_config()
    assert cfg["paths"]["raw"].parent == DEFAULT_CONFIG.parent / "data"
    for name in ("femto", "ims"):
        spec = cfg["datasets"][name]
        assert spec["url"].startswith("https://")
        snapshot_s = spec["snapshot_length"] / spec["sampling_rate_hz"]
        assert snapshot_s < spec["snapshot_interval_s"]
