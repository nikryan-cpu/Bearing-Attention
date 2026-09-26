from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path=None):
    """Read the YAML config and turn every entry of `paths` into an absolute Path."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = path.resolve().parent
    cfg["paths"] = {name: (base / p).resolve() for name, p in cfg["paths"].items()}
    return cfg
