"""Load every FEMTO bearing once (filling the .npz cache) and write a summary table."""
import argparse
import logging
from pathlib import Path

from bearing_attention import femto
from bearing_attention.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    paths = cfg["paths"]
    bearings = femto.load_all(paths["raw"] / "femto", paths["processed"] / "femto", subsets=None)
    table = femto.summary(bearings)

    out = paths["results"] / "femto_bearings.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
