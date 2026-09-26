"""Download FEMTO/PRONOSTIA and one IMS test into data/raw.

Both archives come from the PHM Society mirror of the NASA Prognostics Data
Repository (https://data.phmsociety.org/nasa/). They are free to use with a
citation but may not be redistributed, so nothing fetched here is committed.

    python scripts/download_data.py                 # both datasets
    python scripts/download_data.py femto
    python scripts/download_data.py ims --ims-archive ~/Downloads/4.+Bearings.zip
"""
import argparse
import logging
import shutil
from pathlib import Path

from bearing_attention.config import load_config
from bearing_attention.download import (
    download,
    extract_archive,
    extract_member,
    extract_zip,
    find_archive_tool,
    list_archive,
)

log = logging.getLogger("download_data")

DOC_SUFFIXES = (".pdf", ".txt", ".doc", ".docx")


def get_archive(spec, archives_dir, local_copy):
    if local_copy:
        return Path(local_copy).expanduser()
    return download(spec["url"], archives_dir / spec["archive"], spec.get("archive_bytes"))


def prepare_femto(cfg, local_copy=None):
    spec = cfg["datasets"]["femto"]
    archives_dir = cfg["paths"]["archives"]
    out_dir = cfg["paths"]["raw"] / "femto"

    outer = get_archive(spec, archives_dir, local_copy)
    inner = extract_member(outer, spec["inner_archive"], archives_dir)
    log.info("extracting FEMTO into %s", out_dir)
    extract_zip(inner, out_dir)
    inner.unlink()

    for subset in sorted(p for p in out_dir.rglob("*") if p.is_dir() and _has_bearing_dirs(p)):
        bearings = sorted(d.name for d in subset.iterdir() if d.name.startswith("Bearing"))
        n_files = sum(1 for _ in subset.rglob("acc_*.csv"))
        log.info("%s: %d bearings, %d snapshots", subset.relative_to(out_dir), len(bearings), n_files)


def _has_bearing_dirs(path):
    return any(d.is_dir() and d.name.startswith("Bearing") for d in path.iterdir())


def prepare_ims(cfg, test, local_copy=None):
    spec = cfg["datasets"]["ims"]
    archives_dir = cfg["paths"]["archives"]
    out_dir = cfg["paths"]["raw"] / "ims"

    tool = find_archive_tool()
    if tool is None:
        raise SystemExit(
            "IMS is packed as .7z/.rar: install bsdtar (libarchive-tools) or 7-Zip and run again"
        )

    outer = get_archive(spec, archives_dir, local_copy)
    inner = extract_member(outer, spec["inner_archive"], archives_dir)
    members = list_archive(tool, inner)
    rar = [m for m in members if Path(m).name == f"{test}.rar"]
    if not rar:
        found = ", ".join(Path(m).name for m in members if m.endswith(".rar"))
        raise SystemExit(f"{test}.rar is not in {inner.name} (found: {found})")
    docs = [m for m in members if m.lower().endswith(DOC_SUFFIXES)]

    tmp_dir = archives_dir / "ims_tmp"
    log.info("extracting %s with %s", rar[0], tool[0])
    extract_archive(tool, inner, tmp_dir, rar + docs)
    extract_archive(tool, tmp_dir / rar[0], out_dir)
    for doc in docs:
        shutil.move(tmp_dir / doc, out_dir / Path(doc).name)
    shutil.rmtree(tmp_dir)
    inner.unlink()

    for folder in sorted(p for p in out_dir.rglob("*") if p.is_dir()):
        n_files = sum(1 for f in folder.iterdir() if f.is_file())
        if n_files:
            log.info("%s: %d snapshots", folder.relative_to(out_dir), n_files)


def main():
    parser = argparse.ArgumentParser(description="Download and unpack the bearing datasets.")
    parser.add_argument("datasets", nargs="*", choices=["femto", "ims"], default=["femto", "ims"])
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    parser.add_argument("--femto-archive", help="FEMTO zip downloaded by hand (skips the download)")
    parser.add_argument("--ims-archive", help="IMS zip downloaded by hand (skips the download)")
    parser.add_argument("--ims-test", help="IMS test folder to extract, e.g. 2nd_test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if "femto" in args.datasets:
        prepare_femto(cfg, args.femto_archive)
    if "ims" in args.datasets:
        prepare_ims(cfg, args.ims_test or cfg["datasets"]["ims"]["test"], args.ims_archive)


if __name__ == "__main__":
    main()
