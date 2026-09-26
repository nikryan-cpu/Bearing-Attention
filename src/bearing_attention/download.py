"""Helpers for fetching and unpacking the dataset archives."""
import http.client
import logging
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

CHUNK_BYTES = 1 << 20
# Cloudflare-fronted hosts answer 403 to urllib's default "Python-urllib/x.y"
USER_AGENT = "bearing-attention-download"


def remote_size(url):
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


def download(url, dest, expected_bytes=None, retries=20):
    """Download `url` to `dest`, resuming from `dest.part` after a dropped connection.

    Gives up after `retries` failed attempts in a row that brought no new data.
    """
    dest = Path(dest)
    total = expected_bytes or remote_size(url)
    part = dest.with_name(dest.name + ".part")
    if dest.exists():
        if dest.stat().st_size == total:
            log.info("%s is already downloaded", dest.name)
            return dest
        log.warning("%s has an unexpected size, resuming it as a partial download", dest.name)
        dest.replace(part)
    dest.parent.mkdir(parents=True, exist_ok=True)

    failures = 0
    while (have := _size(part)) != total:
        if have > total:
            part.unlink()
            continue
        error = None
        try:
            _fetch(url, part, have, total)
        except urllib.error.HTTPError as err:
            if err.code < 500:
                raise
            error = err
        except (OSError, http.client.HTTPException) as err:
            error = err
        failures = 0 if _size(part) > have else failures + 1
        if failures > retries:
            raise RuntimeError(f"could not download {url}: {error or 'no data received'}")
        if error is not None:
            wait = min(60, 5 * max(failures, 1))
            log.warning("download interrupted (%s), retrying in %d s", error, wait)
            time.sleep(wait)

    part.replace(dest)
    return dest


def _size(path):
    return path.stat().st_size if path.exists() else 0


def _fetch(url, part, start, total):
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as resp:
        if start and resp.status != 206:
            start = 0
        step = max(total // 20, CHUNK_BYTES)
        next_report = (start // step + 1) * step
        done = start
        with open(part, "ab" if start else "wb") as f:
            while chunk := resp.read(CHUNK_BYTES):
                f.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    log.info("%s: %d / %d MB", part.stem, done >> 20, total >> 20)
                    next_report += step


def extract_member(archive, name_suffix, dest_dir):
    """Copy the single member of a zip whose name ends with `name_suffix` into `dest_dir`."""
    with zipfile.ZipFile(archive) as zf:
        matches = [info for info in zf.infolist() if info.filename.endswith(name_suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one '{name_suffix}' in {archive}, found {len(matches)}")
        info = matches[0]
        target = Path(dest_dir) / Path(info.filename).name
        if target.exists() and target.stat().st_size == info.file_size:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        log.info("unpacking %s from %s", target.name, Path(archive).name)
        with zf.open(info) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK_BYTES)
    return target


def extract_zip(archive, dest_dir, keep=None):
    """Extract a zip into `dest_dir`, unpacking zips found inside it in place.

    `keep(name)` selects members by their path inside the archive.
    """
    with zipfile.ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if keep is None or keep(name)]
        zf.extractall(dest_dir, members)
    for name in members:
        if name.lower().endswith(".zip"):
            nested = Path(dest_dir) / name
            extract_zip(nested, nested.parent)
            nested.unlink()
    return members


# The IMS data is a .7z holding one .rar per test. The standard library reads
# neither format, and every Python rar reader shells out to an external tool
# anyway, so the tool is used directly for both steps.
def find_archive_tool():
    """Return ("bsdtar" | "7z", executable) or None if no suitable tool is installed."""
    candidates = []
    if os.name == "nt":
        # Windows 10+ ships bsdtar as System32\tar.exe; Git's GNU tar can shadow it on PATH
        candidates.append(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "tar.exe")
    candidates += [shutil.which("bsdtar"), shutil.which("tar")]
    for exe in candidates:
        if exe and Path(exe).exists() and "bsdtar" in _version_string(exe):
            return "bsdtar", str(exe)
    for name in ("7z", "7zz"):
        if exe := shutil.which(name):
            return "7z", exe
    return None


def _version_string(exe):
    try:
        return subprocess.run([str(exe), "--version"], capture_output=True, text=True).stdout
    except OSError:
        return ""


def list_archive(tool, archive):
    kind, exe = tool
    if kind == "bsdtar":
        out = _run([exe, "-tf", str(archive)])
        return [line.strip() for line in out.splitlines() if line.strip()]
    out = _run([exe, "l", "-ba", "-slt", str(archive)])
    return [line.removeprefix("Path = ") for line in out.splitlines() if line.startswith("Path = ")]


def extract_archive(tool, archive, dest_dir, members=()):
    kind, exe = tool
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    if kind == "bsdtar":
        _run([exe, "-xf", str(archive), "-C", str(dest_dir), *members])
    else:
        _run([exe, "x", "-y", f"-o{dest_dir}", str(archive), *members])


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{result.stderr.strip()}")
    return result.stdout
