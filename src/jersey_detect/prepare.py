"""Find the dataset on disk, or download it from Roboflow.

Order of preference:

1. A folder that already has ``train/``, ``valid/`` and ``test/`` with ``images/`` inside, found
   by searching the given roots (on Kaggle: ``/kaggle/input``).
2. A Roboflow download, if an API key is available (``ROBOFLOW_API_KEY`` environment variable or
   Kaggle secret). The key is never written to disk or to the logs.
"""

from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from pathlib import Path

from jersey_detect.data import find_split_dirs

DEFAULT_WORKSPACE = "muzammal-masmatixinnovations-co-uk"
DEFAULT_PROJECT = "jersey-qyuun-s5ab4"
DEFAULT_VERSION = 1


def find_dataset(roots: list[str | os.PathLike], max_depth: int = 5) -> Path | None:
    """Return the first directory under ``roots`` that has all three splits."""
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        candidates = [root]
        for depth in range(max_depth):
            next_level = []
            for c in candidates:
                try:
                    find_split_dirs(c)
                    return c
                except FileNotFoundError:
                    pass
                if depth < max_depth - 1:
                    try:
                        next_level += sorted(p for p in c.iterdir() if p.is_dir())
                    except OSError:
                        continue
            candidates = next_level
    return None


def kaggle_secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient  # only exists on Kaggle

        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def roboflow_api_key() -> str | None:
    return os.environ.get("ROBOFLOW_API_KEY") or kaggle_secret("ROBOFLOW_API_KEY")


def download_roboflow(
    dest: str | os.PathLike,
    api_key: str,
    workspace: str = DEFAULT_WORKSPACE,
    project: str = DEFAULT_PROJECT,
    version: int = DEFAULT_VERSION,
) -> Path:
    """Download the YOLOv8 export through the Roboflow REST API and unzip it into ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://api.roboflow.com/{workspace}/{project}/{version}/yolov8?api_key={api_key}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        info = json.load(resp)
    link = info.get("export", {}).get("link")
    if not link:
        raise RuntimeError("Roboflow did not return an export link (check the key and project)")
    archive = dest / "export.zip"
    urllib.request.urlretrieve(link, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    archive.unlink()
    found = find_dataset([dest])
    if found is None:
        raise RuntimeError(f"Downloaded export in {dest} has no train/valid/test split")
    return found


def locate_or_download(search_roots: list[str], download_dir: str) -> Path:
    found = find_dataset(search_roots)
    if found is not None:
        return found
    key = roboflow_api_key()
    if not key:
        print("No ROBOFLOW_API_KEY found (environment variable or attached Kaggle secret).")
        raise FileNotFoundError(
            "Dataset not found. Attach it as a Kaggle input (folder with train/valid/test) "
            "or add a ROBOFLOW_API_KEY secret."
        )
    try:
        return download_roboflow(download_dir, key)
    except Exception as exc:  # noqa: BLE001 - fall back to the official SDK
        reason = str(exc).replace(key, "***")
        print(f"Roboflow REST download failed ({type(exc).__name__}: {reason}); trying the SDK")
    return _download_with_sdk(download_dir, key)


def _download_with_sdk(dest: str, api_key: str) -> Path:
    import subprocess
    import sys

    try:
        import roboflow  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "roboflow"], check=True)
    from roboflow import Roboflow

    project = Roboflow(api_key=api_key).workspace(DEFAULT_WORKSPACE).project(DEFAULT_PROJECT)
    project.version(DEFAULT_VERSION).download("yolov8", location=str(dest), overwrite=True)
    found = find_dataset([dest])
    if found is None:
        raise RuntimeError(f"Roboflow download in {dest} has no train/valid/test split")
    return found
