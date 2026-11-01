import ast
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_train.ipynb"
YOLO_NOTEBOOK = ROOT / "notebooks" / "kaggle_yolo_baseline.ipynb"


def _code_cells():
    nb = json.loads(NOTEBOOK.read_text())
    return [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.mark.parametrize("path", [NOTEBOOK, YOLO_NOTEBOOK], ids=["train", "yolo"])
def test_notebook_is_valid_python_and_has_no_outputs(path):
    nb = json.loads(path.read_text())
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == []


def test_notebook_points_at_files_that_exist():
    source = "\n".join(_code_cells())
    for config in ("configs/kaggle.json", "configs/smoke.json"):
        assert config in source
        assert (ROOT / config).exists()
    assert "/kaggle/working/results" in source


def test_configs_are_valid():
    from jersey_detect.pipeline import PipelineConfig

    for name in ("kaggle.json", "smoke.json"):
        settings = json.loads((ROOT / "configs" / name).read_text())
        PipelineConfig(data_root="x", out_dir="y", **settings)
    pytest.importorskip("torch")
    from jersey_detect.engine import DetectorConfig

    DetectorConfig.from_dict({"data_root": "x", "out_dir": "y",
                              **json.loads((ROOT / "configs" / "jersey_net.json").read_text())})


@pytest.mark.skipif(os.environ.get("RUN_NOTEBOOK") != "1", reason="set RUN_NOTEBOOK=1 to execute it")
def test_notebook_runs_end_to_end_in_smoke_mode(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    out = tmp_path / "results"
    monkeypatch.setenv("JND_SMOKE", "1")
    monkeypatch.setenv("JND_REPO_DIR", str(ROOT))
    monkeypatch.setenv("JND_OUT", str(out))
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbclient.NotebookClient(nb, timeout=1800, kernel_name="python3").execute()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["errors"] == []
    assert "JerseyNet, from scratch (seed 0)" in summary["results"]


@pytest.mark.skipif(os.environ.get("RUN_YOLO") != "1", reason="set RUN_YOLO=1 (needs ultralytics)")
def test_yolo_notebook_runs_in_smoke_mode(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ultralytics")
    out = tmp_path / "yolo"
    monkeypatch.setenv("JND_SMOKE", "1")
    monkeypatch.setenv("JND_REPO_DIR", str(ROOT))
    monkeypatch.setenv("JND_OUT", str(out))
    nb = nbformat.read(YOLO_NOTEBOOK, as_version=4)
    nbclient.NotebookClient(nb, timeout=1800, kernel_name="python3").execute()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["errors"] == [] and len(summary["results"]) == 1
