from __future__ import annotations

import json
from pathlib import Path


# Project root inferred from this file location:
# .../__251106_FinalCode/Github/config.py -> project root is parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parent

OUTPUT_ROOT = MODULE_ROOT / "output"
INTERMEDIATE_DIR = OUTPUT_ROOT / "intermediate"
FIGURES_NORM_DIR = OUTPUT_ROOT / "figures_normalized"
EXAMPLE_PANEL_DIR = OUTPUT_ROOT / "metric_examples"
SHARPNESS_EXAMPLES_DIR = OUTPUT_ROOT / "sharpness_examples"

DATASET_ORDER = ["UBID-ER-val", "CBID-ER-Full", "CODEBRIM", "Dacl10k", "S2DS"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MIN_BOX_SIDE_PX = 12.0

# Centrality metric used in final reporting.
CENTRALITY_CENTER_WINDOW = 0.2
CENTRALITY_HYBRID_WEIGHTS = {"overlap": 0.6, "distance": 0.4}
CENTRALITY_PRIMARY_COL = "centrality_apparent"
CENTRALITY_PRIMARY_LABEL = "Centrality"

OUTLINE_DATASETS = {"UBID-ER-val", "CBID-ER-Full"}
COLOR_OVERRIDES = {
    "UBID-ER-val": ("#56B4E9", "#000000"),
    "CBID-ER-Full": ("#29AD3C", "#444444"),
    "Dacl10k": ("#9868CF", None),
    "S2DS": ("#E97132", None),
    "CODEBRIM": ("#D86EAC", None),
}
DEFAULT_PALETTE = ["#7D8A94", "#A9A9A9", "#B6A58B", "#8A9CA8", "#9CA8B8", "#9BB2A3"]


def set_project_root(project_root: Path) -> None:
    global PROJECT_ROOT
    PROJECT_ROOT = project_root.resolve()


def default_dataset_config(project_root: Path | None = None) -> dict[str, dict[str, Path]]:
    root = project_root or PROJECT_ROOT
    base = root / "1_Datasets" / "0_TrainingDatasets" / "2_FilteredDatasets"
    return {
        "UBID-ER-val": {
            "images_dir": base / "11_UBID-ER-val" / "valid" / "images",
            "labels_dir": base / "11_UBID-ER-val" / "valid" / "labels",
        },
        "CBID-ER-Full": {
            "images_dir": base / "8.0_CBID-ER-Full" / "all" / "images",
            "labels_dir": base / "8.0_CBID-ER-Full" / "all" / "labels",
        },
        "CODEBRIM": {
            "images_dir": base / "6_CODEBRIM" / "valid" / "images",
            "labels_dir": base / "6_CODEBRIM" / "valid" / "labels",
        },
        "Dacl10k": {
            "images_dir": base / "2_Dacl10k" / "all" / "images",
            "labels_dir": base / "2_Dacl10k" / "all" / "labels",
        },
        "S2DS": {
            "images_dir": base / "5_S2DS" / "all" / "images",
            "labels_dir": base / "5_S2DS" / "all" / "labels",
        },
    }


def _resolve_dir(p: str, project_root: Path) -> Path:
    path = Path(p)
    return (project_root / path).resolve() if not path.is_absolute() else path.resolve()


def load_dataset_config(config_path: Path | None, project_root: Path | None = None) -> dict[str, dict[str, Path]]:
    root = (project_root or PROJECT_ROOT).resolve()
    if config_path is None:
        return default_dataset_config(root)

    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    datasets: dict[str, dict[str, Path]] = {}
    if isinstance(payload, dict) and "datasets" in payload and isinstance(payload["datasets"], list):
        for item in payload["datasets"]:
            name = str(item["name"]).strip()
            datasets[name] = {
                "images_dir": _resolve_dir(str(item["images_dir"]), root),
                "labels_dir": _resolve_dir(str(item["labels_dir"]), root),
            }
    elif isinstance(payload, dict):
        for name, item in payload.items():
            datasets[str(name)] = {
                "images_dir": _resolve_dir(str(item["images_dir"]), root),
                "labels_dir": _resolve_dir(str(item["labels_dir"]), root),
            }
    else:
        raise ValueError("datasets config must be a dict or {'datasets': [...]} format")

    if not datasets:
        raise ValueError("No datasets found in datasets config")
    return datasets


def dataset_order_for(datasets: list[str]) -> list[str]:
    preferred = [d for d in DATASET_ORDER if d in datasets]
    rest = sorted([d for d in datasets if d not in preferred])
    return preferred + rest


def style_for_dataset(ds: str) -> tuple[str, str, float]:
    fill, edge = COLOR_OVERRIDES.get(ds, (DEFAULT_PALETTE[hash(ds) % len(DEFAULT_PALETTE)], None))
    lw = 0.9 if ds in OUTLINE_DATASETS and edge else 0.0
    edgecolor = edge if (edge and lw > 0) else fill
    return fill, edgecolor, lw
