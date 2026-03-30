from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

import config


@dataclass(frozen=True)
class BoxRecord:
    dataset: str
    image_path: str
    label_path: str
    class_id: int
    x: float
    y: float
    w: float
    h: float
    width: int
    height: int
    w_px: float
    h_px: float
    bbox_min_side_px: float
    area_frac: float
    center_distance_norm: float
    center_contains_point: bool
    center_window_overlap: float
    centrality_distance_centered: float
    centrality_overlap_center: float
    centrality_apparent: float
    sharpness_ten_abs: float
    sharpness_lap_var: float
    include_in_analysis: bool
    exclusion_reason: str


def ensure_dirs() -> None:
    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config.INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_NORM_DIR.mkdir(parents=True, exist_ok=True)
    config.EXAMPLE_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    config.SHARPNESS_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def to_rel_path(p: Path) -> str:
    try:
        return p.resolve().relative_to(config.PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return p.name


def iter_images(images_dir: Path) -> Iterable[Path]:
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS:
            yield p


def read_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not decode image: {path}")
    return img


def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    out: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return out
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            try:
                c = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            out.append(
                (
                    c,
                    float(np.clip(x, 0.0, 1.0)),
                    float(np.clip(y, 0.0, 1.0)),
                    float(np.clip(w, 0.0, 1.0)),
                    float(np.clip(h, 0.0, 1.0)),
                )
            )
    return out


def box_to_xyxy(x: float, y: float, w: float, h: float, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = int(round((x - w / 2.0) * width))
    y0 = int(round((y - h / 2.0) * height))
    x1 = int(round((x + w / 2.0) * width))
    y1 = int(round((y + h / 2.0) * height))
    x0 = max(0, min(x0, width))
    y0 = max(0, min(y0, height))
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    return x0, y0, x1, y1


def sharpness_metrics(crop_gray: np.ndarray) -> tuple[float, float]:
    roi = crop_gray.astype(np.float32)
    gx = cv2.Scharr(roi, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(roi, cv2.CV_32F, 0, 1)
    ten_abs = float((gx * gx + gy * gy).mean())
    lap = cv2.Laplacian(roi, cv2.CV_32F, ksize=3)
    lap_var = float(lap.var())
    return ten_abs, lap_var


def center_distance_norm(x: float, y: float) -> float:
    d = math.hypot(x - 0.5, y - 0.5)
    dmax = math.sqrt(0.5**2 + 0.5**2)
    return float(np.clip(d / (dmax + 1e-12), 0.0, 1.0))


def center_contains_point(x: float, y: float, w: float, h: float) -> bool:
    return abs(x - 0.5) <= (w / 2.0) and abs(y - 0.5) <= (h / 2.0)


def center_window_overlap(x: float, y: float, w: float, h: float, window_size: float) -> float:
    x0, y0, x1, y1 = x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0
    hw = window_size / 2.0
    wx0, wy0, wx1, wy1 = 0.5 - hw, 0.5 - hw, 0.5 + hw, 0.5 + hw
    ix0, iy0 = max(x0, wx0), max(y0, wy0)
    ix1, iy1 = min(x1, wx1), min(y1, wy1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    return float(np.clip(inter / max(window_size * window_size, 1e-12), 0.0, 1.0))


def overlap_score(contains_center: bool, overlap: float) -> float:
    c = 1.0 if contains_center else 0.0
    return float(np.clip(0.6 * c + 0.4 * math.sqrt(max(overlap, 0.0)), 0.0, 1.0))


def apparent_centrality(overlap: float, distance_centered: float) -> float:
    w_ov = float(config.CENTRALITY_HYBRID_WEIGHTS["overlap"])
    w_dist = float(config.CENTRALITY_HYBRID_WEIGHTS["distance"])
    return float(np.clip(w_ov * overlap + w_dist * distance_centered, 0.0, 1.0))


def process_dataset(dataset: str, roots: dict[str, Path], min_box_side_px: float) -> tuple[list[BoxRecord], dict[str, int]]:
    images_dir, labels_dir = roots["images_dir"], roots["labels_dir"]
    stats = {
        "dataset": dataset,
        "images_discovered": 0,
        "images_read_errors": 0,
        "boxes_parsed_total": 0,
        "boxes_included": 0,
        "boxes_excluded_tiny": 0,
        "boxes_excluded_invalid": 0,
        "boxes_excluded_read_error": 0,
    }
    rows: list[BoxRecord] = []
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found for '{dataset}': {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_dir not found for '{dataset}': {labels_dir}")

    for image_path in iter_images(images_dir):
        stats["images_discovered"] += 1
        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes = parse_yolo_label(label_path)
        stats["boxes_parsed_total"] += len(boxes)
        if not boxes:
            continue
        try:
            gray = read_gray(image_path)
            h_img, w_img = gray.shape[:2]
        except Exception:
            stats["images_read_errors"] += 1
            stats["boxes_excluded_read_error"] += len(boxes)
            for cls, x, y, w, h in boxes:
                rows.append(
                    BoxRecord(
                        dataset=dataset,
                        image_path=to_rel_path(image_path),
                        label_path=to_rel_path(label_path),
                        class_id=cls,
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        width=-1,
                        height=-1,
                        w_px=float("nan"),
                        h_px=float("nan"),
                        bbox_min_side_px=float("nan"),
                        area_frac=float(np.clip(w * h, 0.0, 1.0)),
                        center_distance_norm=center_distance_norm(x, y),
                        center_contains_point=False,
                        center_window_overlap=float("nan"),
                        centrality_distance_centered=float("nan"),
                        centrality_overlap_center=float("nan"),
                        centrality_apparent=float("nan"),
                        sharpness_ten_abs=float("nan"),
                        sharpness_lap_var=float("nan"),
                        include_in_analysis=False,
                        exclusion_reason="read_error",
                    )
                )
            continue
        for cls, x, y, w, h in boxes:
            x0, y0, x1, y1 = box_to_xyxy(x, y, w, h, w_img, h_img)
            bw, bh = float(max(0, x1 - x0)), float(max(0, y1 - y0))
            min_side = float(min(bw, bh))
            area_frac = float(np.clip(w * h, 0.0, 1.0))
            d_norm = center_distance_norm(x, y)
            d_centered = 1.0 - d_norm
            c_contains = center_contains_point(x, y, w, h)
            c_overlap = center_window_overlap(x, y, w, h, config.CENTRALITY_CENTER_WINDOW)
            c_overlap_score = overlap_score(c_contains, c_overlap)
            c_app = apparent_centrality(c_overlap_score, d_centered)

            include, reason = True, ""
            sharp_ten, sharp_lap = float("nan"), float("nan")
            if bw <= 0 or bh <= 0:
                include = False
                reason = "invalid_box"
                stats["boxes_excluded_invalid"] += 1
            elif min_side < min_box_side_px:
                include = False
                reason = "tiny_box"
                stats["boxes_excluded_tiny"] += 1
            else:
                crop = gray[y0:y1, x0:x1]
                if crop.size == 0:
                    include = False
                    reason = "invalid_box"
                    stats["boxes_excluded_invalid"] += 1
                else:
                    sharp_ten, sharp_lap = sharpness_metrics(crop)
                    stats["boxes_included"] += 1

            rows.append(
                BoxRecord(
                    dataset=dataset,
                    image_path=to_rel_path(image_path),
                    label_path=to_rel_path(label_path),
                    class_id=cls,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    width=w_img,
                    height=h_img,
                    w_px=bw,
                    h_px=bh,
                    bbox_min_side_px=min_side,
                    area_frac=area_frac,
                    center_distance_norm=d_norm,
                    center_contains_point=c_contains,
                    center_window_overlap=c_overlap,
                    centrality_distance_centered=d_centered,
                    centrality_overlap_center=c_overlap_score,
                    centrality_apparent=c_app,
                    sharpness_ten_abs=sharp_ten,
                    sharpness_lap_var=sharp_lap,
                    include_in_analysis=include,
                    exclusion_reason=reason,
                )
            )
    return rows, stats


def run(min_box_side_px: float, reuse_cache: bool, datasets_json: Path | None = None, project_root: Path | None = None) -> pd.DataFrame:
    ensure_dirs()
    per_box_csv = config.INTERMEDIATE_DIR / "per_box_metrics.csv"
    meta_json = config.INTERMEDIATE_DIR / "run_metadata.json"
    if reuse_cache and per_box_csv.exists():
        return pd.read_csv(per_box_csv)

    all_rows: list[BoxRecord] = []
    stats_rows: list[dict[str, int]] = []
    if project_root is not None:
        config.set_project_root(project_root)
    roots_all = config.load_dataset_config(datasets_json, config.PROJECT_ROOT)
    ds_order = config.dataset_order_for(list(roots_all.keys()))
    for ds in ds_order:
        rows, stats = process_dataset(ds, roots_all[ds], min_box_side_px)
        all_rows.extend(rows)
        stats_rows.append(stats)

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    if df.empty:
        raise RuntimeError("No annotation metrics created. Check dataset paths.")
    df.to_csv(per_box_csv, index=False)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "project_root_mode": "relative",
                "project_root": str(config.PROJECT_ROOT),
                "centrality_label": config.CENTRALITY_PRIMARY_LABEL,
                "min_box_side_px": min_box_side_px,
                "datasets": {ds: {"images_dir": str(roots_all[ds]["images_dir"]), "labels_dir": str(roots_all[ds]["labels_dir"])} for ds in ds_order},
                "stats": stats_rows,
            },
            f,
            indent=2,
        )
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute dataset metrics (sanitized paths).")
    p.add_argument("--min-box-side-px", type=float, default=config.MIN_BOX_SIDE_PX)
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--datasets-json", type=str, default=None, help="Path to datasets JSON config.")
    p.add_argument("--project-root", type=str, default=None, help="Project root path for resolving relative dataset paths.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets_json = Path(args.datasets_json).resolve() if args.datasets_json else None
    project_root = Path(args.project_root).resolve() if args.project_root else None
    df = run(args.min_box_side_px, args.reuse_cache, datasets_json=datasets_json, project_root=project_root)
    print(f"[done] wrote {config.INTERMEDIATE_DIR / 'per_box_metrics.csv'}")
    print(f"[done] included boxes: {int(df['include_in_analysis'].sum())} / {len(df)}")


if __name__ == "__main__":
    main()
