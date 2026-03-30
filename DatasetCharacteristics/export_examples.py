from __future__ import annotations

import argparse
import os
from pathlib import Path
import uuid

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from compute_metrics import box_to_xyxy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export metric mean example collages.")
    p.add_argument(
        "--sharpness-mode",
        choices=["closest_mean", "near_mean_clear", "near_mean_blurry"],
        default="near_mean_clear",
    )
    return p.parse_args()


def imread_color(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not decode image: {path}")
    return img


def imwrite(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Could not encode: {path}")
    buf.tofile(str(path))


def _hex_to_bgr(h: str) -> tuple[int, int, int]:
    s = h.lstrip("#")
    return int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16)


def _fit_canvas(img: np.ndarray, h: int = 380, w: int = 520) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale = min(w / max(iw, 1), h / max(ih, 1))
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    can = np.full((h, w, 3), 20, dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    can[y0:y0 + nh, x0:x0 + nw] = rs
    return can


def _draw_base_overlay(img: np.ndarray, row: pd.Series, ds: str) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = box_to_xyxy(float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]), w, h)
    fill, edge, _ = config.style_for_dataset(ds)
    edge_bgr = _hex_to_bgr(edge if edge else fill)
    cv2.rectangle(out, (x0, y0), (x1, y1), edge_bgr, 2, lineType=cv2.LINE_AA)
    cx, cy = int(round(w * 0.5)), int(round(h * 0.5))
    bx, by = int(round(float(row["x"]) * w)), int(round(float(row["y"]) * h))
    cv2.drawMarker(out, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
    cv2.circle(out, (bx, by), 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    return out


def _draw_centrality_overlay(img: np.ndarray, row: pd.Series, ds: str) -> np.ndarray:
    out = _draw_base_overlay(img, row, ds)
    h, w = out.shape[:2]
    cx, cy = int(round(w * 0.5)), int(round(h * 0.5))
    bx, by = int(round(float(row["x"]) * w)), int(round(float(row["y"]) * h))
    cv2.line(out, (cx, cy), (bx, by), (255, 255, 255), 2, lineType=cv2.LINE_AA)
    win = max(12, int(round(min(w, h) * config.CENTRALITY_CENTER_WINDOW)))
    hh = win // 2
    cv2.rectangle(out, (cx - hh, cy - hh), (cx + hh, cy + hh), (255, 0, 255), 2, lineType=cv2.LINE_AA)
    return out


def _prepare_view(img: np.ndarray, row: pd.Series, ds: str, metric: str) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box_to_xyxy(float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]), w, h)
    if metric in {"bbox_min_side_px", "sharpness_ten_abs"}:
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            crop = img
        return _fit_canvas(crop)
    if metric == "area_frac":
        return _fit_canvas(_draw_base_overlay(img, row, ds))
    if metric == config.CENTRALITY_PRIMARY_COL:
        return _fit_canvas(_draw_centrality_overlay(img, row, ds))
    return _fit_canvas(img)


def _select_row(d: pd.DataFrame, metric: str, mode: str) -> tuple[pd.Series, float]:
    mean_val = float(d[metric].mean())
    t = d.copy()
    t["dist"] = (t[metric] - mean_val).abs()
    if metric != "sharpness_ten_abs":
        return t.nsmallest(1, "dist").iloc[0], mean_val
    near = t.nsmallest(max(10, int(round(0.2 * len(t)))), "dist")
    if mode == "near_mean_clear":
        return near.nlargest(1, "sharpness_ten_abs").iloc[0], mean_val
    if mode == "near_mean_blurry":
        return near.nsmallest(1, "sharpness_ten_abs").iloc[0], mean_val
    return near.nsmallest(1, "dist").iloc[0], mean_val


def _safe_image_path(rel_or_name: str) -> Path:
    p = Path(rel_or_name)
    return p if p.is_absolute() else (config.PROJECT_ROOT / p)


def _safe_save_figure(fig: plt.Figure, out_path: Path, **kwargs) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f".tmp_{uuid.uuid4().hex}{out_path.suffix}"
    fig.savefig(str(tmp), **kwargs)
    try:
        os.replace(str(tmp), str(out_path))
    except PermissionError:
        # If the target is locked (e.g., open in an image viewer), save with a fallback name.
        fallback = out_path.with_name(f"{out_path.stem}_new{out_path.suffix}")
        os.replace(str(tmp), str(fallback))
        print(f"[warn] target file was locked, wrote fallback file: {fallback}")


def main() -> None:
    args = parse_args()
    src = config.INTERMEDIATE_DIR / "per_box_metrics.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}. Run compute_metrics.py first.")

    df = pd.read_csv(src)
    df = df[df["include_in_analysis"]].copy()
    dataset_order = config.dataset_order_for([str(x) for x in df["dataset"].dropna().unique().tolist()])
    metrics = [
        ("bbox_min_side_px", "BBox shorter side"),
        ("area_frac", "Area fraction"),
        (config.CENTRALITY_PRIMARY_COL, config.CENTRALITY_PRIMARY_LABEL),
        ("sharpness_ten_abs", "Sharpness (Tenengrad index)"),
    ]

    sharp = pd.to_numeric(df["sharpness_ten_abs"], errors="coerce").dropna().to_numpy(float)
    s_lo, s_hi = (np.percentile(sharp, [1, 99]) if len(sharp) else (0.0, 1.0))
    if s_hi <= s_lo:
        s_hi = s_lo + 1.0

    chosen: dict[str, dict[str, pd.Series]] = {m[0]: {} for m in metrics}
    means: dict[str, dict[str, float]] = {m[0]: {} for m in metrics}
    for m, _ in metrics:
        for ds in dataset_order:
            d = df[df["dataset"] == ds].copy()
            if d.empty:
                continue
            row, mean_val = _select_row(d, m, args.sharpness_mode)
            chosen[m][ds] = row
            means[m][ds] = mean_val

    summary_rows = []

    def render(ranked: bool, stem: str, subtitle: str) -> None:
        fig, axes = plt.subplots(len(dataset_order), len(metrics), figsize=(4.8 * len(metrics), 3.35 * len(dataset_order)))
        if len(dataset_order) == 1:
            axes_local = np.expand_dims(axes, axis=0)
        else:
            axes_local = axes
        for j, (metric, mlabel) in enumerate(metrics):
            axes_local[0, j].set_title(mlabel, fontsize=12, color="#111111")
            ordered = sorted(means[metric].keys(), key=lambda dsn: means[metric][dsn], reverse=True) if ranked else [d for d in dataset_order if d in means[metric]]
            for i, ds in enumerate(ordered):
                row = chosen[metric][ds]
                imgp = _safe_image_path(str(row["image_path"]))
                color = imread_color(imgp)
                view = _prepare_view(color, row, ds, metric)
                ax = axes_local[i, j]
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax_in = ax.inset_axes([0.02, 0.17, 0.96, 0.70])
                ax_in.imshow(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
                ax_in.set_xticks([]); ax_in.set_yticks([])
                for sp in ax_in.spines.values():
                    sp.set_visible(False)

                ds_color = config.COLOR_OVERRIDES.get(ds, ("#7D8A94", None))[0]
                ax.text(0.5, 0.985, ds, color=ds_color, fontsize=10.2, ha="center", va="top", transform=ax.transAxes, fontweight="bold", clip_on=False)

                mean_val = float(means[metric][ds])
                if metric == "sharpness_ten_abs":
                    mean_idx = float(np.clip((np.clip(mean_val, s_lo, s_hi) - s_lo) / (s_hi - s_lo), 0.0, 1.0) * 100.0)
                    mean_txt = f"mean sharpness index={mean_idx:.1f}/100"
                else:
                    mean_txt = f"mean {mlabel.lower()}={mean_val:.4g}"
                ax.text(0.5, 0.91, mean_txt, color="#111111", fontsize=8.4, ha="center", va="top", transform=ax.transAxes, clip_on=False)

                h, w = color.shape[:2]
                x0, y0, x1, y1 = box_to_xyxy(float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]), w, h)
                bw, bh = max(1, x1 - x0), max(1, y1 - y0)
                if metric in {"bbox_min_side_px", "sharpness_ten_abs"}:
                    extra = f"box size={bw}x{bh} px"
                elif metric == "area_frac":
                    extra = f"box size={bw}x{bh} px"
                    ax.text(-0.03, 0.50, f"h={bh}px", rotation=90, color="#f0f0f0", fontsize=8.5, ha="right", va="center", transform=ax.transAxes)
                    ax.text(0.50, -0.05, f"w={bw}px", color="#f0f0f0", fontsize=8.5, ha="center", va="top", transform=ax.transAxes)
                elif metric == config.CENTRALITY_PRIMARY_COL:
                    c = float(row[config.CENTRALITY_PRIMARY_COL]); ov = float(row["center_window_overlap"]); dn = float(row["center_distance_norm"])
                    extra = f"score={c:.3f}; overlap={ov:.3f}; d={dn:.3f}"
                else:
                    extra = ""
                if metric == "sharpness_ten_abs":
                    raw = float(row["sharpness_ten_abs"])
                    idx = float(np.clip((np.clip(raw, s_lo, s_hi) - s_lo) / (s_hi - s_lo), 0.0, 1.0) * 100.0)
                    extra = f"example index={idx:.1f}/100 (higher=sharper)"
                ax.text(0.5, 0.06, extra, color="#111111", fontsize=7.6, ha="center", va="bottom", transform=ax.transAxes, clip_on=False)

                crop = cv2.cvtColor(color[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                crop_path = ""
                if crop.size > 0 and metric == "sharpness_ten_abs":
                    outp = config.SHARPNESS_EXAMPLES_DIR / f"{ds}_sharpness_near_mean.png".replace("/", "_")
                    imwrite(outp, crop)
                    crop_path = str(outp.relative_to(config.MODULE_ROOT).as_posix())

                summary_rows.append(
                    {
                        "layout": "ranked" if ranked else "by_dataset",
                        "rank_in_metric": (i + 1) if ranked else np.nan,
                        "dataset": ds,
                        "metric": "centrality" if metric == config.CENTRALITY_PRIMARY_COL else metric,
                        "metric_label": mlabel,
                        "dataset_mean": mean_val,
                        "example_value": float(row[metric]),
                        "example_value_readable": f"example value={float(row[metric]):.4g}",
                        "sharpness_mode": args.sharpness_mode,
                        "box_w_px": int(bw),
                        "box_h_px": int(bh),
                        "image_path": str(row["image_path"]),
                        "sharpness_crop_path": crop_path,
                        "center_window_overlap": float(row["center_window_overlap"]),
                        "center_distance_norm": float(row["center_distance_norm"]),
                        "centrality_apparent": float(row["centrality_apparent"]),
                    }
                )

        fig.suptitle(subtitle, fontsize=13)
        fig.subplots_adjust(left=0.02, right=0.995, top=0.92, bottom=0.03, wspace=0.02, hspace=0.10)
        base = config.EXAMPLE_PANEL_DIR / stem
        base.parent.mkdir(parents=True, exist_ok=True)
        _safe_save_figure(fig, base.with_suffix(".png"), dpi=500, bbox_inches="tight")
        _safe_save_figure(fig, base.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)

    render(True, "metric_mean_examples_grid_ranked", "Metric examples near dataset means (ranked per metric)")
    render(False, "metric_mean_examples_grid_by_dataset", "Metric examples near dataset means (dataset rows)")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.SHARPNESS_EXAMPLES_DIR / "sharpness_mean_example_summary.csv", index=False)
    summary.to_csv(config.EXAMPLE_PANEL_DIR / "metric_mean_examples_summary.csv", index=False)
    print(f"[done] wrote {config.EXAMPLE_PANEL_DIR / 'metric_mean_examples_grid_ranked.png'}")
    print(f"[done] wrote {config.EXAMPLE_PANEL_DIR / 'metric_mean_examples_grid_by_dataset.png'}")


if __name__ == "__main__":
    main()
