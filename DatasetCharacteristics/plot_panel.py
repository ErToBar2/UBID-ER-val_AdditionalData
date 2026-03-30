from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


METRICS = [
    {"col": "bbox_min_side_px", "label": "BBox shorter side", "unit": "px", "q": (1, 99), "mode": "absolute"},
    {"col": "area_frac", "label": "Area fraction", "unit": "ratio", "q": (1, 99), "mode": "absolute"},
    {"col": config.CENTRALITY_PRIMARY_COL, "label": config.CENTRALITY_PRIMARY_LABEL, "unit": "score", "q": (1, 99), "mode": "absolute"},
    {"col": "sharpness_ten_abs", "label": "Sharpness", "unit": "index (0 blurry - 100 sharp)", "q": (1, 99), "mode": "sharpness_index"},
]


def _robust_bounds(values: np.ndarray, qlo: float, qhi: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [qlo, qhi])
    if hi <= lo:
        hi = np.nextafter(lo, lo + 1.0)
    return float(lo), float(hi)


def _load_included() -> pd.DataFrame:
    p = config.INTERMEDIATE_DIR / "per_box_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run compute_metrics.py first.")
    df = pd.read_csv(p)
    return df[df["include_in_analysis"]].copy()


def _dataset_order(df: pd.DataFrame) -> list[str]:
    present = [str(x) for x in df["dataset"].dropna().unique().tolist()]
    return config.dataset_order_for(present)


def transform_for_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    out = df.copy()
    bounds: dict[str, dict[str, float]] = {}
    for m in METRICS:
        col, q = m["col"], m["q"]
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(float)
        lo, hi = _robust_bounds(vals, q[0], q[1])
        clipped = np.clip(vals, lo, hi)
        if m["mode"] == "sharpness_index":
            out[f"{col}_disp"] = ((clipped - lo) / (hi - lo)) * 100.0
        else:
            out[f"{col}_disp"] = clipped
        bounds[col] = {"qlo": q[0], "qhi": q[1], "lo": lo, "hi": hi}
    return out, bounds


def _arrays(df: pd.DataFrame, col: str, dataset_order: list[str]) -> list[np.ndarray]:
    arrs = []
    for ds in dataset_order:
        v = pd.to_numeric(df.loc[df["dataset"] == ds, col], errors="coerce").dropna().to_numpy(float)
        arrs.append(v)
    return arrs


def _save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")


def make_panel(df_disp: pd.DataFrame) -> None:
    dataset_order = _dataset_order(df_disp)
    plt.rcParams.update({"font.family": ["Times New Roman", "serif"], "font.size": 8})
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    pos = np.arange(1, len(dataset_order) + 1)

    for ax, m in zip(axes, METRICS):
        col = f"{m['col']}_disp"
        data = _arrays(df_disp, col, dataset_order)
        parts = ax.violinplot(data, positions=pos, showmeans=True, showmedians=True, widths=0.8)
        for i, body in enumerate(parts["bodies"]):
            ds = dataset_order[i]
            fill, edge, lw = config.style_for_dataset(ds)
            body.set_facecolor(fill)
            body.set_edgecolor(edge)
            body.set_linewidth(lw)
            body.set_alpha(0.5)
        parts["cmeans"].set_color("#222222")
        parts["cmedians"].set_color("#111111")

        allv = np.concatenate([x for x in data if len(x) > 0]) if data else np.array([])
        if len(allv):
            lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
            if m["mode"] == "sharpness_index":
                lo, hi = max(0.0, lo), min(100.0, hi)
            if hi <= lo:
                hi = lo + 1.0
            ax.set_ylim(lo, hi)
        ax.set_title("")
        ax.set_xticks([])
        ax.set_xlabel(f"{m['label']} [{m['unit']}]", fontsize=8, labelpad=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.2)

    # Legend with mean values (black text + colored square).
    lines = []
    for ds in dataset_order:
        d = df_disp[df_disp["dataset"] == ds]
        ms = float(pd.to_numeric(d["bbox_min_side_px_disp"], errors="coerce").mean())
        ma = float(pd.to_numeric(d["area_frac_disp"], errors="coerce").mean())
        mc = float(pd.to_numeric(d[f"{config.CENTRALITY_PRIMARY_COL}_disp"], errors="coerce").mean())
        mh = float(pd.to_numeric(d["sharpness_ten_abs_disp"], errors="coerce").mean())
        fill, _, _ = config.style_for_dataset(ds)
        lines.append((ds, fill, ms, ma, mc, mh))

    y0, dy = -0.30, 0.075
    for i, (ds, color, ms, ma, mc, mh) in enumerate(lines):
        y = y0 - i * dy
        axes[0].text(-0.02, y, "■", transform=axes[0].transAxes, ha="left", va="top", fontsize=8, color=color, clip_on=False)
        axes[0].text(
            0.02,
            y,
            f"{ds}: mean size={ms:.2f}, mean area={ma:.4f}, mean centrality={mc:.3f}, mean sharpness={mh:.1f}",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#111111",
            clip_on=False,
        )

    fig.subplots_adjust(bottom=0.30, left=0.07, right=0.99, top=0.86, wspace=0.22)
    _save(fig, config.FIGURES_NORM_DIR / "normalized_comparison_violinpanel")
    plt.close(fig)


def main() -> None:
    df = _load_included()
    df_disp, bounds = transform_for_panel(df)
    df_disp.to_csv(config.INTERMEDIATE_DIR / "panel_values.csv", index=False)
    with open(config.INTERMEDIATE_DIR / "panel_bounds.json", "w", encoding="utf-8") as f:
        json.dump(bounds, f, indent=2)
    make_panel(df_disp)
    print(f"[done] wrote {config.FIGURES_NORM_DIR / 'normalized_comparison_violinpanel.png'}")


if __name__ == "__main__":
    main()
