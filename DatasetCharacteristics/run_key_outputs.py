from __future__ import annotations

import argparse
import sys

import compute_metrics
import config
import export_examples
import plot_panel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate key outputs for GitHub package.")
    p.add_argument("--min-box-side-px", type=float, default=12.0)
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--datasets-json", type=str, default=None, help="Path to datasets JSON config.")
    p.add_argument("--project-root", type=str, default=None, help="Project root path.")
    p.add_argument(
        "--sharpness-mode",
        choices=["closest_mean", "near_mean_clear", "near_mean_blurry"],
        default="near_mean_clear",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from pathlib import Path

    datasets_json = Path(args.datasets_json).resolve() if args.datasets_json else None
    project_root = Path(args.project_root).resolve() if args.project_root else None
    compute_metrics.run(
        min_box_side_px=args.min_box_side_px,
        reuse_cache=args.reuse_cache,
        datasets_json=datasets_json,
        project_root=project_root,
    )
    plot_panel.main()

    old = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], "--sharpness-mode", args.sharpness_mode]
        export_examples.main()
    finally:
        sys.argv = old
    print("[done] GitHub package key outputs regenerated:")
    print(f" - {config.FIGURES_NORM_DIR / 'normalized_comparison_violinpanel.png'}")
    print(f" - {config.FIGURES_NORM_DIR / 'normalized_comparison_violinpanel.svg'}")
    print(f" - {config.EXAMPLE_PANEL_DIR / 'metric_mean_examples_grid_ranked.png'}")
    print(f" - {config.EXAMPLE_PANEL_DIR / 'metric_mean_examples_grid_ranked.svg'}")
    print(f" - {config.EXAMPLE_PANEL_DIR / 'metric_mean_examples_grid_by_dataset.png'}")
    print(f" - {config.EXAMPLE_PANEL_DIR / 'metric_mean_examples_grid_by_dataset.svg'}")


if __name__ == "__main__":
    main()
