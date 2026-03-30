# Dataset Evaluation and Figure Creation (GitHub-safe)

This folder is a sanitized copy of the dataset-evaluation/figure-creation code.

## Privacy and path safety

- Uses relative project-root resolution (`config.PROJECT_ROOT`) instead of machine-specific absolute paths.
- Stores `image_path` and `label_path` in outputs as paths relative to project root.
- Keeps output files inside this folder (`output/`).

## Scripts

- `compute_metrics.py`  
  Computes per-annotation metrics and writes `output/intermediate/per_box_metrics.csv`.

- `plot_panel.py`  
  Creates `normalized_comparison_violinpanel.png/.svg` (absolute-scale panel with legend).

- `export_examples.py`  
  Creates:
  - `output/metric_examples/metric_mean_examples_grid_ranked.png/.svg`
  - `output/metric_examples/metric_mean_examples_grid_by_dataset.png/.svg`
  - summary CSVs and sharpness crop examples

- `_run_data_comparison.py`  
  Single one-command runner for all key outputs.

## Run

```powershell
python _run_data_comparison.py --reuse-cache --sharpness-mode near_mean_clear
```

For custom/new datasets:

```powershell
python _run_data_comparison.py --datasets-json datasets.example.json --project-root .
```

The run exports these main figures:
- `output/figures_normalized/normalized_comparison_violinpanel.png`
- `output/figures_normalized/normalized_comparison_violinpanel.svg`
- `output/metric_examples/metric_mean_examples_grid_ranked.png`
- `output/metric_examples/metric_mean_examples_grid_ranked.svg`
- `output/metric_examples/metric_mean_examples_grid_by_dataset.png`
- `output/metric_examples/metric_mean_examples_grid_by_dataset.svg`

Sharpness options:
- `closest_mean`
- `near_mean_clear`
- `near_mean_blurry`

## Dependencies

Install:

```powershell
pip install pandas numpy matplotlib opencv-python
```

See `EXPECTED_INPUT.md` for the required dataset format.
