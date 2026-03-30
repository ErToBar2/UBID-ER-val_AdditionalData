# Expected Input (Short)

`_run_data_comparison.py` works for any dataset set if you provide:

1. **Images folder** (jpg/png/tif/webp).
2. **Labels folder** in **YOLO format** (`class x_center y_center width height`, normalized to `[0,1]`).
3. A datasets JSON file (see `datasets.example.json`).

## Run

```powershell
python _run_data_comparison.py --datasets-json datasets.example.json --project-root .
```

## Notes

- Each dataset entry must have: `name`, `images_dir`, `labels_dir`.
- Relative paths in JSON are resolved from `--project-root`.
- Tiny boxes below `--min-box-side-px` are excluded from analysis.
- Output is written to `output/`.
