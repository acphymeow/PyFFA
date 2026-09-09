# Fig. 8 Gaussian-kernel benchmark

This is the clean entry directory for the Referee A-2 longitudinal-field
benchmark.  It does not change the production longitudinal space-charge
kernel.

- `fig8_gaussian_kernel_benchmark.py` runs the validated full calculation and
  writes the original three-panel diagnostic figure to `outputs/`.
- `plot_fig8_gaussian_right_panel.py` generates the standalone paper figure
  corresponding to the rightmost diagnostic panel: `NS=256`, Gaussian
  source-slice averaging, and 8 cm Gaussian smoothing.
- `data/` contains the local three-column snapshot used by the lightweight
  paper plotter.
- `outputs/` contains generated PNG/PDF figures and the full benchmark report.

Run the paper figure:

```bash
python plot_fig8_gaussian_right_panel.py
```

Run the full three-panel benchmark (computationally expensive):

```bash
python fig8_gaussian_kernel_benchmark.py
```

The validated implementation is retained in `../01_fig8_es/` so existing
response-bundle paths are not broken.  The clean full-benchmark entry point
loads that implementation and redirects its outputs here.
