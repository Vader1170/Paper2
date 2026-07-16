# Paper 2: Beyond Quasi-Uniformity — Numerical Reproducibility

This repository contains the complete code to reproduce all numerical experiments in the paper *"Beyond Quasi-Uniformity: Balanced Meshes and Stable Polynomial Approximation under Irregular Sampling"*.

## Requirements
- Python 3.8+
- NumPy, SciPy, Matplotlib, Seaborn, Pandas, tqdm

Install with:
pip install -r requirements.txt

This will:
- Generate all mesh families.
- Compute Gram matrices and condition numbers.
- Run 1000‑trial noise experiments.
- Save numerical data to `data/`.
- Produce all figures in `figures/`.

## Notebooks
A Jupyter notebook (`notebooks/demo.ipynb`) is provided for interactive exploration.

## File Descriptions
- `mesh_generator.py` : Constructs uniform, perturbed, clustered, geometric, balanced geometric, and exponential‑failure meshes.
- `weights.py` : Computes trapezoidal (Voronoi) and uniform quadrature weights.
- `core.py` : Builds the Legendre‑based Gram matrix and computes condition numbers.
- `experiments.py` : Runs Experiments 1–6 sequentially.
- `plotting.py` : Generates Figures 1–8.
