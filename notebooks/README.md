# Notebook guide

Install from the repository root with `python -m pip install -e ".[notebook]"`, then launch `python -m jupyterlab`. Choose the installed environment as the kernel. Run cells in order. All examples import the existing package; the introduction includes optional interactive controls. Defaults are small illustrative experiments, not research-grade estimates. Keep `workers=1` inside notebooks for portable experiments.

Run all notebook smoke tests from the repository root with `python notebooks/check_notebooks.py`.
