# Domain-Independent Visual Expertise

Analysis code and data for the behavioural eye-tracking study
"Domain-Independent Visual Expertise" (a visual-search task and a
multiple-object-tracking, MOT, task).

This repository contains the analysis notebooks, helper modules, input data and
a pinned compute environment, together with a single `run` script that executes
the notebooks and writes all result tables and figures.

## Contents

```text
code/
  utils.py                     shared loading/filtering helpers
  search_analysis/             visual-search feature extraction + performance analyses
  mot_analysis/                multiple-object-tracking analyses + expertise index
  statistic_checks/            GLM / mixed-effects models, plotting, z-scores
data/                          input data (see "Data" below)
environment/
  environment.yml              conda environment (Python 3.10 + R 4.4)
  requirements.txt             pip packages (pymer4 etc.)
  Dockerfile                   optional container image
run                            entry point: executes all notebooks -> results/
```

## Requirements

- Linux or macOS (or Windows with WSL2). CPU only — no GPU needed.
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) **or** Docker.

## Setup

Conda (recommended):

```bash
conda env create -f environment/environment.yml   # creates env "eyetracking"
conda activate eyetracking
pip install -r environment/requirements.txt
```

If conda complains about the Anaconda `defaults` channel, restrict it to
conda-forge:

```bash
conda create -n eyetracking --override-channels -c conda-forge \
  python=3.10 "numpy>=1.26,<3" pandas=2.2 "scipy>=1.11" "statsmodels>=0.14" \
  "scikit-learn>=1.3" "matplotlib>=3.7" "seaborn>=0.13" "shapely>=2.0" \
  "pyyaml>=6" "polars>=1.29,<2" "jupyterlab>=3" "notebook>=6" \
  "nbconvert>=7" "nbclient>=0.7" "nbformat>=5" "ipykernel>=6" matplotlib-inline \
  r-base=4.4 "rpy2>=3.5.11,<4" r-lme4 r-lmertest r-broom r-broom.mixed \
  r-emmeans r-boot r-tibble r-optimx r-insight r-parameters r-performance r-report
conda activate eyetracking && pip install -r environment/requirements.txt
```

Sanity check (should print `ok` twice):

```bash
python -c "from pymer4.models import lmer, glmer; import rpy2, polars; print('ok')"
Rscript -e 'library(lme4); library(lmerTest); library(broom.mixed); cat("ok\n")'
```

Docker:

```bash
docker build -t eyetracking-env environment/
```

## Data

The input data are in `data/`:

| File | Description |
|------|-------------|
| `simple_df_search.csv` | visual-search trials: gaze samples and derived features |
| `search_performance_long.csv` | accuracy and reaction time per participant × condition |
| `simple_df_mot.csv` | MOT gaze samples with per-trial metadata |
| `mot_long_dataframe.csv` | MOT accuracy / tracking performance per participant × condition |
| `movement_phase_full.pkl.gz` | raw MOT gaze trajectories (gzip-compressed pandas pickle; 932 trials) |

## Running the analysis

```bash
bash run
```

This executes the three notebooks in order and writes the executed notebooks
(with all figures) and result tables to `results/`:

1. `code/search_analysis/search_features.ipynb` — search features + `combined_zscore_search.csv`
2. `code/search_analysis/performance_analysis_search.ipynb` — accuracy / reaction-time models
3. `code/mot_analysis/article_mot_analysis.ipynb` — MOT analyses + expertise index

With Docker:

```bash
docker run --rm -v "$PWD/code":/code -v "$PWD/data":/data \
  -v "$PWD/results":/results -w /code eyetracking-env bash run
```

The notebooks can also be run interactively in Jupyter, in the order above (the
third reads the table produced by the first).

## Expected results

All outputs are deterministic given the included data and the pinned
environment. A few numbers printed by the notebooks to spot-check:

| Printed by | Value |
|---|---|
| `article_mot_analysis`, logistic model | `group_coef -0.4533`, `p = 0.0333` |
| `search_features`, fixation-duration LMM | `group_coef 34.9455`, `p = 0.00867` |
| `search_features`, distractor measure | `group_coef 15.8379`, `p = 0.0229` |

## Troubleshooting

- `AttributeError: ... 'pymer4.models' has no attribute 'Lmer'` → install `pymer4==0.9.2`.
- `RRuntimeError: could not find function "lmer"` → install `environment.yml` too (it provides R and the R packages).
- Do not force `MPLBACKEND=Agg` when running `run`; it silently drops the figures.
- `FileNotFoundError: .../data/...` → the `data/` folder is not a sibling of `code/`.
- Run the notebooks in the order given above.

## Citation

If you use this code or data, please cite the associated manuscript.
