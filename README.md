# Domain-Independent Visual Expertise

Analysis code for the behavioural eye-tracking study (a visual-search task and a
multiple-object-tracking, MOT, task).

This repository contains everything needed to reproduce the figures and tables
of the manuscript: the analysis notebooks, the helper modules, the pinned
compute environment, and a single `run` script that executes all notebooks and
writes the results.

## Contents

```text
code/
  utils.py                     shared loading/filtering helpers
  search_analysis/             visual-search feature extraction + performance analyses
  mot_analysis/                multiple-object-tracking analyses + expertise index
  statistic_checks/            GLM / mixed-effects models, plotting, z-scores
environment/
  environment.yml              conda environment (Python 3.10 + R 4.4)
  requirements.txt             pip packages (pymer4 etc.)
  Dockerfile                   optional container image
data/                          input data (see "Data" below)
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

The five input files are stored in `data/`. The large MOT trajectory file is
kept gzip-compressed (`movement_phase_full.pkl.gz`, ~10 MB) so it fits in git.

| File | Used by |
|------|---------|
| `simple_df_search.csv` | `search_analysis/search_features.ipynb` |
| `search_performance_long.csv` | `search_features.ipynb`, `performance_analysis_search.ipynb` |
| `mot_long_dataframe.csv` | `mot_analysis/article_mot_analysis.ipynb` |
| `simple_df_mot.csv` | `article_mot_analysis.ipynb` |
| `movement_phase_full.pkl.gz` | `article_mot_analysis.ipynb` |

The notebooks locate `code/` and `data/` from this folder layout, so they run
from any working directory.

## Run

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

All outputs are deterministic given the deposited data and the pinned
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

## Sharing and citation

To share the code with reviewers and archive it:

1. Push this repository to GitHub (private during review).
2. Deposit the data separately (e.g. Zenodo or figshare) and note its DOI.
3. Create a GitHub release `v1.0.0` with the Zenodo integration switched on to
   mint a citable DOI for the code.

Example statements for the manuscript:

> **Code availability.** The analysis code is available at
> https://doi.org/<code DOI> (archived release of
> https://github.com/<user>/domain-independent-visual-expertise).
>
> **Data availability.** The processed data have been deposited in
> <repository> at https://doi.org/<data DOI>; raw recordings are available from
> the corresponding author on reasonable request.

When the repository is made public, add an open-source `LICENSE` file (e.g.
MIT) if the journal requires one.
