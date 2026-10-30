# Reviewer guide — running this analysis without Code Ocean

This repository is self-contained: **no Code Ocean account, no proprietary
service and no GPU are needed** to reproduce every figure and table of the
manuscript. The analysis runs on a normal laptop/server with conda (or Docker)
in well under an hour.

* Code: this repository / the archived release (Zenodo DOI, see `CITATION.cff`).
* Data: deposited separately (see the Data availability statement of the
  manuscript); the DOI is also in `CITATION.cff` and `.zenodo.json`.
* Environment: fully pinned in `environment/` (Python 3.10 + R 4.4 via
  conda-forge, `pymer4` 0.9.2, all R packages for the mixed models).

---

## 1. What you need

| | |
|---|---|
| Operating system | Linux, macOS (both validated) or Windows with WSL2 |
| Disk | ≈5 GB for the environment, 120 MB for the data, a few MB for results |
| Memory | 8 GB is enough (16 GB comfortable); no GPU |
| Software | [Miniconda](https://docs.conda.io/en/latest/miniconda.html) **or** Docker |

## 2. Get the data

Download the data record and place the five files in a folder `data/` that is a
**sibling** of this repository's `code/` folder:

```
EyeTrackingAnalysis/
├── code/     <- from this repository
├── data/     <- the five files below
└── run
```

| File | Size | Content |
|------|------|---------|
| `simple_df_search.csv` | 16 MB | visual-search trials (gaze samples, features) |
| `search_performance_long.csv` | 60 KB | accuracy / reaction time per participant × condition |
| `simple_df_mot.csv` | 5.4 MB | MOT gaze samples with per-trial metadata |
| `mot_long_dataframe.csv` | 33 KB | MOT accuracy / tracking performance per participant × condition |
| `movement_phase_full.pkl` | 100 MB | raw MOT gaze trajectories (pandas pickle; 932 trials, columns `participant_id, group, trial_id, xpos, ypos`) |

Nothing else is required: the notebooks locate `code/` and `data/` from this
folder layout (first cell of each notebook), so they run from any working
directory.

## 3. Build the environment

**Option A — conda/mamba (recommended, this is what we validated)**

```bash
conda env create -f environment/environment.yml     # creates the env "eyetracking"
conda activate eyetracking
pip install -r environment/requirements.txt         # pymer4==0.9.2 and pure-python deps
```

Takes ~10 min; the R packages (`lme4`, `lmerTest`, `broom.mixed`, `emmeans`,
easystats) are the slow part. If conda complains about Terms of Service for
`repo.anaconda.com`, restrict it to conda-forge:

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

**Option B — Docker**

```bash
docker build -t eyetracking-env environment/
docker run --rm -v "$PWD/code":/code -v "$PWD/data":/data -v "$PWD/results":/results \
  -w /code eyetracking-env bash run
```

## 4. Run everything

```bash
bash run                      # writes to /results if it exists, else ./results
RESULTS_DIR=/tmp/out bash run # or choose the output folder
```

`run` executes the three notebooks in dependency order with
`jupyter nbconvert --execute` and writes the executed copies, with every figure
and table inline, to the results folder:

```
results/search_features_executed.ipynb
results/performance_analysis_search_executed.ipynb
results/article_mot_analysis_executed.ipynb
results/combined_zscore_search.csv
```

It exits non-zero and prints `!!! <notebook> FAILED` if a notebook raises.
**A full run takes ≈ 25 min on a 6-core 2019 MacBook Pro (CPU only)**; the
mixed-effects models in the first and third notebooks dominate. Keeping the
matplotlib *inline* backend matters — forcing `MPLBACKEND=Agg` silently drops
all figures from the executed notebooks.

**Interactive alternative** — run the notebooks in this order (the third reads
the table produced by the first):

1. `code/search_analysis/search_features.ipynb` → `combined_zscore_search.csv`
2. `code/search_analysis/performance_analysis_search.ipynb`
3. `code/mot_analysis/article_mot_analysis.ipynb`

| Notebook | Produces |
|----------|----------|
| `search_features.ipynb` | per-trial/per-participant search features (distractor and scan-path measures), 7 figures, the combined z-score table |
| `performance_analysis_search.ipynb` | accuracy and reaction-time LMMs for the visual-search task |
| `article_mot_analysis.ipynb` | MOT group comparisons, tracking-error thresholds, scan-path metrics, expertise index, 14 figures |

## 5. Check that you reproduced our numbers

Every value below is printed by the notebooks themselves, so you can compare
directly with your own run (all of them are deterministic given the deposited
data and the pinned environment):

| Printed by | Value |
|---|---|
| `article_mot_analysis`, threshold cell | tracking threshold (Lm) ATCs `0.03046`, Novices `0.03126`, combined `0.03086` |
| `article_mot_analysis`, RMS distance cell | RMS distance from centroid ATCs `503.3 px → 5.39°`, Novices `679.2 px → 7.27°` |
| `article_mot_analysis`, 500 ms scan-path cell | ATCs `789.1 px → 8.45°`, Novices `1447.2 px → 15.49°` (`45.47 %` difference) |
| `article_mot_analysis`, logistic model | `group_coef -0.4533`, `p = 0.0333`, marginal R² `0.3902`, conditional R² `0.5580` |
| `search_features`, fixation-duration LMM | `group_coef 34.9455`, `p = 0.00867`, marginal R² `0.3895`, conditional R² `0.6508` |
| `search_features`, distractor measure | `group_coef 15.8379`, `p = 0.0229` |

Small differences in the printed `p`-values or R² are not expected; if you see
them, check that the five data files are the deposited versions and that
`pymer4==0.9.2` is installed (`pip show pymer4`).

## 6. Layout

```
code/
  utils.py                      shared loading / filtering helpers
  search_analysis/              visual-search notebooks (+ combined_zscore_search.csv)
  mot_analysis/                 MOT notebook, mot_helpers.py, mot_config.yaml
  statistic_checks/             GLM / LMM wrappers (statsmodels + pymer4), plotting, z-scores
environment/                    Dockerfile, environment.yml, requirements.txt
run                             entry point (executes all notebooks)
CAPSULE_SETUP.md                environment details, Code Ocean notes, troubleshooting
SHARING.md                      how the code and data are archived and cited
```

## 7. Troubleshooting

* `ModuleNotFoundError: pymer4` / `rpy2` → the environment was not activated, or
  R is missing; reinstall from `environment/environment.yml`.
* `AttributeError: module 'pymer4.models' has no attribute 'Lmer'` → a pymer4
  < 0.9 version was installed; `pip install "pymer4==0.9.2"`.
* conda refuses to run (`Terms of Service … repo.anaconda.com`) → use the
  `--override-channels -c conda-forge` command in §3.
* `FileNotFoundError: …/data/…` → the data folder is not a sibling of `code/`.
* `FileNotFoundError: combined_zscore_search.csv` → run the notebooks in the
  order given in §4 (the table is produced by the first notebook).

More detail (including the Code Ocean variant) is in `CAPSULE_SETUP.md` §7.

## 8. License and citation

MIT license (`LICENSE`) — reuse is permitted with attribution. Please cite the
archived release (DOI in `CITATION.cff`) and the manuscript.
