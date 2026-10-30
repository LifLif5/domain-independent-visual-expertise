# EyeTrackingAnalysis

Analysis code for the behavioural eye-tracking study (a visual-search task and a
multiple-object-tracking (MOT) task), packaged as a
[Code Ocean](https://codeocean.com) Compute Capsule.

## Layout

```
code/
  utils.py                      shared helpers (loading, filtering, plotting entry points)
  search_analysis/
    search_features.ipynb       per-trial/per-participant search features -> combined_zscore_search.csv
    performance_analysis_search.ipynb   accuracy / reaction-time analyses
  mot_analysis/
    article_mot_analysis.ipynb  MOT analyses + overall expertise index
    mot_helpers.py              trajectory loading, phases, target positions
    mot_config.yaml             phase/threshold configuration
  statistic_checks/
    GLMs.py, MixedEffectStatistics.py   OLS / mixed-effects models (statsmodels + pymer4/R lme4)
    article_plots.py, plot_utils.py, z_score.py
environment/
  Dockerfile                    full capsule image definition
  environment.yml               conda part (conda-forge, Python 3.10, R 4.4)
  requirements.txt              pip part (pymer4==0.9.2, ...)
run                             reproducible-run entry point (executes all notebooks -> /results)
CAPSULE_SETUP.md                step-by-step capsule/environment instructions
```

## Quick start

**Easiest – via GitHub.** Push this repository, then in Code Ocean use
*New Capsule → Clone from Git* with the repo URL. Code Ocean maps `code/` to
`/code`, builds `environment/Dockerfile`, and keeps the capsule linked to the
repo (two-way *Sync with GitHub*). Add the data via the capsule's **Data** panel,
then hit *Reproducible Run*. Full walkthrough: [`CAPSULE_SETUP.md`](CAPSULE_SETUP.md) §2.

**Inside Code Ocean without git** – see [`CAPSULE_SETUP.md`](CAPSULE_SETUP.md) §3.
The capsule uses the `environment/` files and runs `bash run`, which executes the
three notebooks in order and writes the executed notebooks + result tables to
`/results`.

**Locally**

```bash
conda env create -f environment/environment.yml
conda activate eyetracking
pip install -r environment/requirements.txt
bash run                 # writes ./results/*.ipynb and ./results/*.csv
```

or with Docker:

```bash
docker build -t eyetracking-env environment/
docker run --rm --platform linux/amd64 \
  -v "$PWD/code":/code -v "$PWD/data":/data -v "$PWD/results":/results \
  -w /code eyetracking-env bash run
```

## Data (not in this repository)

Raw and preprocessed data are **not** versioned here (see `.gitignore`; the
pickle alone is ~100 MB). Put these five files into `data/` (or upload them to
the capsule's `/data`):

| File | Used by |
|------|---------|
| `simple_df_search.csv` | `search_features.ipynb` |
| `search_performance_long.csv` | `search_features.ipynb`, `performance_analysis_search.ipynb` |
| `mot_long_dataframe.csv` | `article_mot_analysis.ipynb` |
| `simple_df_mot.csv` | `article_mot_analysis.ipynb` |
| `movement_phase_full.pkl` | `article_mot_analysis.ipynb` |

Notebooks resolve `CODE_DIR`/`DATA_DIR` from the folder layout (`code/` next to
`data/`, i.e. `/code` and `/data` in a capsule), so they run from any working
directory.

## Environment notes

The mixed-effects models use `pymer4` 0.9.2, which is a wrapper around R's
`lme4`: it requires R with `lme4`, `lmerTest`, `broom(.mixed)`, `emmeans` and the
easystats bundle, and it expects **polars** DataFrames (not pandas). All of this
is installed by `environment/environment.yml` + `environment/requirements.txt`.
See `CAPSULE_SETUP.md` §7 for the usual failure modes.
