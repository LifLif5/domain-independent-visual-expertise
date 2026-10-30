# syntax=docker/dockerfile:1
#
# ---------------------------------------------------------------------------
# EyeTrackingAnalysis – capsule environment
# ---------------------------------------------------------------------------
# Everything the three notebooks in /code need:
#   * Python 3.10 scientific stack (pandas / numpy / scipy / statsmodels /
#     scikit-learn / matplotlib / seaborn / shapely / polars)
#   * Jupyter + nbconvert (IDE kernels and the `run` driver script)
#   * R 4.4 + lme4 / lmerTest / broom(.mixed) / emmeans / easystats and rpy2
#   * pymer4 0.9.2 (mixed-effects LMM/GLMM front-end used by
#     code/statistic_checks/GLMs.py and MixedEffectStatistics.py)
#
# Two ways to use this file
# -------------------------
# 1. Code Ocean: create the capsule, open /environment/Dockerfile in the
#    Environment Editor, unlock it and keep the `FROM` line that Code Ocean
#    generated (it points to the capsule's Starter Environment image, e.g.
#    `FROM $REGISTRY_HOST/codeocean/mambaforge3:...`).  Replace everything
#    *after* that FROM line with the instructions below.
#    (This file's own FROM line is only there so it builds standalone.)
#
# 2. Anywhere else (Docker Desktop / podman):
#       docker build -t eyetracking-env environment/
#       docker run --rm -it -v "$PWD":/code -w /code eyetracking-env bash
# ---------------------------------------------------------------------------

FROM condaforge/miniforge3:24.11.3-0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# The build context is the `environment/` folder (same as on Code Ocean).
COPY environment.yml requirements.txt /tmp/env/

# 1. Conda part: Python + R + the R packages pymer4 shells out to.
#    Channels come from environment.yml (conda-forge only), so the Anaconda
#    `defaults` channel is never used.
RUN mamba env update --name base --file /tmp/env/environment.yml \
    && mamba clean --all --yes

# 2. pip part: pymer4 and its pure-Python dependencies.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/env/requirements.txt

# 3. Fail the build early if the environment is not actually usable.
RUN python -c "import pandas, numpy, scipy, statsmodels, sklearn, matplotlib, seaborn, shapely, yaml, polars, nbconvert, matplotlib_inline, rpy2; from pymer4.models import lmer, glmer; print('environment OK')"

# Code Ocean mounts the capsule's code at /code and starts there.
WORKDIR /code
