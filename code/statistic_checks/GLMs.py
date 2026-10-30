import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pymer4.models import glmer
import polars as pl




import matplotlib.pyplot as plt
import numpy as np
from pymer4.models import lmer
def run_pymer_lmm(df: pd.DataFrame, value_col: str, weight_col: str = None,
                     participant_col: str = "participant_id",
                     group_col: str = "group", condition_col: str = "condition",
                     interaction_term: str = "+"):

    formula = f"{value_col} ~ {group_col} {interaction_term} {condition_col} + (1 | {participant_col})"

    if weight_col is not None:
        model = lmer(formula, data=pl.DataFrame(df), weights=weight_col)
    else:
        model = lmer(formula, data=pl.DataFrame(df))

    model.fit(summary=False)

    results = {
        "group_coef": float('nan'),
        "group_p_value": float('nan'),
        "marginal_r2": float('nan'),
        "conditional_r2": float('nan')
    }

    try:
        coefs_df = model.summary()._tbl_data

        # --- Group coefficient and p-value ---
        group_row = coefs_df.filter(
            pl.col('param').is_not_null() & pl.col('param').str.contains(group_col)
        )
        if len(group_row) > 0:
            results["group_coef"] = float(group_row['estimate'][0])
            raw_p = group_row['p_value'][0]
            results["group_p_value"] = 0.001 if (isinstance(raw_p, str) and '<' in raw_p) else float(raw_p)

        # --- R-squared ---
        results["marginal_r2"], results["conditional_r2"] = _compute_r2(model, coefs_df, participant_col, weight_col)

    except Exception as e:
        print(f"Note: Extraction error ({e})")

    return model, results


def _compute_r2(model, coefs_df, participant_col: str, weight_col: str = None):
    """Compute marginal and conditional R² (Nakagawa), weighted if weight_col is provided."""
    rfx_row   = coefs_df.filter(pl.col('rfx').is_not_null() & pl.col('rfx').str.contains(participant_col))
    resid_row = coefs_df.filter(pl.col('rfx').is_not_null() & pl.col('rfx').str.contains("Residual"))

    if len(rfx_row) == 0 or len(resid_row) == 0:
        return float('nan'), float('nan')

    var_re    = float(rfx_row['estimate'][0]) ** 2
    var_resid = float(resid_row['estimate'][0]) ** 2

    fixed_predictions = model.data["fitted"].to_numpy() - model.data["resid"].to_numpy()

    if weight_col is not None:
        weights = model.data[weight_col].to_numpy()
        weights = weights / weights.sum()
        wmean   = np.average(fixed_predictions, weights=weights)
        var_fixed = max(0, np.average((fixed_predictions - wmean) ** 2, weights=weights) - var_re)
    else:
        var_fixed = max(0, np.var(fixed_predictions) - var_re)

    total_var = var_fixed + var_re + var_resid
    return var_fixed / total_var, (var_fixed + var_re) / total_var

from typing import Callable
import numpy as np
def weighted_condition_aggregator(
    df: pd.DataFrame, 
    value_col: str,
    participant_col: str = "participant_id", 
    group_col: str = "group", 
    condition_col: str = "condition", 
    agg_func: Callable = np.mean,
    weight_method: str = "count"
) -> pd.DataFrame:
    """
    Aggregates raw data and calculates precision weights ('count' or 'inverse_variance')
    for each combination of participant, group, and condition.
    """
    if weight_method not in ["count", "inverse_variance",'precision', "log_count", "log_precision"]:
        raise ValueError("weight_method must be either 'count' or 'inverse_variance' or precision")

    group_cols = [participant_col, group_col, condition_col]

    # 1. Aggregate the main value
    func_name = getattr(agg_func, '__name__', 'agg')
    new_value_col = f'{func_name}_{value_col}'
    
    agg_df = df.groupby(group_cols)[value_col].agg(agg_func).reset_index()
    agg_df.rename(columns={value_col: new_value_col}, inplace=True)

    # 2. Calculate the weights
    if weight_method == "count":
        # Weight by N: higher fixation count = higher weight
        weights = df.groupby(group_cols)[value_col].count().reset_index(name='weight')

    elif weight_method == "log_count":
        weights = df.groupby(group_cols)[value_col].count().apply(np.log).reset_index(name='weight')

    elif weight_method == "log_precision":
        # The Ultimate Hybrid: log(1 + N) / variance
        # Combines outlier protection with the expert consistency bonus
        stats_df = df.groupby(group_cols)[value_col].agg(n_count='count', variance='var').reset_index()
        
        def calculate_log_precision(row):
            v = row['variance']
            n = row['n_count']
            if pd.isna(v) or v == 0:
                return 1e-6
            return np.log1p(n) / v
            
        stats_df['weight'] = stats_df.apply(calculate_log_precision, axis=1)
        weights = stats_df.drop(columns=['n_count', 'variance'])
    elif weight_method == "inverse_variance":
        # Weight by 1/var: lower variance = higher weight
        var_df = df.groupby(group_cols)[value_col].var().reset_index(name='variance')
        
        # Handle zero or NaN variance (e.g., if a cell only has 1 valid fixation)
        def safe_inv_var(v):
            if pd.isna(v) or v == 0:
                return 1e-6  # Assign negligible weight to undefined/zero variance
            return 1.0 / v
    
            
        var_df['weight'] = var_df['variance'].apply(safe_inv_var)
        weights = var_df.drop(columns=['variance'])

    elif weight_method == "precision":
        # Weight by N / variance: Balances trial volume with cognitive consistency
        stats_df = df.groupby(group_cols)[value_col].agg(
            n_count='count', 
            variance='var'
        ).reset_index()
        
        # Calculate N / var, handling zero or NaN variance (e.g., if a cell only has 1 fixation)
        def calculate_precision(row):
            v = row['variance']
            n = row['n_count']
            
            if pd.isna(v) or v == 0:
                return 1e-6  # Assign negligible weight to undefined/zero variance
            return n / v
            
        stats_df['weight'] = stats_df.apply(calculate_precision, axis=1)
        
        # Clean up the dataframe to match your existing return structure
        weights = stats_df.drop(columns=['n_count', 'variance'])
    # 3. Merge the aggregated values and weights
    final_df = pd.merge(agg_df, weights, on=group_cols)
    
    return final_df


def aggregators_pipeline(df: pd.DataFrame, column_name: str, aggregators: list,
                        weight_method: str = None, interaction_term: str = "+") -> pd.DataFrame:

    rows = []
    for aggregator in aggregators:
        func_name = aggregator.__name__
        val_col_name = f"{func_name}_{column_name}"

        if weight_method:
            fix_df_aggregated = weighted_condition_aggregator(df, column_name, agg_func=aggregator, weight_method=weight_method)
            _, extras = run_pymer_lmm(fix_df_aggregated, val_col_name, weight_col="weight", interaction_term=interaction_term)
        
        else:
            fix_df_aggregated = weighted_condition_aggregator(df, column_name, agg_func=aggregator)
            _, extras = run_pymer_lmm(fix_df_aggregated, val_col_name, interaction_term=interaction_term)

        rows.append({"aggregator": func_name, **extras})

    summary_df = pd.DataFrame(rows).set_index("aggregator").round(4)
    print(summary_df)
    return summary_df




def extract_lmm_participant_scores(df: pd.DataFrame, value_col: str, weight_col: str = None,
                                   participant_col: str = "participant_id",
                                   group_col: str = "group", condition_col: str = "condition"):

    formula = f"{value_col} ~ {condition_col} + (1 | {participant_col})"
    model = lmer(formula, data=pl.DataFrame(df), weights=weight_col) if weight_col else lmer(formula, data=pl.DataFrame(df))
    model.fit(summary=False)

    # Random intercepts — level col = participant ID, (Intercept) col = BLUP
    ranef = model.ranef
    ranef = ranef[0] if isinstance(ranef, list) else (next(iter(ranef.values())) if isinstance(ranef, dict) else ranef)

    # Apply Z-score normalization directly to the participant coefficients
    scores = (
        ranef
        .with_columns(
            ((pl.col('(Intercept)') - pl.col('(Intercept)').mean()) / pl.col('(Intercept)').std())
            .alias(f"{value_col}_zscore")
        )
        .rename({'level': participant_col})
        .select([participant_col, f"{value_col}_zscore"])
        .to_pandas()
    )

    group_map = df[[participant_col, group_col]].drop_duplicates()
    group_map[participant_col] = group_map[participant_col].astype(str)
    
    # Ensure participant_col in scores is also string to prevent merge conflicts
    scores[participant_col] = scores[participant_col].astype(str)

    return scores.merge(group_map, on=participant_col, how="left")