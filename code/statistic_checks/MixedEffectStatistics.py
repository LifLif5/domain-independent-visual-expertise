import pandas as pd
import numpy as np
from typing import Callable

def condition_aggregator(
    df: pd.DataFrame, 
    value_col: str,
    participant_col: str = "participant_id", 
    group_col: str = "group", 
    condition_col: str = "condition", 
    agg_func: Callable = np.mean
) -> pd.DataFrame:
    """
    Aggregates raw data using a provided function (e.g., np.median, np.mean) 
    for each combination of participant, group, and condition.
    """
    # Group by the identifiers and apply the custom aggregator function
    agg_df = df.groupby(
        [participant_col, group_col, condition_col]
    )[value_col].agg(agg_func).reset_index()
    
    # Dynamically rename the column based on the function's name
    # Fallback to 'agg' if the function doesn't have a __name__ attribute (like some lambdas)
    func_name = getattr(agg_func, '__name__', 'agg')
    new_value_col = f'{func_name}_{value_col}'
    
    agg_df.rename(columns={value_col: new_value_col}, inplace=True)
    
    return agg_df


from scipy.stats import shapiro

def check_normality_per_cell(df: pd.DataFrame, value_col: str, group_col: str = "group", condition_col: str ="condition") -> pd.DataFrame:
    """
    Checks the normality of the value_col for each Group x Condition cell 
    using the Shapiro-Wilk test. Assumes df is already aggregated per participant.
    """
    results = []
    
    # Group by both group and condition to isolate each specific cell
    for (group_name, cond_name), cell_data in df.groupby([group_col, condition_col]):
        
        # Extract just the numbers for this cell
        values = cell_data[value_col].dropna()
        
        # The Shapiro-Wilk test requires at least 3 data points to run
        if len(values) >= 3:
            stat, p_val = shapiro(values)
            # If p > 0.05, we fail to reject the null hypothesis; the data is considered normal
            is_normal = p_val > 0.05 
        else:
            stat, p_val, is_normal = None, None, None
            
        results.append({
            group_col: group_name,
            condition_col: cond_name,
            'n_participants': len(values),
            'W_statistic': stat,
            'p_value': p_val,
            'is_normal': is_normal
        })
        
    return pd.DataFrame(results)

from scipy.stats import levene

def check_homoscedasticity(df: pd.DataFrame, value_col: str, group_col: str = "group", condition_col: str = "condition") -> dict:
    """
    Performs Levene's test to check for equal variance across all Group x Condition cells.
    Assumes df is aggregated per participant.
    """
    # 1. Group the data into a list of arrays (one array per cell)
    groups_data = [group[value_col].dropna().values for _, group in df.groupby([group_col, condition_col])]
    
    # 2. Filter out groups with fewer than 2 points (Levene needs variance to calculate)
    groups_data = [g for g in groups_data if len(g) >= 2]
    
    # 3. Run Levene's Test
    # 'median' is the most robust center for skewed reaction time data
    stat, p_val = levene(*groups_data, center='median')
    
    # 4. Interpret
    is_homoscedastic = p_val > 0.05
    
    return {
        'test_type': "Levene's Test (centered on median)",
        'statistic': stat,
        'p_value': p_val,
        'is_homoscedastic': is_homoscedastic,
        'conclusion': "Equal variance (Passed)" if is_homoscedastic else "Unequal variance (Failed)"
    }

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt

def run_linear_mixed_model(df: pd.DataFrame, value_col: str,
                               participant_col: str = "participant_id", 
                               group_col: str = "group", condition_col: str ="condition",
                               plot: bool = False):
    """
    Runs LMM, returns results, R2 metrics, and the joint significance of condition.
    """
    # 1. Define the formula (Fixed effects)
    formula = f"{value_col} ~ C({group_col}) + C({condition_col})"
    
    # 2. Build and Fit the model
    model = smf.mixedlm(formula, data=df, groups=df[participant_col])
    result = model.fit()
    
    # 3. Calculate R-squared (Marginal and Conditional)
    # Marginal: Variance explained by fixed effects
    # Conditional: Variance explained by both fixed and random effects
    # Note: Statsmodels doesn't have built-in R2 for mixedlm; we use the 
    # Nakagawa & Schielzeth (2013) approach logic or helper attributes.
    # For a quick implementation in statsmodels:
    fe_var = result.fittedvalues.var()
    re_var = float(result.cov_re.iloc[0, 0])
    resid_var = result.scale
    
    marginal_r2 = fe_var / (fe_var + re_var + resid_var)
    conditional_r2 = (fe_var + re_var) / (fe_var + re_var + resid_var)
    
    # 4. Joint P-value for Condition (Wald Test)
    # This tests if all levels of 'condition' are zero.
    condition_terms = [term for term in result.params.index if f"C({condition_col})" in term]
    wald_test = result.wald_test(condition_terms)
    condition_p_value = wald_test.pvalue

    # Print results
    # print(result.summary())
    # print(f"\n--- Model Metrics ---")
    # print(f"Marginal R2 (Fixed effects): {marginal_r2:.4f}")
    # print(f"Conditional R2 (Total model): {conditional_r2:.4f}")
    # print(f"Joint p-value for '{condition_col}': {condition_p_value:.4e}")
    
    # 5. Diagnostic Plots
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(result.fittedvalues, result.resid, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel("Fitted Values")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs. Fitted")
        
        sm.qqplot(result.resid, line='45', fit=True, ax=axes[1], alpha=0.5)
        axes[1].set_title("Q-Q Plot")
        plt.tight_layout()
        plt.show()
        
    return result, {"marginal_r2": marginal_r2, 
                    "conditional_r2": conditional_r2, 
                    "condition_p_value": condition_p_value}

from scipy.stats import shapiro, skew
def check_raw_distributions(df: pd.DataFrame, value_col: str,
                            participant_col: str = "participant_id", condition_col: str ="condition"
) -> pd.DataFrame:
    """
    Tests raw data within each participant x condition cell.
    Includes dynamic p-values for large N and calculates skewness.
    """
    results = []
    
    for (pid, cond), cell_data in df.groupby([participant_col, condition_col]):
        values = cell_data[value_col].dropna()
        n = len(values)
        
        if n >= 3:
            # 1. Standard Shapiro-Wilk
            stat_norm, p_norm = shapiro(values)
            
            # 2. Dynamic Alpha (Baseline N=50)
            # If N < 50, keep alpha at 0.05. If N > 50, scale it down.
            dynamic_alpha = 0.05 * np.sqrt(50 / n) if n > 50 else 0.05
            is_normal_dynamic = p_norm > dynamic_alpha
            
            # 3. Log-Normal test
            log_values = np.log(values)
            stat_log, p_log = shapiro(log_values)
            is_lognormal = p_log > dynamic_alpha
            
            # 4. Effect Size: Skewness (Is it between -2 and 2?)
            cell_skew = skew(values)
            is_skew_acceptable = -2.0 <= cell_skew <= 2.0
            
        else:
            p_norm, dynamic_alpha, is_normal_dynamic = None, None, None
            p_log, is_lognormal = None, None
            cell_skew, is_skew_acceptable = None, None
            
        results.append({
            participant_col: pid,
            condition_col: cond,
            'n_fixations': n,
            'p_norm': p_norm,
            'dynamic_alpha': dynamic_alpha,
            'is_normal_dynamic': is_normal_dynamic,
            'is_lognormal': is_lognormal,
            'skewness': cell_skew,
            'skew_acceptable': is_skew_acceptable
        })
        
    return pd.DataFrame(results)




import warnings
from typing import Callable

def statistical_pipeline_wrapper(
    df: pd.DataFrame, 
    value_col: str,
    participant_col: str = "participant_id", 
    group_col: str = "group", 
    condition_col: str = "condition", 
    agg_func: Callable = np.mean,
    normality_threshold: float = 0.80
) -> pd.Series:
    
    # a. Aggregate the data
    agg_df = condition_aggregator(
        df, value_col, participant_col, group_col, condition_col, agg_func
    )
    
    # Identify the newly created aggregated column name
    func_name = getattr(agg_func, '__name__', 'agg')
    agg_value_col = f'{func_name}_{value_col}'
    
    # b. Run normality test and check threshold
    normality_results = check_normality_per_cell(
        agg_df, agg_value_col, group_col, condition_col
    )
    
    # Calculate the percentage of cells that passed the normality test
    valid_cells = normality_results.dropna(subset=['is_normal'])
    if not valid_cells.empty:
        normality_pct = valid_cells['is_normal'].mean()
    else:
        normality_pct = 0.0
        
    if normality_pct < normality_threshold:
        warnings.warn(
            f"Normality warning: Only {normality_pct*100:.1f}% of cells passed the Shapiro-Wilk test "
            f"(Threshold: {normality_threshold*100:.1f}%)."
        )

    # c. Run homoscedasticity check
    homo_results = check_homoscedasticity(
        agg_df, agg_value_col, group_col, condition_col
    )
    
    log_used = False
    lmm_target_col = agg_value_col
    lmm_df = agg_df.copy()

    # d. If it fails, try log transforming
    if not homo_results['is_homoscedastic']:
        # Apply log transformation
        lmm_target_col = f"log_{agg_value_col}"
        # We drop or handle 0s to prevent -inf in log transformation
        lmm_df = lmm_df[lmm_df[agg_value_col] > 0].copy() 
        lmm_df[lmm_target_col] = np.log(lmm_df[agg_value_col])
        
        # Re-run homoscedasticity on log-transformed data
        homo_results_log = check_homoscedasticity(
            lmm_df, lmm_target_col, group_col, condition_col
        )
        
        if homo_results_log['is_homoscedastic']:
            log_used = True
        else:
            # If it still fails after log transformation, return nothing
            return None
                
    # Run the Linear Mixed Model
    lmm_result = run_linear_mixed_model(
        lmm_df, lmm_target_col, participant_col, group_col, condition_col
    )
    
    # Extract the p-value(s), coef, and key for the group effect
    group_keys = [key for key in lmm_result.params.index if 'group' in key]
    group_key = group_keys[0] if group_keys else "Unknown Group"
    
    coef = lmm_result.params.get(group_key, np.nan)
    p_val = lmm_result.pvalues.get(group_key, np.nan)
    
    # Return everything needed for the summary
    return pd.Series({
        'group_key': group_key,
        'coef': coef,
        'p_value': p_val,
        'normality_passed_pct': normality_pct,
        'log_used': log_used,
        'lmm_result': lmm_result,
        'final_col_name': lmm_target_col
    })


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from typing import Callable

def get_adjusted_participant_scores(
    df: pd.DataFrame, 
    value_col: str,
    participant_col: str = "participant_id", 
    group_col: str = "group", 
    condition_col: str = "condition", 
    agg_func: Callable = np.mean
) -> pd.DataFrame:
    
    # 1. Aggregate data to the condition level per participant
    agg_df = df.groupby(
        [participant_col, group_col, condition_col], as_index=False
    )[value_col].agg(agg_func)
    
    func_name = getattr(agg_func, '__name__', 'agg')
    agg_value_col = f'{func_name}_{value_col}'
    agg_df.rename(columns={value_col: agg_value_col}, inplace=True)
    
    # 2. Fit the LMM (Adjusting for condition, random intercept per participant)
    # Note: We wrap condition in C() to ensure it's treated categorically
    formula = f"{agg_value_col} ~ C({condition_col})"
    model = smf.mixedlm(formula, agg_df, groups=agg_df[participant_col])
    
    # Suppress warnings for non-convergence if you only care about the intercepts
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = model.fit()
    
    # 3. Extract Fixed Intercept and Random Effects
    fixed_intercept = result.params["Intercept"]
    
    # result.random_effects returns a dict: {participant: Series(Group=value)}
    random_effects = {k: v.iloc[0] for k, v in result.random_effects.items()}
    
    # 4. Build the final DataFrame
    scores_df = pd.DataFrame(
        list(random_effects.items()), 
        columns=[participant_col, 'random_effect']
    )
    
    # Adjusted score = Baseline average (Intercept) + participant's specific deviation
    scores_df[value_col] = fixed_intercept + scores_df['random_effect']
    
    # Merge the group column back in (assuming 1 group per participant)
    group_map = agg_df[[participant_col, group_col]].drop_duplicates()
    final_df = scores_df.merge(group_map, on=participant_col, how='left')
    
    return final_df[[participant_col, group_col, value_col]]

def calculate_z_score(series: pd.Series, invert_direction: bool = False) -> pd.Series:
    """
    Calculates the Z-score for a given metric.
    
    Parameters:
    - series: The pandas Series containing the raw metric.
    - invert_direction: Set to True if a lower value represents better performance.
    """
    # ddof=0 calculates the population standard deviation
    z_scores = (series - series.mean()) / series.std(ddof=0)
    
    if invert_direction:
        z_scores = z_scores * -1
        
    return z_scores
