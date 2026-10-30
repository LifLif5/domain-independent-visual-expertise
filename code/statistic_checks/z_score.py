import pandas as pd

def add_composite_score(df: pd.DataFrame, direction_dict: dict) -> pd.DataFrame:
    """
    Calculates Z-scores for specified columns, aligns their direction, 
    and adds a combined composite score.
    
    direction_dict format: {'col_name': True/False} 
    (True means lower is better, so the Z-score will be inverted).
    """
    # Work on a copy to avoid modifying the original dataframe warnings
    df_out = df.copy()
    z_cols = []
    
    for col, invert in direction_dict.items():
        # Calculate standard Z-score (ddof=0 for population std)
        z_series = (df_out[col] - df_out[col].mean()) / df_out[col].std(ddof=0)
        
        # Invert the sign if lower raw values mean better performance
        if invert:
            z_series *= -1
            
        z_col_name = f"{col}_z"
        df_out[z_col_name] = z_series
        z_cols.append(z_col_name)
        
    # Average the aligned Z-scores to create the final score
    df_out['composite_score'] = df_out[z_cols].mean(axis=1)
    
    return df_out



from sklearn.metrics import roc_auc_score

def evaluate_loocv_composite(df: pd.DataFrame, direction_dict: dict, label_col = "group", pos_label = "ATCs") -> dict:
    """
    Executes Leave-One-Out Cross-Validation (LOOCV) to calculate out-of-sample 
    composite Z-scores and evaluates performance using ROC-AUC.
    
    Args:
        df: The dataset containing participant metrics and labels.
        direction_dict: {'col_name': True/False} (True means lower is better).
        label_col: The name of the column containing the group labels (e.g., 'Group').
        pos_label: The string or value indicating the target class (e.g., 'Expert').
        
    Returns:
        A dictionary containing the scored DataFrame and the ROC-AUC score.
    """
    df_out = df.copy()
    df_out['loocv_composite'] = np.nan
    
    # Execute LOOCV
    for idx in df_out.index:
        # 1. Split into Train (N-1) and Test (1)
        train_df = df_out.drop(index=idx)
        test_row = df_out.loc[idx]
        
        z_scores = []
        
        # 2. Fit and Transform without Data Leakage
        for col, invert in direction_dict.items():
            # Fit: Calculate mean and std strictly on the N-1 train set
            train_mean = train_df[col].mean()
            train_std = train_df[col].std(ddof=0)
            
            # Transform: Scale the left-out test sample
            z_val = (test_row[col] - train_mean) / train_std
            
            # Invert sign if lower raw values indicate better performance
            if invert:
                z_val *= -1
                
            z_scores.append(z_val)
            
        # 3. Calculate composite score for the out-of-sample participant
        df_out.at[idx, 'loocv_composite'] = np.mean(z_scores)
        
    # 4. Evaluate overall predictive validity
    y_true = (df_out[label_col] == pos_label).astype(int)
    y_scores = df_out['loocv_composite']
    
    auc_score = roc_auc_score(y_true, y_scores)
    
    return {
        'scored_dataframe': df_out,
        'roc_auc': auc_score
    }