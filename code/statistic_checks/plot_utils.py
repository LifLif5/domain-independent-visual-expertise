import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.stats import ttest_ind

# 1. Define your specific color mapping
# 'tab:blue', 'tab:orange', and 'tab:green' are the standard Matplotlib defaults
color_map = {
    "Novices": "tab:blue",
    "Experienced": "tab:orange",
    "Experts": "tab:green",
    "Students": "tab:blue",
    "ATCs": "tab:orange",}

# 2. Define the specific order you want on the X-axis
# This ensures the categories appear in the sequence you prefer
group_order = ["Novices", "Experienced", "Experts", "Students", "ATCs"]

def plot_feature_by_group(
    df: pd.DataFrame,
    feature: str,
    group_col: str = "group",
    kind: str = "dots",          # "dots" or "dist"
    dist_style: str = "kde",     # if kind="dist": "kde" or "violin"
    add_summary: bool = True,    # overlay medians (and IQR for violins)
    jitter: float = 0.25,        # horizontal jitter for dots
    point_alpha: float = 0.7,    # transparency for dots
    figsize=(8, 4.5),
    title: str | None = None,
    save_path: str | None = None # e.g., "feature_plot.png"
):
    """
    Plot one feature split by groups.

    kind="dots":   jittered dots per group (optionally with group median line)
    kind="dist":   painted distribution per group:
                   - dist_style="kde": overlaid KDEs (filled)
                   - dist_style="violin": side-by-side violins

    Notes:
    - Uses seaborn default palette to color groups.
    - Drops rows where feature or group is NA.
    - For KDE, extreme outliers can dominate the bandwidth; consider
      transforming or clipping before plotting if needed.
    """
    assert kind in {"dots", "dist"}, "kind must be 'dots' or 'dist'"
    if kind == "dist":
        assert dist_style in {"kde", "violin"}, "dist_style must be 'kde' or 'violin'"

    # tidy subset
    d = df[[group_col, feature]].dropna()
    if d.empty:
        raise ValueError(f"No data to plot after dropping NA in {feature} or {group_col}.")

    # order groups as they appear
    order = d[group_col].dropna().astype(str).unique().tolist()

    plt.figure(figsize=figsize)
    sns.set(style="whitegrid", context="talk")

    if kind == "dots":
        # stripplot (jittered points)
        ax = sns.stripplot(
            data=d, x=group_col, y=feature, order=order,
            jitter=jitter, dodge=False, alpha=point_alpha
        )
        # (optional) overlay medians
        if add_summary:
            med = d.groupby(group_col, sort=False)[feature].median()
            for i, g in enumerate(order):
                if g in med.index:
                    ax.hlines(med[g], i-0.35, i+0.35, linewidth=3)

        ax.set_xlabel(group_col)
        ax.set_ylabel(feature)

    else:
        if dist_style == "kde":
            # Overlaid KDEs
            # Draw each group's KDE with fill; add a thin outline for clarity
            palette = sns.color_palette(n_colors=len(order))
            for c, g in zip(palette, order):
                x = d.loc[d[group_col] == g, feature].values
                if len(x) < 2:
                    continue
                sns.kdeplot(x=x, fill=True, alpha=0.35, linewidth=0.8, label=str(g), color=c)
            if add_summary:
                # vertical median lines per group
                med = d.groupby(group_col, sort=False)[feature].median()
                for c, g in zip(palette, order):
                    if g in med.index and np.isfinite(med[g]):
                        plt.axvline(med[g], linestyle="--", linewidth=1, alpha=0.8, color=c)

            plt.xlabel(feature)
            plt.ylabel("Density")
            plt.legend(title=group_col, frameon=False)

        elif dist_style == "violin":
            ax = sns.violinplot(
                data=d, x=group_col, y=feature, order=order,
                inner=None, cut=0
            )
            if add_summary:
                # add median and IQR
                stats = d.groupby(group_col, sort=False)[feature].describe()
                for i, g in enumerate(order):
                    if g in stats.index:
                        q1, med, q3 = stats.loc[g, ["25%", "50%", "75%"]]
                        ax.hlines(med, i-0.25, i+0.25, linewidth=3)
                        ax.vlines(i, q1, q3, linewidth=3, alpha=0.8)
            ax.set_xlabel(group_col)
            ax.set_ylabel(feature)

    if title is None:
        title = f"{feature} by {group_col} ({'dots' if kind=='dots' else dist_style})"
    plt.title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_feature_over_trials(
    df, feature, id_col='participant_id', 
    trial_col='trial_id', group_col='group'
):
    """
    Plot group-level mean and SEM over trials using specific colors and order.
    Assumes 'color_map' and 'group_order' are defined in the outer scope.
    """
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # --- compute group-level stats ---
    gstats = (
        df.groupby([group_col, trial_col])[feature]
        .agg(['mean', 'sem'])
        .reset_index()
    )

    # --- plot each group in the specific order ---
    # We loop through group_order to ensure the sequence and legend order
    for g in group_order:
        # Skip if the group in group_order isn't actually in the dataframe
        if g not in gstats[group_col].unique():
            continue
            
        gdf = gstats[gstats[group_col] == g]
        # Get color from your map; default to 'gray' if missing
        clr = color_map.get(g, "gray")

        # mean curve
        plt.plot(
            gdf[trial_col],
            gdf['mean'],
            linewidth=2.5,
            label=f"{g}",
            color=clr
        )

        # SEM band
        plt.fill_between(
            gdf[trial_col],
            gdf['mean'] - gdf['sem'],
            gdf['mean'] + gdf['sem'],
            color=clr,
            alpha=0.2,
            edgecolor=None # Removes thin lines around the shade
        )

    plt.title(f"Trial-wise evolution of {feature}", fontsize=14)
    plt.xlabel("Trial Number")
    plt.ylabel(feature)
    
    # Legend outside to the right
    plt.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    plt.show()



def plot_participant_means(df, feature):
    # aggregate trials → participant mean
    agg = (
        df.groupby(["group", "participant_id"])[feature]
        .mean()
        .reset_index()
    )

    
    
    # Filter group_order to only include groups actually present in the data
    groups = [g for g in group_order if g in agg["group"].unique()]

    # Use subplots for better control (replaces plt.figure)
    fig, ax = plt.subplots(figsize=(6, 4))

    for g in groups:
        sub = agg[agg["group"] == g]
        ax.scatter(
            [g] * len(sub),
            sub[feature],
            color=color_map.get(g, "gray"),  # Look up color from our map
            label=g,
            s=60,
            alpha=0.8
        )

    ax.set_title(f"Participants {feature}")
    ax.set_ylabel(feature)
    
    # Legend outside plot (right side), no overlap
    ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show() # Or plt.savefig('output.png')

def plot_features_xy(
    df,
    feature_x,
    feature_y,
    per_participant=True
):
    """
    Scatter plot of two features with specific colors and group ordering.
    Assumes 'color_map' and 'group_order' are defined in the outer scope.
    """

    if per_participant:
        # Aggregating all trials to a single mean point per participant
        data = (
            df.groupby(["group", "participant_id"])[[feature_x, feature_y]]
            .mean()
            .reset_index()
        )
    else:
        data = df.copy()

    plt.figure(figsize=(6, 5))

    # --- plot each group in the specific order ---
    for g in group_order:
        # Ensure the group exists in the current data subset
        if g not in data["group"].unique():
            continue
            
        sub = data[data["group"] == g]
        
        plt.scatter(
            sub[feature_x],
            sub[feature_y],
            color=color_map.get(g, "gray"), # Apply your specific color
            label=g,
            s=60,
            alpha=0.8,  # Slight transparency helps see overlapping points
            edgecolors='white', # Added white edge for better point separation
            linewidths=0.5
        )

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    
    title_suffix = "(participant means)" if per_participant else "(trial level)"
    plt.title(f"{feature_x} vs {feature_y}\n{title_suffix}", fontsize=12)

    # Legend outside to the right
    plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    plt.show()


def tsne_trials_dual_plots(
    df: pd.DataFrame,
    id_col: str = "participant_id",
    group_col: str = "group",
    trial_col: str = "trial_id",
    feature_cols: list | None = None,
    perplexity: float | None = None,
    n_iter: int = 1000,          # sklearn>=1.7: passed as max_iter
    random_state: int = 42,
    pca_components: int | None = 50,
    point_size: int = 28,
    alpha: float = 0.85,
    legend: bool = True,
    legend_max_items_group: int = 18,
    legend_max_items_participant: int = 18,
    annotate: bool = False,      # annotate with participant_id
):
    """
    Compute ONE t-SNE embedding of trial-level features and plot it TWICE:
      (1) colored by group
      (2) colored by participant_id
    Ensures identical 2D representation across both plots (same coordinates/limits).

    Returns
    -------
    (fig_group, ax_group), (fig_part, ax_part), emb_df

    emb_df has columns: ['tsne1','tsne2', id_col, group_col, trial_col]
    """

    df = df.copy()

    # --------- Select features ----------
    meta_cols = [c for c in [id_col, group_col, trial_col] if c in df.columns]
    if feature_cols is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric if c not in meta_cols]
        if not feature_cols:
            raise ValueError("No numeric feature columns found after excluding id/group/trial.")

    X = df[feature_cols].to_numpy()
    n_samples, n_features = X.shape

    # --------- Impute & standardize ----------
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    # --------- Optional PCA pre-step ----------
    if pca_components is not None:
        k = int(np.clip(pca_components, 2, min(50, n_features)))
        X = PCA(n_components=k, random_state=random_state).fit_transform(X)

    # --------- Perplexity heuristic ----------
    if perplexity is None:
        perplexity = max(5, min(30, (n_samples - 1) // 3))
    if perplexity >= n_samples:
        perplexity = max(5, min(30, n_samples // 3)) or 5

    # --------- t-SNE (sklearn>=1.7 uses max_iter) ----------
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        metric="euclidean",
        verbose=0,
    )
    emb = tsne.fit_transform(X)

    emb_df = pd.DataFrame({"tsne1": emb[:, 0], "tsne2": emb[:, 1]})
    if id_col in df.columns:    emb_df[id_col] = df[id_col].astype(str).values
    if group_col in df.columns: emb_df[group_col] = df[group_col].astype(str).values
    if trial_col in df.columns: emb_df[trial_col] = df[trial_col].values

    # Shared axis limits for identical visual framing
    x_min, x_max = emb_df["tsne1"].min(), emb_df["tsne1"].max()
    y_min, y_max = emb_df["tsne2"].min(), emb_df["tsne2"].max()
    pad_x = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    pad_y = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    xlims = (x_min - pad_x, x_max + pad_x)
    ylims = (y_min - pad_y, y_max + pad_y)

    def _plot_colored(emb_df, color_col, title, legend_max_items):
        fig, ax = plt.subplots(figsize=(8, 6))
        cats = emb_df[color_col].astype(str)
        uniq = sorted(cats.unique())

        # Build a long color cycle (repeat tab20)
        base = plt.get_cmap("tab20").colors
        colors = [base[i % len(base)] for i in range(len(uniq))]
        cmap = {u: colors[i] for i, u in enumerate(uniq)}

        for u in uniq:
            m = cats == u
            ax.scatter(
                emb_df.loc[m, "tsne1"],
                emb_df.loc[m, "tsne2"],
                s=point_size, alpha=alpha, edgecolors="none",
                c=[cmap[u]], label=u
            )

        if annotate and (id_col in emb_df.columns):
            for _, r in emb_df.iterrows():
                ax.annotate(str(r[id_col]), (r["tsne1"], r["tsne2"]), fontsize=7, alpha=0.7)

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(title)

        if legend:
            if len(uniq) <= legend_max_items:
                ax.legend(title=color_col, frameon=True, loc="best")
            else:
                # Keep legend readable by showing the most frequent categories
                top = cats.value_counts().head(legend_max_items).index.tolist()
                handles, labels = ax.get_legend_handles_labels()
                keep = [i for i, lab in enumerate(labels) if lab in top]
                ax.legend([handles[i] for i in keep], [labels[i] for i in keep],
                          title=f"{color_col} (top {len(top)})", frameon=True, loc="best")

        fig.tight_layout()
        return fig, ax

    # Plot 1: colored by group
    if group_col not in emb_df.columns:
        raise ValueError(f"Column `{group_col}` not found; required for group-colored plot.")
    fig_g, ax_g = _plot_colored(
        emb_df, group_col, f"t-SNE of trial features (colored by {group_col})",
        legend_max_items_group
    )

    # Plot 2: colored by participant_id
    if id_col not in emb_df.columns:
        raise ValueError(f"Column `{id_col}` not found; required for participant-colored plot.")
    fig_p, ax_p = _plot_colored(
        emb_df, id_col, f"t-SNE of trial features (colored by {id_col})",
        legend_max_items_participant
    )

    return (fig_g, ax_g), (fig_p, ax_p), emb_df


def pca_trials_plot(
    df: pd.DataFrame,
    id_col: str = "participant_id",
    group_col: str = "group",
    trial_col: str = "trial_id",
    feature_cols: list | None = None,
    color_by: str = "group",
    n_components: int = 2,
    point_size: int = 28,
    alpha: float = 0.85,
    legend: bool = True,
    legend_max_items: int = 18,
):
    """
    PCA visualization of trial-level features in 2D.
    Color points by 'group', 'participant', or any categorical column.
    """

    df = df.copy()
    meta_cols = [id_col, group_col, trial_col]
    if feature_cols is None:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_cols
        ]

    if color_by not in df.columns:
        raise ValueError(f"{color_by} column not found in dataframe.")

    X = df[feature_cols].to_numpy()
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(X)

    emb_df = pd.DataFrame(
        {"PC1": pcs[:, 0], "PC2": pcs[:, 1], color_by: df[color_by].astype(str)}
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    cats = emb_df[color_by]
    uniq = sorted(cats.unique())
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(uniq))]

    for i, u in enumerate(uniq):
        mask = cats == u
        ax.scatter(
            emb_df.loc[mask, "PC1"],
            emb_df.loc[mask, "PC2"],
            s=point_size,
            alpha=alpha,
            edgecolors="none",
            color=colors[i],
            label=u,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"PCA of trial features (colored by {color_by})")

    if legend:
        if len(uniq) <= legend_max_items:
            ax.legend(title=color_by, frameon=True, loc="best")
        else:
            ax.legend(
                [u for u in uniq[:legend_max_items]],
                title=f"{color_by} (top {legend_max_items})",
                frameon=True,
                loc="best",
            )

    fig.tight_layout()
    return fig, ax, emb_df, pca



def plot_feature_over_bins(
    df, feature, bin_col='duration_bin', group_col='group',
    color_map=None, group_order=None
):
    """
    Plot group-level mean and SEM across discrete bins (e.g., duration quintiles).
    Expects 'color_map' and 'group_order' to be passed as arguments or defined in scope.
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid") # Updated to set_theme (modern seaborn)

    participant_col = 'participant_id' # Change this to your actual column name

    # Step 1: Participant-level average
    # Gets one single mean value per participant, per group, per bin
    participant_stats = (
        df.groupby([group_col, participant_col, bin_col])[feature]
        .mean() 
        .reset_index()
    )

    # Step 2: Group-level average
    # Calculates the grand mean and standard error ACROSS participants
    gstats = (
        participant_stats.groupby([group_col, bin_col])[feature]
        .agg(['mean', 'sem'])
        .reset_index()
    )

    # Fallbacks in case order/colors aren't passed
    if group_order is None:
        group_order = df[group_col].unique()
    if color_map is None:
        color_map = {g: "blue" if i==0 else "orange" for i, g in enumerate(group_order)}

    # --- plot each group ---
    for g in group_order:
        if g not in gstats[group_col].unique():
            continue
            
        # Sort by bin to ensure the line draws left-to-right correctly
        gdf = gstats[gstats[group_col] == g].sort_values(by=bin_col)
        clr = color_map.get(g, "gray")

        # mean curve - Added markers for discrete bins
        plt.plot(
            gdf[bin_col],
            gdf['mean'],
            marker='o',          # Explicitly marks the 5 bin centers
            markersize=8,
            linewidth=2.5,
            label=f"{g}",
            color=clr
        )

        # SEM band
        plt.fill_between(
            gdf[bin_col],
            gdf['mean'] - gdf['sem'],
            gdf['mean'] + gdf['sem'],
            color=clr,
            alpha=0.2,
            edgecolor=None
        )

    plt.title(f"Intra-fixation {feature} across Duration Quintiles", fontsize=14, pad=15)
    plt.xlabel("Fixation Duration Bin (Quintile)", labelpad=10)
    plt.ylabel(feature, labelpad=10)
    
    # Force X-axis ticks to strictly align with your 5 bins (e.g., 1, 2, 3, 4, 5)
    plt.xticks(sorted(df[bin_col].unique()))
    
    # Legend outside to the right
    plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    plt.show()


def plot_group_diff_by_factor(
    df, feature, factor_col, group_col='group', participant_col='participant_id'
):
    """
    Plots participant-level differences between groups, split by a specific factor.
    Aggregates from raw data to correctly weight the true number of fixations.
    """
    # 1. Aggregate from raw data to get accurate participant-level means per factor level.
    # This naturally accounts for different fixation counts because we average the raw rows directly.
    df_subj = df.groupby([participant_col, group_col, factor_col])[feature].mean().reset_index()
    
    # 2. Setup the plot
    plt.figure(figsize=(11, 7))
    sns.set_theme(style="whitegrid")
    
    # 3. Plot participant dots (stripplot) to show the underlying participant distribution
    sns.stripplot(
        data=df_subj, x=factor_col, y=feature, hue=group_col,
        dodge=True, alpha=0.5, jitter=True, zorder=1,
        palette={'Novices': 'blue', 'ATCs': 'orange'} # adjust colors if needed
    )
    
    # 4. Plot group means and confidence intervals (pointplot)
    ax = sns.pointplot(
        data=df_subj, x=factor_col, y=feature, hue=group_col,
        dodge=0.4, markers=['o', 's'], err_kws={'linewidth': 2}, capsize=.1,
        palette={'Novices': 'darkblue', 'ATCs': 'darkorange'}, 
        linestyles=["-", "--"], zorder=2
    )
    
    # Clean up the duplicate legends created by using both strip and point plots
    handles, labels = ax.get_legend_handles_labels()
    n_groups = df_subj[group_col].nunique()
    plt.legend(
        handles[:n_groups], labels[:n_groups], 
        title="Group", bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    
    # 5. Statistical Annotations
    factors = sorted(df_subj[factor_col].dropna().unique())
    y_max = df_subj[feature].max()
    y_range = y_max - df_subj[feature].min()
    
    groups = df_subj[group_col].unique()
    
    if len(groups) == 2:
        g1_name, g2_name = groups[0], groups[1]
        
        for i, factor_val in enumerate(factors):
            # Isolate the participant means for this specific factor level
            g1_data = df_subj[(df_subj[factor_col] == factor_val) & (df_subj[group_col] == g1_name)][feature]
            g2_data = df_subj[(df_subj[factor_col] == factor_val) & (df_subj[group_col] == g2_name)][feature]
            
            # Run Welch's t-test (assumes unequal variance, which is safer)
            if len(g1_data) > 1 and len(g2_data) > 1:
                stat, p_val = ttest_ind(g1_data, g2_data, equal_var=False)
                
                if p_val < 0.01: sig = '**'
                elif p_val < 0.05: sig = '*'
                elif p_val < 0.10: sig = f'p={p_val:.2f}'
                else: sig = 'ns'
                
                # Draw significance text
                plt.text(i, y_max + (y_range * 0.05), sig, ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.ylim(top=y_max + (y_range * 0.15))
    
    # Dynamic title and labels
    formatted_factor = factor_col.replace('_', ' ').title()
    plt.title(f"Participant-Level {feature} across {formatted_factor}", fontsize=15, pad=20)
    plt.xlabel(formatted_factor, labelpad=10)
    plt.ylabel(f"Mean {feature}", labelpad=10)
    
    plt.tight_layout()
    plt.show()