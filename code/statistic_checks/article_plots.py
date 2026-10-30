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

# Shared font sizes for article plots
SUPTITLE_FONT_SIZE = 19
SUBTITLE_FONT_SIZE = 17
X_FONT_SIZE = 17
Y_FONT_SIZE = 17
TICK_FONT_SIZE = 15
LEGEND_FONT_SIZE = 11
STAT_HEADER_FONT_SIZE = 17
STAT_TEXT_FONT_SIZE = 15


def plot_phase_subplots(
    df, feature, y_label, plot_title, stat_box=None,
    condition_col='condition', group_col='group', participant_col='participant_id'
):
    """
    Creates 3 side-by-side subplots (one for each phase). 
    Maps the 9 continuous conditions into Phase (1, 2, 3) and Set Size.
    """
    condition_mapping = {
        "Cond_0": {'Phase': '1: Color', 'Set_Size': 8},
        "Cond_1": {'Phase': '1: Color', 'Set_Size': 32},
        "Cond_2": {'Phase': '1: Color', 'Set_Size': 66},
        "Cond_3": {'Phase': '2: Shape', 'Set_Size': 8},
        "Cond_4": {'Phase': '2: Shape', 'Set_Size': 32},
        "Cond_5": {'Phase': '2: Shape', 'Set_Size': 66},
        "Cond_6": {'Phase': '3: Conjunction', 'Set_Size': 8},
        "Cond_7": {'Phase': '3: Conjunction', 'Set_Size': 32},
        "Cond_8": {'Phase': '3: Conjunction', 'Set_Size': 66}
    }
    
    df_plot = df.copy()
    df_plot['Phase'] = df_plot[condition_col].map(lambda c: condition_mapping.get(c, {}).get('Phase'))
    df_plot['Set_Size'] = df_plot[condition_col].map(lambda c: condition_mapping.get(c, {}).get('Set_Size'))
    
    df_subj = df_plot.groupby(
        [participant_col, group_col, 'Phase', 'Set_Size']
    )[feature].mean().reset_index()
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    
    phases = ['1: Color', '2: Shape', '3: Conjunction']
    
    for i, phase in enumerate(phases):
        ax = axes[i]
        phase_data = df_subj[df_subj['Phase'] == phase]
        
        sns.pointplot(
            data=phase_data, x='Set_Size', y=feature, hue=group_col,
            ax=ax, dodge=0.1, markers=['o', 's'], err_kws={'linewidth': 2}, capsize=.05,
            palette={'Novices': 'darkblue', 'ATCs': 'darkorange'}, 
            linestyles=["-", "--"]
        )
        
        ax.set_title(f"Phase {phase}", fontsize=SUBTITLE_FONT_SIZE, pad=10)
        ax.set_xlabel('')
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        
        # Original Legend Logic Restored
        if i == 0:
            ax.set_ylabel(y_label, labelpad=12, fontsize=Y_FONT_SIZE)
            ax.legend(title="Expertise Level", loc="upper left", frameon=True, fontsize=LEGEND_FONT_SIZE)
        else:
            ax.set_ylabel('')
            if ax.get_legend():
                ax.get_legend().remove()
                
    # Pushed the title much higher
    plt.suptitle(plot_title, fontsize=SUPTITLE_FONT_SIZE, y=1.05, fontweight='bold')
    
    fig.text(0.5, 0.02, 'Set Size', ha='center', va='center', fontsize=X_FONT_SIZE)
    
    if stat_box:
        # Place "Group significance:" alone on the left (top banner level)
        fig.text(
            0.01, 0.92, "Group significance:", 
            ha='left', va='center', fontsize=STAT_HEADER_FONT_SIZE, fontweight='bold'
        )

        # Render the stats inside the box, centered
        stat_string = "   •   ".join([f"{k}: {v}" for k, v in stat_box.items()]).replace('\n', ' ')
        
        fig.text(
            0.5, 0.92, stat_string, 
            ha='center', va='center', fontsize=STAT_TEXT_FONT_SIZE, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', edgecolor='darkgray', alpha=1.0)
        )
        
        # The top of the plots is strictly at 0.90, making the gap tight
        plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    else:
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig, axes

def plot_mot_feature(
    df, feature, y_label, plot_title, stat_box=None,
    condition_col='condition', group_col='group', participant_col='participant_id'
):
    condition_mapping = {
        0: "(4, 11)", 1: "(4, 13)", 2: "(4, 15)", 3: "(4, 17)", 
        4: "(4, 19)", 5: "(5, 10)", 6: "(5, 12)", 7: "(6, 9)", 
        8: "(6, 11)", 9: "(7, 8)", 10: "(7, 10)"
    }
    
    df_subj = df.groupby(
        [participant_col, group_col, condition_col]
    )[feature].mean().reset_index()
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 6)) 
    
    x_order = list(condition_mapping.keys())
    
    # Switched to barplot: inherently unlinked, calculates mean/CI by default
    sns.barplot(
        data=df_subj, x=condition_col, y=feature, hue=group_col, order=x_order,
        ax=ax, capsize=.05, err_kws={'linewidth': 2},
        palette={'Novices': 'darkblue', 'ATCs': 'darkorange'}
    )
    
    ax.set_xticklabels([condition_mapping[cond] for cond in x_order], fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
    
    # Consistent Y and X labels
    ax.set_xlabel("(Number of Targets, Number of Distractors)", labelpad=12, fontsize=X_FONT_SIZE)
    ax.set_ylabel(y_label, labelpad=12, fontsize=Y_FONT_SIZE)
    
    # Title restored to its original height to maintain space
    plt.suptitle(plot_title, fontsize=SUPTITLE_FONT_SIZE, y=1.05, fontweight='bold')
    
    # Legend moved to the right side
    ax.legend(
        title="Expertise Level", 
        bbox_to_anchor=(0.96, 0.92), 
        loc="center right", 
        bbox_transform=fig.transFigure, 
        frameon=True, 
        fontsize=LEGEND_FONT_SIZE, 
        ncol=2 
    )
    
    # Consistent Stats Banner
    if stat_box:
        # 1. Place "Group significance:" alone on the left (where legend used to be)
        fig.text(
            0.01, 0.92, "Group significance:", 
            ha='left', va='center', fontsize=STAT_HEADER_FONT_SIZE, fontweight='bold'
        )

        # 2. Render the stats inside the box, centered
        stat_string = "   •   ".join([f"{k}: {v}" for k, v in stat_box.items()]).replace('\n', ' ')
        
        fig.text(
            0.5, 0.92, stat_string, 
            ha='center', va='center', fontsize=STAT_TEXT_FONT_SIZE, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', edgecolor='darkgray', alpha=1.0)
        )

        plt.tight_layout(rect=[0, 0.02, 1.0, 1.0])
    else:
        plt.tight_layout(rect=[0, 0.02, 1.0, 0.95])
        
    return fig, ax


def plot_mot_feature_old(
    df, feature, y_label, plot_title, stat_box=None,
    condition_col='condition', group_col='group', participant_col='participant_id'
):
    condition_mapping = {
        0: "(4, 11)", 1: "(4, 13)", 2: "(4, 15)", 3: "(4, 17)", 
        4: "(4, 19)", 5: "(5, 10)", 6: "(5, 12)", 7: "(6, 9)", 
        8: "(6, 11)", 9: "(7, 8)", 10: "(7, 10)", 11: "(4, 13)"
    }
    
    df_subj = df.groupby(
        [participant_col, group_col, condition_col]
    )[feature].mean().reset_index()
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 6)) 
    
    x_order = list(condition_mapping.keys())
    
    sns.pointplot(
        data=df_subj, x=condition_col, y=feature, hue=group_col, order=x_order,
        ax=ax, dodge=0.2, markers=['o', 's'], err_kws={'linewidth': 2}, capsize=.05,
        palette={'Novices': 'darkblue', 'ATCs': 'darkorange'}, 
        linestyles=["-", "--"]
    )
    
    ax.set_xticklabels([condition_mapping[cond] for cond in x_order], fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
    
    # Consistent Y and X labels
    ax.set_xlabel("(Number of Targets, Number of Distractors)", labelpad=12, fontsize=X_FONT_SIZE)
    ax.set_ylabel(y_label, labelpad=12, fontsize=Y_FONT_SIZE)
    
    # Title restored to its original height to maintain space
    plt.suptitle(plot_title, fontsize=SUPTITLE_FONT_SIZE, y=1.05, fontweight='bold')
    
    # Legend moved to the right side
    ax.legend(
        title="Expertise Level", 
        bbox_to_anchor=(0.96, 0.92), 
        loc="center right", 
        bbox_transform=fig.transFigure, 
        frameon=True, 
        fontsize=LEGEND_FONT_SIZE, 
        ncol=2 
    )
    
    # Consistent Stats Banner
    if stat_box:
        # 1. Place "Group significance:" alone on the left (where legend used to be)
        fig.text(
            0.01, 0.92, "Group significance:", 
            ha='left', va='center', fontsize=STAT_HEADER_FONT_SIZE, fontweight='bold'
        )

        # 2. Render the stats inside the box, centered
        stat_string = "   •   ".join([f"{k}: {v}" for k, v in stat_box.items()]).replace('\n', ' ')
        
        fig.text(
            0.5, 0.92, stat_string, 
            ha='center', va='center', fontsize=STAT_TEXT_FONT_SIZE, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', edgecolor='darkgray', alpha=1.0)
        )

        plt.tight_layout(rect=[0, 0.02, 1.0, 1.0])
    else:
        plt.tight_layout(rect=[0, 0.02, 1.0, 0.95])
    return fig, ax



from scipy.stats import ttest_ind


def plot_aggregated_feature(df, feature_col, y_label, plot_title, group_col='group',participant_col = "participant_id"):
    """
    Plots an aggregated feature with a boxplot and stripplot, 
    and displays the p-value and Cohen's d cleanly in the title.
    """
    df = df.groupby([participant_col, group_col])[feature_col].mean().reset_index()

    # 1. Statistical Test Setup
    novices = df[df[group_col] == 'Novices'][feature_col].dropna()
    atcs = df[df[group_col] == 'ATCs'][feature_col].dropna()
    
    # Welch's t-test (unequal variances)
    t_stat, p_val = ttest_ind(novices, atcs, equal_var=False)

    # Cohen's d calculation
    n1, n2 = len(novices), len(atcs)
    var1, var2 = np.var(novices, ddof=1), np.var(atcs, ddof=1)
    
    # Calculate pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Calculate effect size (absolute magnitude)
    cohens_d = np.abs(np.mean(novices) - np.mean(atcs)) / pooled_sd

    # Determine significance string and exact p-value formatting
    if p_val < 0.001:
        sig_symbol = "***"
        p_str = "p < .001"
    elif p_val <= 0.01:
        sig_symbol = "**"
        p_str = f"p = {p_val:.3f}"
    elif p_val <= 0.05:
        sig_symbol = "*"
        p_str = f"p = {p_val:.3f}"
    else:
        sig_symbol = "ns"
        p_str = f"p = {p_val:.3f}"

    # 2. Plotting Setup
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {'Novices': 'darkblue', 'ATCs': 'darkorange'}

    # Create Boxplot and overlay Stripplot for individual points
    sns.boxplot(
        data=df, x=group_col, y=feature_col, order=['Novices', 'ATCs'], ax=ax,
        palette=palette, width=0.4, showfliers=False, boxprops={'alpha': 0.7},
        showmeans=True, 
        meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"}
    )
    sns.stripplot(
        data=df, x=group_col, y=feature_col, order=['Novices', 'ATCs'],
        color='black', alpha=0.6, jitter=True, size=6, ax=ax
    )

    # 3. Formatting
    # Append both the p-value and Cohen's d to the subtitle
    full_title = f"{plot_title}\nWelch's t-test: ${p_str}$, $d = {cohens_d:.2f}$ ({sig_symbol})"
    ax.set_title(full_title, fontsize=SUBTITLE_FONT_SIZE, pad=15)
    
    ax.set_xlabel("Expertise Level", labelpad=12, fontsize=X_FONT_SIZE)
    ax.set_ylabel(y_label, labelpad=12, fontsize=Y_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)

    plt.tight_layout()
    print(f"Means - Novices: {np.mean(novices):.3f}, ATCs: {np.mean(atcs):.3f} | Linear Diff: {np.mean(novices) - np.mean(atcs):.3f} | % Diff: {((np.mean(novices) - np.mean(atcs)) / np.mean(novices)) * 100:.2f}%")
    return fig, ax, df



import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

def plot_bimodal_saccade_distribution(df, thresholds_dict, duration_col='duration_sec', group_col='group'):
    # Slightly wider figure to give the text boxes room to breathe
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.set_theme(style="whitegrid", context="talk") 
    
    palette = {'Novices': 'darkblue', 'ATCs': 'darkorange'}
    max_dur_sec = 0.15 
    
    plot_df = df[df[duration_col] <= max_dur_sec].copy()
    plot_df['duration_ms'] = plot_df[duration_col] * 1000
    
    # Set legend=False here to manually build a bulletproof legend later
    sns.kdeplot(
        data=plot_df, x='duration_ms', hue=group_col, palette=palette,
        fill=True, alpha=0.35, linewidth=2.5, common_norm=False, legend=False
    )
    
    groups = ['ATCs', 'Novices']
    y_max = ax.get_ylim()[1]
    
    for i, grp in enumerate(groups):
        grp_data_sec = df[(df[group_col] == grp) & (df[duration_col] <= max_dur_sec)][duration_col].dropna()
        if grp_data_sec.empty: continue
            
        color = palette[grp]
        lm_thresh_sec = thresholds_dict.get(grp, 0.032)
        lm_thresh_ms = lm_thresh_sec * 1000
        
        pct_short = (grp_data_sec <= lm_thresh_sec).mean() * 100
        pct_long = (grp_data_sec > lm_thresh_sec).mean() * 100
        
        kde = gaussian_kde(grp_data_sec)
        x_eval = np.linspace(0, max_dur_sec, 500)
        y_eval = kde(x_eval)
        peaks, _ = find_peaks(y_eval)
        
        peak_short, peak_long = np.nan, np.nan
        if len(peaks) > 0:
            short_peaks = peaks[x_eval[peaks] < lm_thresh_sec]
            long_peaks = peaks[x_eval[peaks] > lm_thresh_sec]
            
            if len(short_peaks) > 0:
                peak_short = x_eval[short_peaks[np.argmax(y_eval[short_peaks])]] * 1000
            if len(long_peaks) > 0:
                peak_long = x_eval[long_peaks[np.argmax(y_eval[long_peaks])]] * 1000
        
        # Thicker threshold lines for visibility
        plt.axvline(lm_thresh_ms, color=color, linestyle='--', alpha=0.9, linewidth=2)
        
        # Streamlined text
        annot_text = (
            f"$\\bf{{{grp}}}$\n"
            f"$L_m$: {lm_thresh_ms:.1f} ms\n"
            f"Short (≤$L_m$): {pct_short:.1f}% | Peak: {peak_short:.1f} ms\n"
            f"Long (>$L_m$): {pct_long:.1f}% | Peak: {peak_long:.1f} ms"
        )
        
        # Shifted to the right (starting at 60% of the x-axis)
        box_y = y_max * (0.70 - (i * 0.35))
        plt.text(
            max_dur_sec * 1000 * 0.55, box_y, annot_text, 
            color=color, fontsize=STAT_TEXT_FONT_SIZE, linespacing=1.5,
            bbox=dict(facecolor='white', alpha=0.95, edgecolor=color, boxstyle='round,pad=0.5')
        )
        
    plt.title("Bimodal Distribution of Saccade Durations: ATCs vs. Novices", fontsize=SUPTITLE_FONT_SIZE, pad=20)
    plt.xlabel("Saccade Duration (ms)", labelpad=12, fontsize=X_FONT_SIZE)
    plt.ylabel("Probability Density", labelpad=12, fontsize=Y_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    plt.xlim(0, max_dur_sec * 1000)
    
    # Custom Legend using Proxy Artists
    handles = [
        mpatches.Patch(facecolor=palette[g], edgecolor=palette[g], label=g, alpha=0.5, linewidth=2) 
        for g in groups
    ]
    
    # Explicitly pass labels=groups and use ax.legend
    ax.legend(
        handles=handles, 
        labels=groups, 
        title="Expertise Level", 
        bbox_to_anchor=(1.02, 1), 
        loc="upper left", 
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE
    )
    
    plt.tight_layout()
    return fig, ax




def plot_phase_subplots_split_stats(
    df, feature, y_label, plot_title, 
    stat_box_phase1=None, stat_box_phase23=None,
    condition_col='condition', group_col='group', participant_col='participant_id'
):
    """
    Creates 3 side-by-side subplots (one for each phase) with split statistical banners. 
    One banner aligns over Phase 1, and a second banner aligns over Phases 2 & 3.
    """
    condition_mapping = {
        "Cond_0": {'Phase': '1: Color', 'Set_Size': 8},
        "Cond_1": {'Phase': '1: Color', 'Set_Size': 32},
        "Cond_2": {'Phase': '1: Color', 'Set_Size': 66},
        "Cond_3": {'Phase': '2: Shape', 'Set_Size': 8},
        "Cond_4": {'Phase': '2: Shape', 'Set_Size': 32},
        "Cond_5": {'Phase': '2: Shape', 'Set_Size': 66},
        "Cond_6": {'Phase': '3: Conjunction', 'Set_Size': 8},
        "Cond_7": {'Phase': '3: Conjunction', 'Set_Size': 32},
        "Cond_8": {'Phase': '3: Conjunction', 'Set_Size': 66}
    }
    
    df_plot = df.copy()
    df_plot['Phase'] = df_plot[condition_col].map(lambda c: condition_mapping.get(c, {}).get('Phase'))
    df_plot['Set_Size'] = df_plot[condition_col].map(lambda c: condition_mapping.get(c, {}).get('Set_Size'))
    
    df_subj = df_plot.groupby(
        [participant_col, group_col, 'Phase', 'Set_Size']
    )[feature].mean().reset_index()
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    
    phases = ['1: Color', '2: Shape', '3: Conjunction']
    
    for i, phase in enumerate(phases):
        ax = axes[i]
        phase_data = df_subj[df_subj['Phase'] == phase]
        
        sns.pointplot(
            data=phase_data, x='Set_Size', y=feature, hue=group_col,
            ax=ax, dodge=0.1, markers=['o', 's'], err_kws={'linewidth': 2}, capsize=.05,
            palette={'Novices': 'darkblue', 'ATCs': 'darkorange'}, 
            linestyles=["-", "--"]
        )
        
        ax.set_title(f"Phase {phase}", fontsize=SUBTITLE_FONT_SIZE, pad=10)
        ax.set_xlabel('')
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        
        # Original Legend Logic
        if i == 0:
            ax.set_ylabel(y_label, labelpad=12, fontsize=Y_FONT_SIZE)
            ax.legend(title="Expertise Level", loc="upper left", frameon=True, fontsize=LEGEND_FONT_SIZE)
        else:
            ax.set_ylabel('')
            if ax.get_legend():
                ax.get_legend().remove()
                
    # Title
    plt.suptitle(plot_title, fontsize=SUPTITLE_FONT_SIZE, y=1.05, fontweight='bold')
    fig.text(0.5, 0.02, 'Set Size', ha='center', va='center', fontsize=X_FONT_SIZE)
    
    if stat_box_phase1 or stat_box_phase23:
        fig.text(
            0.01, 0.92, "Group significance:", 
            ha='left', va='center', fontsize=STAT_HEADER_FONT_SIZE, fontweight='bold'
        )

        # 1. Box for Phase 1 (Centered around x=0.3)
        if stat_box_phase1:
            stat_str_1 = "   •   ".join([f"{k}: {v}" for k, v in stat_box_phase1.items()]).replace('\n', ' ')
            fig.text(
                0.28, 0.92, stat_str_1, 
                ha='center', va='center', fontsize=STAT_TEXT_FONT_SIZE, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', edgecolor='darkgray', alpha=1.0)
            )

        # 2. Box for Phases 2 & 3 (Centered around x=0.73)
        if stat_box_phase23:
            stat_str_23 = "   •   ".join([f"{k}: {v}" for k, v in stat_box_phase23.items()]).replace('\n', ' ')
            fig.text(
                0.7, 0.92, stat_str_23, 
                ha='center', va='center', fontsize=STAT_TEXT_FONT_SIZE, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', edgecolor='darkgray', alpha=1.0)
            )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    else:
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig, axes


from scipy.stats import lognorm
import matplotlib.patches as mpatches

def plot_lognormal_fixation_distribution(df, duration_col='duration_sec', group_col='group'):
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.set_theme(style="whitegrid", context="talk") 
    
    palette = {'Novices': 'darkblue', 'ATCs': 'darkorange'}
    max_dur_sec = 1.5  # Typical cap for analyzing standard fixations
    
    plot_df = df[df[duration_col] <= max_dur_sec].copy()
    plot_df['duration_ms'] = plot_df[duration_col] * 1000
    max_dur_ms = max_dur_sec * 1000
    
    groups = ['ATCs', 'Novices']
    x_eval = np.linspace(0.001, max_dur_ms, 1000)
    
    # Store max y-value dynamically to position text boxes
    y_max = 0

    for i, grp in enumerate(groups):
        grp_data_ms = plot_df[plot_df[group_col] == grp]['duration_ms'].dropna()
        if grp_data_ms.empty: continue
            
        color = palette[grp]
        
        # 1. Plot Empirical KDE
        sns.kdeplot(
            data=grp_data_ms, color=color,
            fill=True, alpha=0.25, linewidth=0, label=f"{grp} (Empirical)", ax=ax
        )
        
        # 2. Fit Log-Normal Distribution
        # floc=0 anchors the lower bound at 0ms, which is necessary for eye movement data
        shape, loc, scale = lognorm.fit(grp_data_ms, floc=0)
        
        # 3. Calculate and Plot Theoretical PDF
        pdf_fitted = lognorm.pdf(x_eval, shape, loc=loc, scale=scale)
        ax.plot(x_eval, pdf_fitted, color=color, linestyle='--', linewidth=2.5, label=f"{grp} (Log-Norm Fit)")
        
        y_max = max(y_max, max(pdf_fitted))
        
        # 4. Extract Parameters for Annotation
        mu_fit = np.log(scale)
        sigma_fit = shape
        median_ms = np.median(grp_data_ms)
        
        annot_text = (
            f"$\\bf{{{grp}}}$\n"
            f"Median: {median_ms:.0f} ms\n"
            f"Fit $\\mu$: {mu_fit:.2f}\n"
            f"Fit $\\sigma$: {sigma_fit:.2f}"
        )
        
        box_y = y_max * (0.85 - (i * 0.35))
        ax.text(
            max_dur_ms * 0.65, box_y, annot_text, 
            color=color, fontsize=STAT_TEXT_FONT_SIZE, linespacing=1.5,
            bbox=dict(facecolor='white', alpha=0.95, edgecolor=color, boxstyle='round,pad=0.5')
        )
        
    ax.set_title("Log-Normal Distribution of Fixation Durations: ATCs vs. Novices", fontsize=SUPTITLE_FONT_SIZE, pad=20)
    ax.set_xlabel("Fixation Duration (ms)", labelpad=12, fontsize=X_FONT_SIZE)
    ax.set_ylabel("Probability Density", labelpad=12, fontsize=Y_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    ax.set_xlim(0, max_dur_ms)
    
    # Legend formatting
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, 
        title="Expertise & Fit", 
        bbox_to_anchor=(1.02, 1), 
        loc="upper left", 
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE
    )
    
    plt.tight_layout()
    return fig, ax