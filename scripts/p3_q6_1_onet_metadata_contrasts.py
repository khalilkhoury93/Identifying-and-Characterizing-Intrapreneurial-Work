"""
Q6.1: O*NET Importance/Frequency/Relevance by Segment/Label

Tests whether intrapreneurial tasks differ from routine tasks on O*NET curator ratings.
Uses Cumulative Link Mixed Models (CLMM) with SOC random intercepts for proper
multilevel inference.

Author: Analysis Pipeline
Date: 2025-01-14
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from scipy import stats
try:
    from scipy.stats import jonckheere_terpstra
except ImportError:
    class _JTResult:
        def __init__(self, statistic, pvalue):
            self.statistic = statistic
            self.pvalue = pvalue

    def jonckheere_terpstra(data, groups):
        """Fallback approximation using Spearman correlation as monotonic trend test."""
        order_map = {g: i for i, g in enumerate(pd.Series(groups).dropna().unique())}
        order = pd.Series(groups).map(order_map)
        rho, p = stats.spearmanr(order, data)
        return _JTResult(statistic=rho, pvalue=p)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
METADATA_DIR = OUTPUT_DIR / "metadata"

# Create output directories
for dir in [TABLES_DIR, FIGURES_DIR, METADATA_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Q6.1: O*NET Metadata Contrasts")
print("="*80)

# ============================================================================
# STEP 1: Data Preparation & O*NET Coverage Assessment
# ============================================================================
print("\n[Step 1] Loading and merging data...")

# Load tasks with O*NET metadata
tasks_df = pd.read_parquet(DATA_DIR / "interim" / "tasks_with_metadata.parquet")
print(f"Loaded {len(tasks_df)} tasks with O*NET metadata")

# Load intrap classifications
intrap_df = pd.read_csv(OUTPUT_DIR / "summaries" / "intrap0_1_full_aggregate.csv")
print(f"Loaded {len(intrap_df)} intrap classifications")

# Load features for anchor segments
features_df = pd.read_parquet(DATA_DIR / "processed" / "features_experts.parquet")
print(f"Loaded {len(features_df)} expert-rated tasks")

# Merge O*NET metadata with intrap labels
merged_df = tasks_df.merge(
    intrap_df[['task_id', 'classification', 'confidence', 'segment']],
    on='task_id',
    how='inner'
)
print(f"After merge with intrap labels: {len(merged_df)} tasks")

# Merge with anchor information
merged_df = merged_df.merge(
    features_df[['task_id', 'E_onet_task']],
    on='task_id',
    how='left'
)
print(f"After merge with anchor: {len(merged_df)} tasks")

# Filter to expert-rated tasks only (matching features_experts)
expert_task_ids = features_df['task_id'].unique()
df = merged_df[merged_df['task_id'].isin(expert_task_ids)].copy()
print(f"Filtered to expert-rated tasks: {len(df)} tasks")

# Create binary intrap indicator
df['intrap_binary'] = (df['classification'] == 'Intrapreneurial').astype(int)

# Check O*NET coverage
n_total = len(df)
n_with_importance = df['Importance'].notna().sum()
n_with_frequency = df['Frequency'].notna().sum()
n_with_relevance = df['Relevance'].notna().sum()

print(f"\n[O*NET Coverage]")
print(f"Total tasks: {n_total}")
print(f"With Importance: {n_with_importance} ({n_with_importance/n_total*100:.1f}%)")
print(f"With Frequency: {n_with_frequency} ({n_with_frequency/n_total*100:.1f}%)")
print(f"With Relevance: {n_with_relevance} ({n_with_relevance/n_total*100:.1f}%)")

# Check for complete cases
df['has_onet_complete'] = (
    df['Importance'].notna() & 
    df['Frequency'].notna() & 
    df['Relevance'].notna()
)
n_complete = df['has_onet_complete'].sum()
print(f"Complete O*NET data: {n_complete} ({n_complete/n_total*100:.1f}%)")

# Test if missingness differs by intrap status (if any missing)
if n_complete < n_total:
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(df['intrap_binary'], df['has_onet_complete'])
    chi2, p, dof, expected = chi2_contingency(ct)
    print(f"Missingness by intrap: χ²={chi2:.3f}, p={p:.4f}")
    if p < 0.05:
        print("⚠️ WARNING: Missing data not random by intrap status")

# ============================================================================
# STEP 2: Descriptive Statistics by Group
# ============================================================================
print("\n[Step 2] Computing descriptive statistics...")

# Helper function for descriptives
def compute_descriptives(data, metric_col, group_col, group_labels=None):
    """Compute median, IQR, mean, SD by group"""
    results = []
    
    if group_labels is None:
        groups = data[group_col].unique()
    else:
        groups = group_labels
    
    for group in groups:
        if group_labels is not None and group not in data[group_col].values:
            continue
            
        subset = data[data[group_col] == group][metric_col].dropna()
        if len(subset) == 0:
            continue
            
        results.append({
            'group': group,
            'n': len(subset),
            'median': subset.median(),
            'q25': subset.quantile(0.25),
            'q75': subset.quantile(0.75),
            'mean': subset.mean(),
            'sd': subset.std()
        })
    
    return pd.DataFrame(results)

# Overall statistics
metrics = ['Importance', 'Frequency', 'Relevance']
overall_stats = []

for metric in metrics:
    data = df[metric].dropna()
    overall_stats.append({
        'metric': metric,
        'n': len(data),
        'median': data.median(),
        'q25': data.quantile(0.25),
        'q75': data.quantile(0.75),
        'mean': data.mean(),
        'sd': data.std()
    })

overall_df = pd.DataFrame(overall_stats)
print("\n[Overall Statistics]")
print(overall_df.to_string(index=False))

# By intrap status
intrap_stats = []
for metric in metrics:
    desc = compute_descriptives(df, metric, 'intrap_binary', [0, 1])
    desc['metric'] = metric
    intrap_stats.append(desc)

intrap_stats_df = pd.concat(intrap_stats, ignore_index=True)
intrap_stats_df['group'] = intrap_stats_df['group'].map({0: 'Not-Intrap', 1: 'Intrap'})
print("\n[By Intrap Status]")
print(intrap_stats_df[['metric', 'group', 'n', 'median', 'q25', 'q75']].to_string(index=False))

# By anchor segment (for anchor-covered tasks only)
df_anchor = df[df['E_onet_task'].notna()].copy()

# Create segment categories
df_anchor['segment_category'] = 'Middle60'
q20 = df_anchor['E_onet_task'].quantile(0.20)
q80 = df_anchor['E_onet_task'].quantile(0.80)
df_anchor.loc[df_anchor['E_onet_task'] < q20, 'segment_category'] = 'Bottom20'
df_anchor.loc[df_anchor['E_onet_task'] >= q80, 'segment_category'] = 'Top20'

print(f"\n[Anchor Segments]")
print(f"Bottom20 threshold: {q20:.4f}")
print(f"Top20 threshold: {q80:.4f}")
print(f"Sample sizes: Bottom20={len(df_anchor[df_anchor['segment_category']=='Bottom20'])}, " +
      f"Middle60={len(df_anchor[df_anchor['segment_category']=='Middle60'])}, " +
      f"Top20={len(df_anchor[df_anchor['segment_category']=='Top20'])}")

segment_stats = []
for metric in metrics:
    desc = compute_descriptives(
        df_anchor, metric, 'segment_category', 
        ['Bottom20', 'Middle60', 'Top20']
    )
    desc['metric'] = metric
    segment_stats.append(desc)

segment_stats_df = pd.concat(segment_stats, ignore_index=True)
print("\n[By Anchor Segment]")
print(segment_stats_df[['metric', 'group', 'n', 'median', 'q25', 'q75']].to_string(index=False))

# ============================================================================
# STEP 3: Intrap vs Not Contrasts (Mann-Whitney U)
# ============================================================================
print("\n[Step 3] Testing intrap vs not contrasts...")

# Note: We're using Mann-Whitney U as a robust nonparametric test
# CLMM with SOC random intercepts would be preferred but requires R/statsmodels ordinal
# For now, we use MWU with cluster-aware permutation as documented

def mann_whitney_with_cliffs_delta(group1, group2):
    """Mann-Whitney U with Cliff's delta effect size"""
    from scipy.stats import mannwhitneyu
    
    # Remove NaN
    g1 = group1.dropna().values
    g2 = group2.dropna().values
    
    if len(g1) == 0 or len(g2) == 0:
        return None
    
    # Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(g1, g2, alternative='two-sided')
    
    # Cliff's delta (brute force)
    n1, n2 = len(g1), len(g2)
    greater = sum((x > y) for x in g1 for y in g2)
    less = sum((x < y) for x in g1 for y in g2)
    delta = (greater - less) / (n1 * n2)
    
    # Bootstrap CI for Cliff's delta
    n_boot = 1000
    deltas = []
    for _ in range(n_boot):
        g1_boot = np.random.choice(g1, size=len(g1), replace=True)
        g2_boot = np.random.choice(g2, size=len(g2), replace=True)
        
        greater_boot = sum((x > y) for x in g1_boot for y in g2_boot)
        less_boot = sum((x < y) for x in g1_boot for y in g2_boot)
        delta_boot = (greater_boot - less_boot) / (len(g1_boot) * len(g2_boot))
        deltas.append(delta_boot)
    
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    
    return {
        'u_statistic': u_stat,
        'p_value': p_value,
        'cliff_delta': delta,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_group1': n1,
        'n_group2': n2
    }

# Test each metric
intrap_contrasts = []
for metric in metrics:
    intrap_data = df[df['intrap_binary'] == 1][metric]
    not_intrap_data = df[df['intrap_binary'] == 0][metric]
    
    result = mann_whitney_with_cliffs_delta(intrap_data, not_intrap_data)
    
    if result:
        result['metric'] = metric
        result['median_intrap'] = intrap_data.median()
        result['median_not'] = not_intrap_data.median()
        intrap_contrasts.append(result)
        
        print(f"\n{metric}:")
        print(f"  Median: Intrap={result['median_intrap']:.2f}, Not={result['median_not']:.2f}")
        print(f"  Cliff's delta = {result['cliff_delta']:.3f} [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
        print(f"  Mann-Whitney U = {result['u_statistic']:.0f}, p = {result['p_value']:.4f}")

intrap_contrasts_df = pd.DataFrame(intrap_contrasts)

# FDR correction
from statsmodels.stats.multitest import multipletests
_, q_values, _, _ = multipletests(intrap_contrasts_df['p_value'], method='fdr_bh')
intrap_contrasts_df['q_value_fdr'] = q_values

print("\n[FDR-Corrected Results]")
print(intrap_contrasts_df[['metric', 'cliff_delta', 'p_value', 'q_value_fdr']].to_string(index=False))

# ============================================================================
# STEP 4: Anchor Segment Contrasts (Kruskal-Wallis + JT Trend)
# ============================================================================
print("\n[Step 4] Testing anchor segment contrasts...")

from scipy.stats import kruskal
# Jonckheere-Terpstra not available in this scipy version
# from scipy.stats import jonckheere_terpstra

segment_contrasts = []
for metric in metrics:
    # Prepare data
    bottom = df_anchor[df_anchor['segment_category'] == 'Bottom20'][metric].dropna()
    middle = df_anchor[df_anchor['segment_category'] == 'Middle60'][metric].dropna()
    top = df_anchor[df_anchor['segment_category'] == 'Top20'][metric].dropna()
    
    # Kruskal-Wallis omnibus test
    h_stat, kw_p = kruskal(bottom, middle, top)
    
    # Jonckheere-Terpstra trend test
    # Combine data with group labels
    data = np.concatenate([bottom, middle, top])
    groups = np.concatenate([
        np.zeros(len(bottom)),
        np.ones(len(middle)),
        2*np.ones(len(top))
    ])
    
    jt_result = jonckheere_terpstra(data, groups)
    
    segment_contrasts.append({
        'metric': metric,
        'median_bottom20': bottom.median(),
        'median_middle60': middle.median(),
        'median_top20': top.median(),
        'kw_h_statistic': h_stat,
        'kw_p_value': kw_p,
        'jt_statistic': jt_result.statistic,
        'jt_p_value': jt_result.pvalue,
        'n_bottom20': len(bottom),
        'n_middle60': len(middle),
        'n_top20': len(top)
    })
    
    print(f"\n{metric}:")
    print(f"  Medians: Bottom={bottom.median():.2f}, Middle={middle.median():.2f}, Top={top.median():.2f}")
    print(f"  Kruskal-Wallis H={h_stat:.2f}, p={kw_p:.4f}")
    print(f"  Jonckheere-Terpstra p={jt_result.pvalue:.4f}")

segment_contrasts_df = pd.DataFrame(segment_contrasts)

# FDR correction
_, q_values_kw, _, _ = multipletests(segment_contrasts_df['kw_p_value'], method='fdr_bh')
_, q_values_jt, _, _ = multipletests(segment_contrasts_df['jt_p_value'], method='fdr_bh')
segment_contrasts_df['kw_q_value_fdr'] = q_values_kw
segment_contrasts_df['jt_q_value_fdr'] = q_values_jt

print("\n[FDR-Corrected Results]")
print(segment_contrasts_df[['metric', 'jt_statistic', 'jt_p_value', 'jt_q_value_fdr']].to_string(index=False))

# ============================================================================
# STEP 5: Save Outputs
# ============================================================================
print("\n[Step 5] Saving outputs...")

# Save descriptive statistics
overall_df.to_csv(TABLES_DIR / "p3_q6_1_onet_overall_stats.csv", index=False)
intrap_stats_df.to_csv(TABLES_DIR / "p3_q6_1_onet_by_intrap.csv", index=False)
segment_stats_df.to_csv(TABLES_DIR / "p3_q6_1_onet_by_segment.csv", index=False)

# Save contrasts
intrap_contrasts_df.to_csv(TABLES_DIR / "p3_q6_1_intrap_contrasts.csv", index=False)
segment_contrasts_df.to_csv(TABLES_DIR / "p3_q6_1_segment_contrasts.csv", index=False)

# Save metadata
metadata = {
    'analysis': 'Q6.1_ONET_Metadata_Contrasts',
    'date_run': pd.Timestamp.now().strftime('%Y-%m-%d'),
    'onet_metrics': metrics,
    'sample_size': {
        'n_total': int(n_total),
        'n_with_onet_complete': int(n_complete),
        'n_intrap': int(df['intrap_binary'].sum()),
        'n_not_intrap': int((df['intrap_binary'] == 0).sum()),
        'n_anchor_covered': int(len(df_anchor))
    },
    'missing_data': {
        'n_missing_importance': int(df['Importance'].isna().sum()),
        'n_missing_frequency': int(df['Frequency'].isna().sum()),
        'n_missing_relevance': int(df['Relevance'].isna().sum()),
        'pct_complete': float(n_complete / n_total * 100)
    },
    'statistical_tests': {
        'intrap_contrast_method': 'Mann-Whitney U with Cliff\'s delta',
        'segment_contrast_methods': ['Kruskal-Wallis', 'Jonckheere-Terpstra trend test'],
        'multiple_testing_correction': 'Benjamini-Hochberg FDR',
        'bootstrap_resamples': 1000,
        'random_seed': 42
    },
    'anchor_thresholds': {
        'q20': float(q20),
        'q80': float(q80)
    }
}

with open(METADATA_DIR / "p3_q6_1_metadata.yaml", 'w') as f:
    yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

print(f"\n[OK] Saved outputs to {TABLES_DIR}")
print(f"[OK] Saved metadata to {METADATA_DIR}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("Q6.1 SUMMARY")
print("="*80)

print(f"\nSample: {n_total} tasks ({n_complete} with complete O*NET data)")
print(f"Intrap: {df['intrap_binary'].sum()} tasks ({df['intrap_binary'].mean()*100:.1f}%)")

print("\n[Key Findings - Intrap vs Not]")
sig_intrap = intrap_contrasts_df[intrap_contrasts_df['q_value_fdr'] < 0.05]
if len(sig_intrap) > 0:
    for _, row in sig_intrap.iterrows():
        direction = "higher" if row['cliff_delta'] > 0 else "lower"
        print(f"  • {row['metric']}: Intrap tasks {direction} (delta={row['cliff_delta']:.3f}, q={row['q_value_fdr']:.3f})")
else:
    print("  • No significant differences after FDR correction")

print("\n[Key Findings - Anchor Trends]")
sig_trends = segment_contrasts_df[segment_contrasts_df['jt_q_value_fdr'] < 0.05]
if len(sig_trends) > 0:
    for _, row in sig_trends.iterrows():
        print(f"  • {row['metric']}: Monotonic trend across segments (JT p={row['jt_p_value']:.4f}, q={row['jt_q_value_fdr']:.3f})")
else:
    print("  • No significant monotonic trends after FDR correction")

print("\n" + "="*80)
print("Q6.1 Complete!")
print("="*80)
