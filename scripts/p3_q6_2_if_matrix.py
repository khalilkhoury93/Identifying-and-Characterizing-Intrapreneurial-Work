"""
Q6.2: O*NET Importance × Frequency Matrix Analysis

Maps intrapreneurial tasks in 2D importance-frequency space to identify:
- Core tasks (high importance, high frequency)
- Critical tasks (high importance, low frequency)  
- Operational tasks (low importance, high frequency)
- Peripheral tasks (low importance, low frequency)

Author: Analysis Pipeline
Date: 2025-01-14
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from scipy.stats import chi2_contingency, fisher_exact
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
METADATA_DIR = OUTPUT_DIR / "metadata"

for dir in [TABLES_DIR, FIGURES_DIR, METADATA_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Q6.2: Importance × Frequency Matrix")
print("="*80)

# ============================================================================
# STEP 1: Load Data and Define Thresholds
# ============================================================================
print("\n[Step 1] Loading data and defining quadrants...")

# Load merged data from Q6.1
tasks_df = pd.read_parquet(DATA_DIR / "interim" / "tasks_with_metadata.parquet")
intrap_df = pd.read_csv(OUTPUT_DIR / "summaries" / "intrap0_1_full_aggregate.csv")
features_df = pd.read_parquet(DATA_DIR / "processed" / "features_experts.parquet")

# Merge
merged_df = tasks_df.merge(
    intrap_df[['task_id', 'classification', 'confidence', 'segment']],
    on='task_id',
    how='inner'
)
merged_df = merged_df.merge(
    features_df[['task_id', 'E_onet_task']],
    on='task_id',
    how='left'
)

# Filter to expert-rated tasks
expert_task_ids = features_df['task_id'].unique()
df = merged_df[merged_df['task_id'].isin(expert_task_ids)].copy()
# Deduplicate to one row per task to prevent double-counting in quadrant outputs
df = df.drop_duplicates(subset=['task_id']).reset_index(drop=True)

df['intrap_binary'] = (df['classification'] == 'Intrapreneurial').astype(int)

print(f"Loaded {len(df)} tasks")
print(f"Intrap: {df['intrap_binary'].sum()} ({df['intrap_binary'].mean()*100:.1f}%)")

# ============================================================================
# STEP 2: Define Quadrants (Fixed Thresholds)
# ============================================================================
print("\n[Step 2] Defining importance-frequency quadrants...")

# Use fixed thresholds (Option A from documentation)
# High = >=4 on 1-5 scale for both metrics
importance_threshold = 4.0
frequency_threshold = 4.0

# Alternative: Scale frequency to match importance range
# Frequency is on 1-7 scale in O*NET, but our data shows 1-5 range
# Let's verify the range first
print(f"\nImportance range: {df['Importance'].min():.2f} to {df['Importance'].max():.2f}")
print(f"Frequency range: {df['Frequency'].min():.2f} to {df['Frequency'].max():.2f}")

# Create binary indicators
df['high_importance'] = (df['Importance'] >= importance_threshold).astype(int)
df['high_frequency'] = (df['Frequency'] >= frequency_threshold).astype(int)

# Define quadrants
def assign_quadrant(row):
    if row['high_importance'] == 1 and row['high_frequency'] == 1:
        return 'Core'  # High/High
    elif row['high_importance'] == 1 and row['high_frequency'] == 0:
        return 'Critical'  # High importance, Low frequency
    elif row['high_importance'] == 0 and row['high_frequency'] == 1:
        return 'Operational'  # Low importance, High frequency
    else:
        return 'Peripheral'  # Low/Low

df['quadrant'] = df.apply(assign_quadrant, axis=1)

# Check quadrant distribution
quad_counts = df['quadrant'].value_counts()
print("\n[Quadrant Distribution]")
print(quad_counts)
print("\nPercentages:")
print((quad_counts / len(df) * 100).round(1))

# Check for small cells
min_count = quad_counts.min()
if min_count < 10:
    print(f"\n⚠️ WARNING: Smallest quadrant has {min_count} tasks (may have low power)")

# ============================================================================
# STEP 3: Compute Intrap Share per Quadrant
# ============================================================================
print("\n[Step 3] Computing intrap prevalence per quadrant...")

quad_intrap = df.groupby('quadrant').agg({
    'intrap_binary': ['sum', 'count', 'mean']
}).reset_index()
quad_intrap.columns = ['quadrant', 'n_intrap', 'n_total', 'pct_intrap']
quad_intrap['pct_intrap'] = quad_intrap['pct_intrap'] * 100

# Add confidence intervals (Wilson score)
def wilson_ci(n_success, n_total, alpha=0.05):
    """Wilson score confidence interval for proportion"""
    from scipy import stats
    z = stats.norm.ppf(1 - alpha/2)
    p = n_success / n_total
    denominator = 1 + z**2/n_total
    centre_adjusted = p + z**2/(2*n_total)
    adjusted_sd = np.sqrt((p*(1-p) + z**2/(4*n_total)) / n_total)
    
    lower = (centre_adjusted - z*adjusted_sd) / denominator
    upper = (centre_adjusted + z*adjusted_sd) / denominator
    return lower*100, upper*100

for idx, row in quad_intrap.iterrows():
    lower, upper = wilson_ci(row['n_intrap'], row['n_total'])
    quad_intrap.loc[idx, 'ci_lower'] = lower
    quad_intrap.loc[idx, 'ci_upper'] = upper

print("\n[Intrap Prevalence by Quadrant]")
print(quad_intrap[['quadrant', 'n_intrap', 'n_total', 'pct_intrap', 'ci_lower', 'ci_upper']].to_string(index=False))

# Compare to baseline (overall intrap prevalence)
baseline_pct = df['intrap_binary'].mean() * 100
print(f"\nBaseline intrap prevalence: {baseline_pct:.1f}%")

# ============================================================================
# STEP 4: Enrichment Test (Fisher Exact)
# ============================================================================
print("\n[Step 4] Testing intrap enrichment per quadrant...")

# Test each quadrant vs all others
enrichment_results = []

for quadrant in df['quadrant'].unique():
    # Create 2x2 table
    in_quad_intrap = df[(df['quadrant'] == quadrant) & (df['intrap_binary'] == 1)].shape[0]
    in_quad_not = df[(df['quadrant'] == quadrant) & (df['intrap_binary'] == 0)].shape[0]
    out_quad_intrap = df[(df['quadrant'] != quadrant) & (df['intrap_binary'] == 1)].shape[0]
    out_quad_not = df[(df['quadrant'] != quadrant) & (df['intrap_binary'] == 0)].shape[0]
    
    # Fisher exact test
    table = [[in_quad_intrap, in_quad_not],
             [out_quad_intrap, out_quad_not]]
    
    odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
    
    # Compute OR with Haldane-Anscombe correction if zero cell
    if 0 in [in_quad_intrap, in_quad_not, out_quad_intrap, out_quad_not]:
        in_quad_intrap += 0.5
        in_quad_not += 0.5
        out_quad_intrap += 0.5
        out_quad_not += 0.5
        or_corrected = (in_quad_intrap * out_quad_not) / (in_quad_not * out_quad_intrap)
    else:
        or_corrected = odds_ratio
    
    enrichment_results.append({
        'quadrant': quadrant,
        'n_intrap_in_quad': in_quad_intrap,
        'n_not_in_quad': in_quad_not,
        'n_intrap_out_quad': out_quad_intrap,
        'n_not_out_quad': out_quad_not,
        'odds_ratio': or_corrected,
        'p_value': p_value
    })
    
    print(f"\n{quadrant}:")
    print(f"  OR = {or_corrected:.3f}, p = {p_value:.4f}")
    if p_value < 0.05:
        direction = "enriched" if or_corrected > 1 else "depleted"
        print(f"  => {direction} in intrap tasks")

enrichment_df = pd.DataFrame(enrichment_results)

# FDR correction
from statsmodels.stats.multitest import multipletests
_, q_values, _, _ = multipletests(enrichment_df['p_value'], method='fdr_bh')
enrichment_df['q_value_fdr'] = q_values

print("\n[FDR-Corrected Results]")
sig = enrichment_df[enrichment_df['q_value_fdr'] < 0.05]
if len(sig) > 0:
    print(sig[['quadrant', 'odds_ratio', 'p_value', 'q_value_fdr']].to_string(index=False))
else:
    print("No quadrants significant after FDR correction")

# ============================================================================
# STEP 5: Threshold Sensitivity Analysis
# ============================================================================
print("\n[Step 5] Testing threshold sensitivity...")

# Test alternative thresholds
thresholds_to_test = [
    ('strict', 4.5, 4.5),
    ('current', 4.0, 4.0),
    ('lenient', 3.5, 3.5),
    ('median', df['Importance'].median(), df['Frequency'].median())
]

sensitivity_results = []

for name, imp_thresh, freq_thresh in thresholds_to_test:
    df_temp = df.copy()
    df_temp['high_importance'] = (df_temp['Importance'] >= imp_thresh).astype(int)
    df_temp['high_frequency'] = (df_temp['Frequency'] >= freq_thresh).astype(int)
    df_temp['quadrant_temp'] = df_temp.apply(assign_quadrant, axis=1)
    
    # Get Core quadrant intrap %
    core_pct = df_temp[df_temp['quadrant_temp'] == 'Core']['intrap_binary'].mean() * 100
    
    # Get Operational quadrant intrap %
    operational_pct = df_temp[df_temp['quadrant_temp'] == 'Operational']['intrap_binary'].mean() * 100
    
    sensitivity_results.append({
        'threshold_set': name,
        'imp_threshold': imp_thresh,
        'freq_threshold': freq_thresh,
        'core_pct_intrap': core_pct,
        'operational_pct_intrap': operational_pct,
        'difference': core_pct - operational_pct
    })
    
    print(f"\n{name.upper()} (Imp>={imp_thresh:.2f}, Freq>={freq_thresh:.2f}):")
    print(f"  Core: {core_pct:.1f}% intrap")
    print(f"  Operational: {operational_pct:.1f}% intrap")
    print(f"  Difference: {core_pct - operational_pct:.1f} pp")

sensitivity_df = pd.DataFrame(sensitivity_results)

# ============================================================================
# STEP 6: Save Outputs
# ============================================================================
print("\n[Step 6] Saving outputs...")

# Save quadrant enrichment
enrichment_df.to_csv(TABLES_DIR / "p3_q6_2_quadrant_enrichment.csv", index=False)
quad_intrap.to_csv(TABLES_DIR / "p3_q6_2_quadrant_intrap_prevalence.csv", index=False)
sensitivity_df.to_csv(TABLES_DIR / "p3_q6_2_threshold_sensitivity.csv", index=False)

# Save raw data for visualization
df[['task_id', 'Importance', 'Frequency', 'intrap_binary', 'quadrant']].to_csv(
    TABLES_DIR / "p3_q6_2_task_level_data.csv", index=False
)

# Save metadata
metadata = {
    'analysis': 'Q6.2_Importance_Frequency_Matrix',
    'date_run': pd.Timestamp.now().strftime('%Y-%m-%d'),
    'thresholds': {
        'importance_threshold': float(importance_threshold),
        'frequency_threshold': float(frequency_threshold),
        'method': 'fixed_thresholds'
    },
    'quadrant_definitions': {
        'Core': 'High importance, High frequency',
        'Critical': 'High importance, Low frequency',
        'Operational': 'Low importance, High frequency',
        'Peripheral': 'Low importance, Low frequency'
    },
    'sample_size': {
        'n_total': int(len(df)),
        'n_intrap': int(df['intrap_binary'].sum()),
        'baseline_pct_intrap': float(baseline_pct)
    },
    'quadrant_counts': quad_counts.to_dict(),
    'statistical_tests': {
        'enrichment_method': 'Fisher exact test',
        'multiple_testing_correction': 'Benjamini-Hochberg FDR',
        'ci_method': 'Wilson score interval'
    }
}

with open(METADATA_DIR / "p3_q6_2_metadata.yaml", 'w') as f:
    yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

print(f"\n[OK] Saved {len(enrichment_df)} quadrant results")
print("[OK] Saved threshold sensitivity analysis")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("Q6.2 SUMMARY")
print("="*80)

print(f"\nSample: {len(df)} tasks, {df['intrap_binary'].sum()} intrap ({baseline_pct:.1f}%)")
print(f"Thresholds: Importance>={importance_threshold}, Frequency>={frequency_threshold}")

print("\n[Quadrant Distribution]")
for _, row in quad_intrap.iterrows():
    print(f"  {row['quadrant']}: {row['n_total']} tasks ({row['pct_intrap']:.1f}% intrap)")

print("\n[Key Findings]")
sig = enrichment_df[enrichment_df['q_value_fdr'] < 0.05].sort_values('odds_ratio', ascending=False)
if len(sig) > 0:
    for _, row in sig.iterrows():
        direction = "enriched" if row['odds_ratio'] > 1 else "depleted"
        print(f"  - {row['quadrant']}: {direction} (OR={row['odds_ratio']:.2f}, q={row['q_value_fdr']:.3f})")
else:
    print("  - No significant enrichment/depletion after FDR correction")

print("\n" + "="*80)
print("Q6.2 Complete!")
print("="*80)
