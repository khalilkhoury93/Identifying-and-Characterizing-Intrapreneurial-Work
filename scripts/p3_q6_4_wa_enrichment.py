"""
Q6.4: O*NET Work Activities (WA) Enrichment Analysis

Tests which O*NET Work Activities are over-represented (enriched) or under-represented
(depleted) in intrapreneurial tasks.

This characterizes the behavioral profile of intrapreneurial work using the official
O*NET taxonomy of work activities.

Author: Analysis Pipeline
Date: 2025-01-14
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from scipy.stats import fisher_exact
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
METADATA_DIR = OUTPUT_DIR / "metadata"

for dir in [TABLES_DIR, METADATA_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Q6.4: Work Activities Enrichment")
print("="*80)

# ============================================================================
# STEP 1: Load and Parse Work Activities
# ============================================================================
print("\n[Step 1] Loading data and parsing Work Activities...")

tasks_df = pd.read_parquet(DATA_DIR / "interim" / "tasks_with_metadata.parquet")
intrap_df = pd.read_csv(OUTPUT_DIR / "summaries" / "intrap0_1_full_aggregate.csv")
features_df = pd.read_parquet(DATA_DIR / "processed" / "features_experts.parquet")

merged_df = tasks_df.merge(
    intrap_df[['task_id', 'classification']],
    on='task_id',
    how='inner'
)

expert_task_ids = features_df['task_id'].unique()
df = merged_df[merged_df['task_id'].isin(expert_task_ids)].copy()

df['intrap_binary'] = (df['classification'] == 'Intrapreneurial').astype(int)

print(f"Loaded {len(df)} tasks")
print(f"Intrap: {df['intrap_binary'].sum()} ({df['intrap_binary'].mean()*100:.1f}%)")

# Parse skill_onet_wa column
# This contains Work Activities as a string (may be semicolon or comma-separated)
print("\nParsing Work Activities...")

def parse_wa(wa_string):
    """Parse work activities from string to list"""
    if pd.isna(wa_string):
        return []
    # Try different separators
    if ';' in str(wa_string):
        return [w.strip() for w in str(wa_string).split(';') if w.strip()]
    elif ',' in str(wa_string):
        return [w.strip() for w in str(wa_string).split(',') if w.strip()]
    else:
        return [str(wa_string).strip()]

df['wa_list'] = df['skill_onet_wa'].apply(parse_wa)
df['n_wa'] = df['wa_list'].apply(len)

print(f"Tasks with Work Activities: {(df['n_wa'] > 0).sum()} ({(df['n_wa'] > 0).mean()*100:.1f}%)")
print(f"Mean WA per task: {df['n_wa'].mean():.1f}")
print(f"Median WA per task: {df['n_wa'].median():.0f}")

# Check if we have WA data
if (df['n_wa'] > 0).sum() == 0:
    print("\n[WARNING] No Work Activities found in data")
    print("This may indicate:")
    print("  1. skill_onet_wa column is empty/null")
    print("  2. Format needs different parsing")
    print("\nSample of skill_onet_wa values:")
    print(df['skill_onet_wa'].head(10))
    
    # Try to salvage by checking actual content
    sample = df['skill_onet_wa'].dropna().head(5)
    if len(sample) > 0:
        print("\nNon-null samples:")
        for val in sample:
            print(f"  '{val}'")

# ============================================================================
# STEP 2: Build Activity Catalog and Binary Matrix
# ============================================================================
print("\n[Step 2] Building Work Activity catalog...")

# Collect all unique WAs
all_was = []
for wa_list in df['wa_list']:
    all_was.extend(wa_list)

wa_counts = pd.Series(all_was).value_counts()
print(f"\nFound {len(wa_counts)} unique Work Activities")
print(f"Total WA mentions: {len(all_was)}")

if len(wa_counts) == 0:
    print("\n[ERROR] No Work Activities could be parsed")
    print("Exiting Q6.4 - check data format")
    exit(1)

print("\nTop 10 most common Work Activities:")
print(wa_counts.head(10))

# Filter to WAs appearing in at least 5% of tasks (to avoid rare activities)
prevalence_threshold = 0.05
min_tasks = int(len(df) * prevalence_threshold)
common_was = wa_counts[wa_counts >= min_tasks].index.tolist()

print(f"\nWAs appearing in >={prevalence_threshold*100:.0f}% of tasks (>={min_tasks}): {len(common_was)}")

# Create binary matrix for common WAs
for wa in common_was:
    df[f'has_{wa}'] = df['wa_list'].apply(lambda x: 1 if wa in x else 0)

# ============================================================================
# STEP 3: Enrichment Testing (Fisher Exact per WA)
# ============================================================================
print("\n[Step 3] Testing enrichment for each Work Activity...")

enrichment_results = []

for wa in common_was:
    col = f'has_{wa}'
    
    # 2x2 table
    intrap_has = df[(df['intrap_binary'] == 1) & (df[col] == 1)].shape[0]
    intrap_not = df[(df['intrap_binary'] == 1) & (df[col] == 0)].shape[0]
    not_has = df[(df['intrap_binary'] == 0) & (df[col] == 1)].shape[0]
    not_not = df[(df['intrap_binary'] == 0) & (df[col] == 0)].shape[0]
    
    table = [[intrap_has, intrap_not],
             [not_has, not_not]]
    
    # Fisher exact
    or_val, p_val = fisher_exact(table, alternative='two-sided')
    
    # Apply Haldane-Anscombe correction if zero cell
    if 0 in [intrap_has, intrap_not, not_has, not_not]:
        intrap_has_c = intrap_has + 0.5
        intrap_not_c = intrap_not + 0.5
        not_has_c = not_has + 0.5
        not_not_c = not_not + 0.5
        or_corrected = (intrap_has_c * not_not_c) / (intrap_not_c * not_has_c)
    else:
        or_corrected = or_val
    
    # Prevalence
    prev_intrap = intrap_has / (intrap_has + intrap_not) * 100
    prev_not = not_has / (not_has + not_not) * 100
    
    enrichment_results.append({
        'work_activity': wa,
        'n_tasks_with_wa': intrap_has + not_has,
        'prevalence_overall': (intrap_has + not_has) / len(df) * 100,
        'prevalence_intrap': prev_intrap,
        'prevalence_not': prev_not,
        'n_intrap_with': intrap_has,
        'n_intrap_without': intrap_not,
        'n_not_with': not_has,
        'n_not_without': not_not,
        'odds_ratio': or_corrected,
        'p_value': p_val
    })

enrichment_df = pd.DataFrame(enrichment_results)

# Sort by OR
enrichment_df = enrichment_df.sort_values('odds_ratio', ascending=False)

# FDR correction
from statsmodels.stats.multitest import multipletests
_, q_values, _, _ = multipletests(enrichment_df['p_value'], method='fdr_bh')
enrichment_df['q_value_fdr'] = q_values

# Identify significant after FDR
enrichment_df['significant'] = enrichment_df['q_value_fdr'] < 0.05

print(f"\nTested {len(enrichment_df)} Work Activities")
print(f"Significant after FDR (q<0.05): {enrichment_df['significant'].sum()}")

# ============================================================================
# STEP 4: Report Top Enriched and Depleted
# ============================================================================
print("\n[Step 4] Top enriched and depleted Work Activities...")

sig_df = enrichment_df[enrichment_df['significant']].copy()

if len(sig_df) > 0:
    enriched = sig_df[sig_df['odds_ratio'] > 1].sort_values('odds_ratio', ascending=False)
    depleted = sig_df[sig_df['odds_ratio'] < 1].sort_values('odds_ratio')
    
    if len(enriched) > 0:
        print(f"\n[TOP {min(10, len(enriched))} ENRICHED (OR>1, q<0.05)]")
        for _, row in enriched.head(10).iterrows():
            print(f"  {row['work_activity'][:60]:<60} OR={row['odds_ratio']:.2f}, q={row['q_value_fdr']:.3f}")
    
    if len(depleted) > 0:
        print(f"\n[TOP {min(10, len(depleted))} DEPLETED (OR<1, q<0.05)]")
        for _, row in depleted.head(10).iterrows():
            print(f"  {row['work_activity'][:60]:<60} OR={row['odds_ratio']:.2f}, q={row['q_value_fdr']:.3f}")
else:
    print("\nNo Work Activities significant after FDR correction")
    print("\nTop 10 by nominal p-value (before correction):")
    top_nominal = enrichment_df.sort_values('p_value').head(10)
    for _, row in top_nominal.iterrows():
        direction = "enriched" if row['odds_ratio'] > 1 else "depleted"
        print(f"  {row['work_activity'][:60]:<60} OR={row['odds_ratio']:.2f}, p={row['p_value']:.4f} ({direction})")

# ============================================================================
# STEP 5: Save Outputs
# ============================================================================
print("\n[Step 5] Saving outputs...")

# Save full enrichment results
enrichment_df.to_csv(TABLES_DIR / "p3_q6_4_wa_enrichment_full.csv", index=False)

# Save significant only
if len(sig_df) > 0:
    sig_df.to_csv(TABLES_DIR / "p3_q6_4_wa_enrichment_significant.csv", index=False)

# Save metadata
metadata = {
    'analysis': 'Q6.4_Work_Activities_Enrichment',
    'date_run': pd.Timestamp.now().strftime('%Y-%m-%d'),
    'sample_size': {
        'n_total': int(len(df)),
        'n_intrap': int(df['intrap_binary'].sum()),
        'n_with_wa_data': int((df['n_wa'] > 0).sum())
    },
    'filtering': {
        'prevalence_threshold': float(prevalence_threshold),
        'min_tasks': int(min_tasks),
        'n_unique_was': int(len(wa_counts)),
        'n_common_was_tested': int(len(common_was))
    },
    'statistical_tests': {
        'method': 'Fisher exact test',
        'multiple_testing_correction': 'Benjamini-Hochberg FDR',
        'or_correction': 'Haldane-Anscombe for zero cells'
    },
    'results': {
        'n_significant': int(enrichment_df['significant'].sum()),
        'n_enriched': int(((enrichment_df['odds_ratio'] > 1) & enrichment_df['significant']).sum()),
        'n_depleted': int(((enrichment_df['odds_ratio'] < 1) & enrichment_df['significant']).sum())
    }
}

with open(METADATA_DIR / "p3_q6_4_metadata.yaml", 'w') as f:
    yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

print(f"\n[OK] Saved {len(enrichment_df)} Work Activity enrichment results")
if len(sig_df) > 0:
    print(f"[OK] Saved {len(sig_df)} significant results")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("Q6.4 SUMMARY")
print("="*80)

print(f"\nSample: {len(df)} tasks ({df['intrap_binary'].sum()} intrap)")
print(f"Work Activities tested: {len(enrichment_df)}")
print(f"Significant (q<0.05): {enrichment_df['significant'].sum()}")

if enrichment_df['significant'].sum() > 0:
    n_enriched = ((enrichment_df['odds_ratio'] > 1) & enrichment_df['significant']).sum()
    n_depleted = ((enrichment_df['odds_ratio'] < 1) & enrichment_df['significant']).sum()
    print(f"  Enriched in intrap: {n_enriched}")
    print(f"  Depleted in intrap: {n_depleted}")
    
    print("\n[Top 3 Enriched]")
    for _, row in sig_df[sig_df['odds_ratio'] > 1].head(3).iterrows():
        print(f"  - {row['work_activity']}")
        print(f"    OR={row['odds_ratio']:.2f}, {row['prevalence_intrap']:.1f}% intrap vs {row['prevalence_not']:.1f}% not")

print("\n" + "="*80)
print("Q6.4 Complete!")
print("="*80)
