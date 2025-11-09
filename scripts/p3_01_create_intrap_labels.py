"""
Script: p3_01_create_intrap_labels.py
Purpose: Merge intrap classifications with features_experts for Part 3 analyses
Author: Entre-Audit Project
Date: 2025-01-14
Python version: 3.11.x

This script:
1. Loads intrap0_1_full_aggregate.csv (LLM classification results)
2. Loads features_experts.parquet (expert-rated task features)
3. Merges intrap labels: Intrapreneurial=1, Not Intrapreneurial=0
4. Handles ambiguous cases (exclude or flag)
5. Saves enhanced features_experts_with_intrap.parquet

Expected runtime: <1 minute
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set random seed for reproducibility
np.random.seed(42)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PART 3: CREATE INTRAP LABELS")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("[1/5] Loading data files...")

# Load intrap classification results
intrap_path = PROJECT_ROOT / "outputs" / "summaries" / "intrap0_1_full_aggregate.csv"
intrap_df = pd.read_csv(intrap_path)
print(f"   [OK] Loaded intrap classifications: {len(intrap_df):,} rows")

# Load features_experts
features_path = DATA_DIR / "processed" / "features_experts.parquet"
features_df = pd.read_parquet(features_path)
print(f"   [OK] Loaded features_experts: {len(features_df):,} tasks")
print()

# ============================================================================
# 2. INSPECT INTRAP CLASSIFICATIONS
# ============================================================================
print("[2/5] Inspecting intrap classifications...")

# Check classification values
if 'classification' in intrap_df.columns:
    classification_counts = intrap_df['classification'].value_counts()
    print("   Classification distribution:")
    for cls, count in classification_counts.items():
        pct = 100 * count / len(intrap_df)
        print(f"      {cls}: {count} ({pct:.1f}%)")
    print()
    
    # Check confidence levels
    if 'confidence' in intrap_df.columns:
        confidence_counts = intrap_df['confidence'].value_counts()
        print("   Confidence distribution:")
        for conf, count in confidence_counts.items():
            pct = 100 * count / len(intrap_df)
            print(f"      {conf}: {count} ({pct:.1f}%)")
        print()
else:
    print("   [WARN] 'classification' column not found")
    print(f"   Available columns: {list(intrap_df.columns)}")
    sys.exit(1)

# ============================================================================
# 3. CREATE BINARY INTRAP LABEL
# ============================================================================
print("[3/5] Creating binary intrap labels...")

# Map classifications to binary labels
def map_classification(cls):
    if cls == "Intrapreneurial":
        return 1
    elif cls == "Not Intrapreneurial":
        return 0
    else:
        return None  # Ambiguous cases

intrap_df['intrap'] = intrap_df['classification'].apply(map_classification)

# Count labels
n_intrap = (intrap_df['intrap'] == 1).sum()
n_not_intrap = (intrap_df['intrap'] == 0).sum()
n_ambiguous = intrap_df['intrap'].isna().sum()

print(f"   Intrapreneurial (intrap=1): {n_intrap}")
print(f"   Not Intrapreneurial (intrap=0): {n_not_intrap}")
print(f"   Ambiguous (intrap=None): {n_ambiguous}")
print()

if n_ambiguous > 0:
    print(f"   [WARN] {n_ambiguous} ambiguous tasks will be excluded from analyses")
    print()

# ==========================================================================
# 3b. DEDUPLICATE BY TASK_ID (KNOWN AGGREGATE DUPES IN intrap0_1 CSV)
# ==========================================================================
if 'task_id' in intrap_df.columns:
    before = len(intrap_df)
    # Keep first occurrence per task_id; segment assignment is handled elsewhere
    intrap_df = intrap_df.drop_duplicates(subset=['task_id'], keep='first')
    after = len(intrap_df)
    if after < before:
        print(f"[INFO] Dropped {before - after} duplicate rows in intrap classifications (by task_id)")
        print()

# ============================================================================
# 4. MERGE WITH FEATURES_EXPERTS
# ============================================================================
print("[4/5] Merging intrap labels with features_experts...")

# Check common columns for merging
common_cols = set(intrap_df.columns) & set(features_df.columns)
print(f"   Common columns: {sorted(common_cols)}")

# Determine merge key
if 'task_id' in common_cols:
    merge_key = 'task_id'
elif 'task_text' in common_cols:
    merge_key = 'task_text'
else:
    print("   [ERROR] No common merge key found (task_id or task_text)")
    sys.exit(1)

print(f"   Using merge key: {merge_key}")
print()

# Select columns to merge from intrap_df
intrap_cols_to_merge = [merge_key, 'intrap', 'classification', 'confidence', 
                        'matched_categories', 'matched_indicators', 'justification']
intrap_cols_to_merge = [c for c in intrap_cols_to_merge if c in intrap_df.columns]

# Perform merge
features_with_intrap = features_df.merge(
    intrap_df[intrap_cols_to_merge],
    on=merge_key,
    how='left'
)

print(f"   [OK] Merge complete: {len(features_with_intrap):,} tasks")

# Check merge success
n_matched = features_with_intrap['intrap'].notna().sum()
n_unmatched = features_with_intrap['intrap'].isna().sum()

print(f"   Matched tasks with intrap label: {n_matched}")
print(f"   Unmatched tasks: {n_unmatched}")

if n_unmatched > 0:
    print(f"   [WARN] {n_unmatched} tasks do not have intrap labels")
    # Show sample unmatched tasks
    unmatched_sample = features_with_intrap[features_with_intrap['intrap'].isna()].head(5)
    print(f"   Sample unmatched {merge_key}s:")
    for idx, row in unmatched_sample.iterrows():
        print(f"      - {row[merge_key]}")
print()

# ============================================================================
# 5. SAVE ENHANCED DATASET
# ============================================================================
print("[5/5] Saving enhanced dataset...")

# Save to parquet
output_path = DATA_DIR / "processed" / "features_experts_with_intrap.parquet"
features_with_intrap.to_parquet(output_path, index=False)
print(f"   [OK] Saved: {output_path}")
print(f"   Rows: {len(features_with_intrap):,}")
print(f"   Columns: {len(features_with_intrap.columns)}")
print()

# Save summary statistics
summary_stats = {
    'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_tasks': len(features_with_intrap),
    'tasks_with_intrap_label': int(n_matched),
    'intrapreneurial_tasks': int(n_intrap),
    'not_intrapreneurial_tasks': int(n_not_intrap),
    'ambiguous_tasks': int(n_ambiguous),
    'unmatched_tasks': int(n_unmatched),
    'intrap_prevalence_pct': float(100 * n_intrap / n_matched) if n_matched > 0 else None,
    'merge_key': merge_key,
    'added_columns': [c for c in intrap_cols_to_merge if c != merge_key]
}

import json
with open(OUTPUT_DIR / "p3_intrap_labels_summary.json", 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"   [OK] Saved: {OUTPUT_DIR / 'p3_intrap_labels_summary.json'}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("INTRAP LABELS CREATED SUCCESSFULLY")
print("="*80)
print()
print("KEY RESULTS:")
print(f"  1. [OK] {n_intrap} intrapreneurial tasks ({100*n_intrap/n_matched:.1f}%)")
print(f"  2. [OK] {n_not_intrap} not intrapreneurial tasks ({100*n_not_intrap/n_matched:.1f}%)")
if n_ambiguous > 0:
    print(f"  3. [WARN] {n_ambiguous} ambiguous tasks (will be excluded)")
if n_unmatched > 0:
    print(f"  4. [WARN] {n_unmatched} tasks without intrap labels")
print()
print("NEXT STEPS:")
print("  -> Update preliminary diagnostics to use features_experts_with_intrap.parquet")
print("  -> Proceed to Q2.1 implementation")
print()
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
