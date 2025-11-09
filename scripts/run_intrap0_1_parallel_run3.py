"""
Intrapreneurial classification - RUN 3 (parallelized with 20 workers)
Processes all 844 unique tasks for voting analysis.
Output directory: intrap0_1_run3/
"""
import os
import json
from pathlib import Path
import sys
import pandas as pd
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from datetime import datetime

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from openai import OpenAI
from src.screening.validators import parse_json_strict


def _extract_text(resp) -> str:
    """Extract text from OpenAI response object."""
    txt = getattr(resp, 'output_text', None)
    if isinstance(txt, str) and txt.strip():
        return txt
    try:
        parts = []
        for item in getattr(resp, 'output', []) or []:
            for c in getattr(item, 'content', []) or []:
                if getattr(c, 'type', None) == 'output_text' and getattr(c, 'text', None):
                    parts.append(c.text)
        if parts:
            return "\n".join(parts)
    except Exception:
        pass
    try:
        d = resp.to_dict()
        out = d.get('output') or []
        if out and 'content' in out[0] and out[0]['content']:
            t = out[0]['content'][0].get('text')
            if t:
                return t
    except Exception:
        pass
    return ""


def process_single_task(row, client, model, system_prompt, dwa_map):
    """Process a single task with API call."""
    text = str(row['task_text'])
    meta = {
        'occupation_title': row.get('occupation_title'),
        'skill_onet_wa': row.get('skill_onet_wa'),
        'dwa_titles': dwa_map.get(int(row['task_id']), []),
    }
    meta_json = json.dumps(meta, ensure_ascii=False)
    combined = (
        f"{system_prompt.strip()}\n\n"
        f"Task metadata (JSON):\n{meta_json}\n\n"
        f"Task:\n{text}\n\n"
        f"Return EXACTLY one JSON object with the schema and keys specified."
    )
    
    try:
        resp = client.responses.create(
            model=model,
            input=combined,
            max_output_tokens=4000,
            reasoning={"effort": "low"},
        )
        resp_text = _extract_text(resp)
    except Exception as e:
        resp_text = str(e)
    
    parsed, errs = parse_json_strict(resp_text)
    rec = {
        'task_id': int(row['task_id']),
        'task_text': text,
        'metadata': meta,
        'response_text': resp_text,
        'parsed': parsed,
        'parse_ok': parsed is not None and not errs,
        'errors': errs,
    }
    return rec


def main() -> None:
    base = Path(r"C:\Users\ACER\Desktop\Thesis\entre-audit\PLAN_3_FILES")
    
    print("\n" + "="*70)
    print("INTRAPRENEURIAL CLASSIFICATION - RUN 3")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load all unique tasks from run 1
    agg_path = base / 'outputs' / 'summaries' / 'intrap0_1_full_aggregate.csv'
    df_all = pd.read_csv(agg_path)
    
    # Remove duplicates (keep first)
    tasks = df_all.drop_duplicates(subset='task_id', keep='first').copy()
    print(f"\nTotal unique tasks: {len(tasks)}")
    print(f"Segments: {tasks['segment'].value_counts().to_dict()}")
    
    # Load metadata
    feats = pd.read_parquet(base / 'data' / 'processed' / 'features_experts.parquet')
    keep_cols = ['task_id','occupation_title','skill_onet_wa']
    feats = feats[[c for c in keep_cols if c in feats.columns]].copy()
    feats['task_id'] = feats['task_id'].astype(int)
    tasks['task_id'] = tasks['task_id'].astype(int)
    
    # Merge and use features metadata
    tasks = tasks.merge(feats[keep_cols], on='task_id', how='left', suffixes=('', '_feat'))
    if 'occupation_title_feat' in tasks.columns:
        tasks['occupation_title'] = tasks['occupation_title_feat'].fillna(tasks['occupation_title'])
        tasks = tasks.drop(columns=['occupation_title_feat'])
    
    # Load DWA mapping
    dwa_path = base / 'data' / 'raw' / 'Tasks to DWAs.xlsx'
    dwa_map = {}
    try:
        df_dwa = pd.read_excel(dwa_path)
        cols = {c.lower(): c for c in df_dwa.columns}
        task_col = next((v for k, v in cols.items() if 'task' in k and 'id' in k), None)
        dwa_title_col = next((v for k, v in cols.items() if 'dwa' in k and ('title' in k or 'name' in k)), None)
        if task_col and dwa_title_col:
            tmp = df_dwa[[task_col, dwa_title_col]].dropna()
            tmp[task_col] = tmp[task_col].astype(int)
            grp = tmp.groupby(task_col)[dwa_title_col].apply(lambda s: sorted(set(map(str, s)))).reset_index()
            dwa_map = {int(r[task_col]): r[dwa_title_col] for _, r in grp.iterrows()}
            print(f"Loaded DWA mapping for {len(dwa_map)} tasks")
    except Exception as e:
        print(f"Warning: Could not load DWA mapping: {e}")
    
    # Load prompt
    prompt_yaml = base / 'prompts' / 'intrap0.1_single.yaml'
    cfg = yaml.safe_load(prompt_yaml.read_text(encoding='utf-8'))
    system_prompt = cfg['instruction']
    
    # Initialize OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY is not set in the environment')
    client = OpenAI(api_key=api_key)
    model = 'gpt-5-mini'
    
    # Output directory - RUN 3
    out_dir = base / 'outputs' / 'screening' / 'intrap0_1_run3'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'intrap0_1_run3.ndjson'
    
    print(f"\nConfiguration:")
    print(f"  Workers: 20")
    print(f"  Model: {model}")
    print(f"  Output: {out_path}")
    print("="*70 + "\n")
    
    # Process with thread pool
    start_time = time.time()
    results = []
    file_lock = Lock()
    
    # Progress tracking
    completed = 0
    intrap_count = 0
    not_intrap_count = 0
    parse_errors = 0
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks
        future_to_task = {}
        for idx, row in tasks.iterrows():
            future = executor.submit(process_single_task, row, client, model, system_prompt, dwa_map)
            future_to_task[future] = (row['task_id'], row.get('segment', 'unknown'))
        
        print(f"Submitted {len(future_to_task)} tasks to worker pool\n")
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            task_id, segment = future_to_task[future]
            try:
                rec = future.result()
                results.append(rec)
                completed += 1
                
                # Write immediately to file (thread-safe)
                with file_lock:
                    with open(out_path, 'a', encoding='utf-8') as fout:
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
                # Track classifications
                if rec['parsed']:
                    classification = rec['parsed'].get('classification', 'UNKNOWN')
                    confidence = rec['parsed'].get('confidence', 'N/A')
                    
                    if 'Intrapreneurial' in classification and 'Not' not in classification:
                        intrap_count += 1
                    elif 'Not Intrapreneurial' in classification:
                        not_intrap_count += 1
                else:
                    classification = 'PARSE_ERROR'
                    confidence = 'N/A'
                    parse_errors += 1
                
                # Print progress every 50 tasks
                if completed % 50 == 0 or completed == len(tasks):
                    elapsed = time.time() - start_time
                    rpm = (completed / elapsed) * 60 if elapsed > 0 else 0
                    print(f"[{completed:3d}/{len(tasks)}] RPM: {rpm:5.1f} | Intrap: {intrap_count:3d} | Not: {not_intrap_count:3d} | Errors: {parse_errors}")
                
            except Exception as e:
                print(f"ERROR processing task {task_id}: {e}")
    
    elapsed = time.time() - start_time
    
    # Final summary
    print("\n" + "="*70)
    print("RUN 3 COMPLETE")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Completed: {len(results)}")
    print(f"Time: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
    print(f"Average rate: {len(results)/elapsed*60:.1f} RPM")
    
    # Parse success
    parse_ok = sum(1 for r in results if r['parse_ok'])
    print(f"\nParse success: {parse_ok}/{len(results)} ({100*parse_ok/len(results):.1f}%)")
    
    # Classifications
    classifications = {}
    confidences = {}
    for r in results:
        if r['parsed']:
            cls = r['parsed'].get('classification', 'UNKNOWN')
            conf = r['parsed'].get('confidence', 'Unknown')
            classifications[cls] = classifications.get(cls, 0) + 1
            confidences[conf] = confidences.get(conf, 0) + 1
    
    print("\nClassifications:")
    for cls, count in sorted(classifications.items()):
        pct = 100 * count / len(results)
        print(f"  {cls}: {count} ({pct:.1f}%)")
    
    print("\nConfidence levels:")
    for conf, count in sorted(confidences.items()):
        pct = 100 * count / len(results)
        print(f"  {conf}: {count} ({pct:.1f}%)")
    
    print(f"\nOutput file: {out_path}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
