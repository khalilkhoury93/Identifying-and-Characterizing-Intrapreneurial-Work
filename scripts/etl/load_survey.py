import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml


def _snake(s: str) -> str:
    return (
        s.strip()
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .lower()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(__file__).resolve().parents[3] / 'config' / 'run.yaml'))
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    paths = cfg['paths']
    cfg_dir = cfg_path.parent
    wb_path = (cfg_dir / paths['raw_workbook']).resolve()

    xls = pd.ExcelFile(wb_path)

    # Load sheets
    df_tasks = pd.read_excel(wb_path, sheet_name='task_statement_with_metadata')
    df_workers = pd.read_excel(wb_path, sheet_name='domain_worker_desires')
    df_wmeta = pd.read_excel(wb_path, sheet_name='domain_worker_metadata')
    df_expert = pd.read_excel(wb_path, sheet_name='expert_rated_technological_capa')
    # Load authoritative variable metadata (distinct sheet)
    try:
        df_meta = pd.read_excel(wb_path, sheet_name='Metadata')
    except Exception:
        df_meta = pd.DataFrame()

    # Normalize columns
    df_tasks = df_tasks.rename(columns={
        'Task ID': 'task_id',
        'Task': 'task_text',
        'O*NET-SOC Code': 'soc',
        'Occupation (O*NET-SOC Title)': 'occupation_title',
        'Task Type': 'task_type',
        'Occupation Mean Annual Wage': 'occ_mean_annual_wage',
        'Occupation Employment': 'occ_employment',
        'Skill (O*NET Work Activity)': 'skill_onet_wa',
        'Skill ID (O*NET Generalized Work Activity ID)': 'skill_id_onet_gwa',
    })
    df_tasks['task_id'] = df_tasks['task_id'].astype(int)
    df_tasks['soc'] = df_tasks['soc'].astype(str)

    df_workers = df_workers.rename(columns={
        'Task ID': 'task_id',
        'Task': 'task_text',
        'User ID': 'user_id',
        'Automation Desire Rating': 'automation_desire',
        'Core Skill Rating': 'core_skill',
        'Job Security Rating': 'job_security',
        'Enjoyment Rating': 'enjoyment',
        'Human Agency Scale Rating': 'human_agency',
        'Physical Action Requirement': 'physical_action',
        'Interpersonal Communication Requirement': 'interpersonal_comm',
        'Involved Uncertainty': 'involved_uncertainty',
        'Domain Expertise Requirement': 'domain_expertise',
        'Time': 'time',
    })
    df_workers['task_id'] = df_workers['task_id'].astype(int)
    # Ensure free-text columns are strings to avoid mixed-type Arrow issues
    for col in ['Other Reason for Automation Desire', 'Other Reason for Human Agency']:
        if col in df_workers.columns:
            df_workers[col] = df_workers[col].astype(str)

    df_wmeta = df_wmeta.rename(columns={'User ID': 'user_id'})

    df_expert = df_expert.rename(columns={
        'Task ID': 'task_id',
        'Task': 'task_text',
        'User ID': 'expert_user_id',
        'Automation Capacity Rating': 'automation_capacity',
        'Physical Action Requirement': 'expert_physical_action',
        'Interpersonal Communication Requirement': 'expert_interpersonal_comm',
        'Involved Uncertainty': 'expert_involved_uncertainty',
        'Domain Expertise Requirement': 'expert_domain_expertise',
        'Human Agency Scale Rating': 'expert_human_agency',
    })
    df_expert['task_id'] = df_expert['task_id'].astype(int)

    # Persist raw/interim
    def _obj_to_str(d: pd.DataFrame) -> pd.DataFrame:
        for c in d.columns:
            if d[c].dtype == 'object':
                d[c] = d[c].astype(str)
        return d

    df_tasks = _obj_to_str(df_tasks)
    df_workers = _obj_to_str(df_workers)
    df_wmeta = _obj_to_str(df_wmeta)
    df_expert = _obj_to_str(df_expert)

    out_tasks = (cfg_dir / paths['tasks_interim']).resolve(); out_tasks.parent.mkdir(parents=True, exist_ok=True)
    out_wdes = (cfg_dir / paths['worker_desires_interim']).resolve()
    out_wmeta = (cfg_dir / paths['worker_metadata_interim']).resolve()
    out_expt = (cfg_dir / paths['expert_rated_interim']).resolve()
    out_meta = (cfg_dir / paths.get('survey_metadata_interim', '../data/interim/survey_metadata.parquet')).resolve()
    df_tasks.to_parquet(out_tasks, index=False)
    df_workers.to_parquet(out_wdes, index=False)
    df_wmeta.to_parquet(out_wmeta, index=False)
    df_expert.to_parquet(out_expt, index=False)
    if not df_meta.empty:
        # normalize metadata columns to snake_case for convenience
        df_meta.columns = [_snake(c) for c in df_meta.columns]
        df_meta = _obj_to_str(df_meta)
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        df_meta.to_parquet(out_meta, index=False)

    # Aggregations
    w_agg = df_workers.groupby('task_id').agg(
        pref_augment=('automation_desire','mean'),
        time=('time','mean'),
        core_skill=('core_skill','mean'),
        job_security=('job_security','mean'),
        enjoyment=('enjoyment','mean'),
        human_agency=('human_agency','mean'),
        n_worker_raters=('user_id','nunique'),
    ).reset_index()

    e_agg = df_expert.groupby('task_id').agg(
        expert_capability=('automation_capacity','mean'),
        n_expert_raters=('expert_user_id','nunique'),
    ).reset_index()

    ratings = w_agg.merge(e_agg, on='task_id', how='outer')
    out_ratings = (cfg_dir / paths['ratings_agg_out']).resolve()
    ratings.to_parquet(out_ratings, index=False)

    extra = f"\n  {out_meta}" if out_meta.exists() else ""
    print(f"WROTE:\n  {out_tasks}\n  {out_wdes}\n  {out_wmeta}\n  {out_expt}\n  {out_ratings}{extra}")


if __name__ == '__main__':
    main()
