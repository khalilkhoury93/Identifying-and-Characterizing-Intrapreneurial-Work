import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml


def zscore_by_group(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    def _z(g):
        v = g[value_col].astype(float)
        m = v.mean()
        s = v.std(ddof=0)
        if s == 0 or np.isnan(s):
            z = np.zeros(len(v))
        else:
            z = (v - m) / s
        g = g.copy()
        g['z'] = z
        return g
    return df.groupby(group_cols, as_index=False, sort=False).apply(_z).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(__file__).resolve().parents[3] / 'config' / 'run.yaml'))
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    paths = cfg['paths']
    cfg_dir = cfg_path.parent
    wa = pd.read_parquet((cfg_dir / paths['wa_out']).resolve())
    wc = pd.read_parquet((cfg_dir / paths['wc_out']).resolve())

    # z-score within each element_id+scale across SOCs
    wa_z = zscore_by_group(wa, ['element_id','scale'], 'value')
    wc_z = zscore_by_group(wc, ['element_id','scale'], 'value')

    # Apply invert for specific WC elements if configured
    invert_ids = {x['id'] for x in cfg['onet_elements']['work_context_keep'] if x.get('invert')}
    if not wc_z.empty and 'z' in wc_z.columns:
        wc_z.loc[wc_z['element_id'].isin(invert_ids), 'z'] *= -1

    # Pivot to wide feature matrix
    wa_w = pd.DataFrame()
    wc_w = pd.DataFrame()
    if not wa_z.empty:
        wa_z['feat'] = wa_z['element_id'] + '_' + wa_z['scale']
        wa_w = wa_z.pivot_table(index='soc', columns='feat', values='z', aggfunc='mean')
    if not wc_z.empty:
        wc_z['feat'] = wc_z['element_id'] + '_' + wc_z['scale']
        wc_w = wc_z.pivot_table(index='soc', columns='feat', values='z', aggfunc='mean')

    feats = []
    if not wa_w.empty:
        wa_w.columns = [f'WA_{c}' for c in wa_w.columns]
        feats.append(wa_w)
    if not wc_w.empty:
        wc_w.columns = [f'WC_{c}' for c in wc_w.columns]
        feats.append(wc_w)

    # Concatenate features
    X = pd.concat(feats, axis=1).sort_index()
    X['E_onet_SOC'] = X.mean(axis=1, skipna=True)

    # Optional WA groups: compute group means and min-max per SOC
    wa_groups = cfg.get('wa_groups', [])
    if wa_groups and 'WA_' in ''.join(X.columns.astype(str)):
        # We need access to WA-only wide table for column lookup
        wa_cols = [c for c in X.columns if isinstance(c, str) and c.startswith('WA_')]
        for grp in wa_groups:
            name = grp.get('name') or 'group'
            elts = grp.get('elements', [])
            cols = []
            for eid in elts:
                for sc in ('IM','LV'):
                    col = f'WA_{eid}_{sc}'
                    if col in X.columns:
                        cols.append(col)
            if not cols:
                continue
            gmean = X[cols].mean(axis=1, skipna=True)
            col_mean = f'E_onet_SOC_wa_{name}'
            col_mm = f'{col_mean}_mm'
            X[col_mean] = gmean
            # min-max scale group mean
            gmin, gmax = float(gmean.min()), float(gmean.max())
            if pd.isna(gmin) or pd.isna(gmax) or gmin == gmax:
                X[col_mm] = 0.5
            else:
                X[col_mm] = (gmean - gmin) / (gmax - gmin)

    # Min-max scale to [0,1]
    e = X['E_onet_SOC']
    emin, emax = e.min(), e.max()
    if pd.isna(emin) or pd.isna(emax) or emin == emax:
        X['E_onet_SOC_mm'] = 0.5
    else:
        X['E_onet_SOC_mm'] = (e - emin) / (emax - emin)

    X = X.reset_index().rename(columns={'soc':'soc'})

    out_path = (cfg_dir / paths['anchor_out']).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(out_path, index=False)
    print(f"WROTE: {out_path}  (rows={len(X)}, features={X.shape[1]-3})")


if __name__ == '__main__':
    main()
