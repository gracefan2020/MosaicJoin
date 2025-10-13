#!/usr/bin/env python3
"""
Generate analyses comparing DeepJoin vs Semantic Sketches (our method).

Outputs four CSVs under an analyses directory:
  - deepjoin_right_semantic_wrong.csv: Cases where DeepJoin predicted a correct candidate
    (present in ground truth) that our method did not predict.
  - semantic_right_deepjoin_wrong.csv: Cases where our method predicted a correct candidate
    that DeepJoin did not predict.
  - deepjoin_false_positives.csv: Cases where DeepJoin predicted a candidate that is not
    in the ground truth (false positive).
  - semantic_false_positives.csv: Cases where our method predicted a candidate that is not
    in the ground truth (false positive).

Correctness is defined against the full candidate lists (no top-k cutoff).
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
import pandas as pd


def load_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    gt_df = pd.read_csv(gt_path)
    gt_df['target_ds'] = gt_df['target_ds'].str.replace('.csv', '', regex=False).str.lower()
    gt_df['candidate_ds'] = gt_df['candidate_ds'].str.replace('.csv', '', regex=False).str.lower()
    by_query: Dict[str, Set[str]] = {}
    for _, r in gt_df.iterrows():
        q = f"{r['target_ds']}.{r['target_attr']}"
        c = f"{r['candidate_ds']}.{r['candidate_attr']}"
        by_query.setdefault(q, set()).add(c)
    return by_query


def load_deepjoin_results(deepjoin_path: str) -> Dict[str, List[Tuple[str, float]]]:
    df = pd.read_csv(deepjoin_path)
    df['query_table'] = df['query_table'].str.replace('datalake-', '', regex=False).str.replace('.csv', '', regex=False).str.lower()
    df['candidate_table'] = df['candidate_table'].str.replace('datalake-', '', regex=False).str.replace('.csv', '', regex=False).str.lower()
    out: Dict[str, List[Tuple[str, float]]]= {}
    for _, r in df.iterrows():
        q = f"{r['query_table']}.{r['query_col']}"
        c = f"{r['candidate_table']}.{r['candidate_col']}"
        out.setdefault(q, []).append((c, float(r['score'])))
    # sort by score desc
    for k in out:
        out[k].sort(key=lambda x: x[1], reverse=True)
    return out


def load_semantic_results(sketch_dir: str) -> Dict[str, List[Tuple[str, float]]]:
    by_query: Dict[str, List[Tuple[str, float]]] = {}
    for p in Path(sketch_dir).glob("*_joinable_tables.csv"):
        query_table = p.stem.replace("_joinable_tables", "").lower()
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        for _, r in df.iterrows():
            q = f"{query_table}.{r['query_column']}"
            cand_tbl = str(r['candidate_table']).replace('datalake-', '').replace('.csv', '').lower().strip()
            c = f"{cand_tbl}.{r['candidate_column']}"
            score = float(r['best_table_density'])
            by_query.setdefault(q, []).append((c, score))
    # sort by density desc
    for k in by_query:
        by_query[k].sort(key=lambda x: x[1], reverse=True)
    return by_query


def remove_self_joins(preds: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[Tuple[str, float]]]:
    cleaned: Dict[str, List[Tuple[str, float]]] = {}
    for q, items in preds.items():
        cleaned[q] = [(c, s) for c, s in items if c != q]
    return cleaned


def to_rank_map(items: List[Tuple[str, float]]) -> Dict[str, Tuple[int, float]]:
    return {cand: (i + 1, score) for i, (cand, score) in enumerate(items)}


def split_key(key: str) -> Tuple[str, str]:
    if '.' in key:
        t, c = key.split('.', 1)
        return t, c
    return key, ''


def main():
    ap = argparse.ArgumentParser(description='Analyze disagreements between DeepJoin and Semantic Sketches')
    ap.add_argument('--ground-truth', required=True)
    ap.add_argument('--deepjoin-results', required=True)
    ap.add_argument('--semantic-sketches-dir', required=True)
    ap.add_argument('--out-dir', required=True, help='Output directory for analyses CSVs')
    ap.add_argument('--datalake-dir', required=False, default='datasets/freyja-semantic-join/datalake',
                    help='Root directory of the datalake for sampling column values')
    ap.add_argument('--sample-count', type=int, default=5, help='Number of sample values per column to include')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(args.ground_truth)
    dj = load_deepjoin_results(args.deepjoin_results)
    ss = load_semantic_results(args.semantic_sketches_dir)
    # Remove self-joins for output CSVs
    dj = remove_self_joins(dj)
    ss = remove_self_joins(ss)

    rows_dj_right = []
    rows_ss_right = []
    rows_dj_false_positives = []
    rows_ss_false_positives = []

    # Simple cache for sampled values to avoid re-reading files repeatedly
    sample_cache: Dict[Tuple[str, str], List[str]] = {}

    def load_column_samples(table: str, column: str, datalake_root: str, k: int) -> List[str]:
        key = (table, column)
        if key in sample_cache:
            return sample_cache[key]
        # Try CSV first, then Parquet
        csv_path = Path(datalake_root) / f"{table}.csv"
        pq_path = Path(datalake_root) / f"{table}.parquet"
        values: List[str] = []
        try:
            if csv_path.exists():
                ser = pd.read_csv(csv_path, usecols=[column], dtype=str)[column]
            elif pq_path.exists():
                ser = pd.read_parquet(pq_path, columns=[column]).astype(str)[column]
            else:
                sample_cache[key] = []
                return []
            # Dropna, strip, take unique order-preserving
            seen = set()
            for v in ser.dropna().astype(str):
                sv = v.strip()
                if not sv:
                    continue
                if sv in seen:
                    continue
                seen.add(sv)
                values.append(sv)
                if len(values) >= k:
                    break
        except Exception:
            values = []
        sample_cache[key] = values
        return values

    # Cache for full unique value sets per column for overlap computation
    valueset_cache: Dict[Tuple[str, str], Set[str]] = {}

    def get_value_set(table: str, column: str, datalake_root: str) -> Set[str]:
        key = (table, column)
        if key in valueset_cache:
            return valueset_cache[key]
        csv_path = Path(datalake_root) / f"{table}.csv"
        pq_path = Path(datalake_root) / f"{table}.parquet"
        vals: Set[str] = set()
        try:
            if csv_path.exists():
                ser = pd.read_csv(csv_path, usecols=[column], dtype=str)[column]
            elif pq_path.exists():
                ser = pd.read_parquet(pq_path, columns=[column]).astype(str)[column]
            else:
                valueset_cache[key] = set()
                return set()
            # Normalize similar to value_overlap: remove non-word chars, trim, lowercase
            ser = ser.dropna().astype(str)
            ser = ser.str.replace(r'[^\w\s]', '', regex=True).str.strip().str.lower()
            vals = set(ser[ser != ''])
        except Exception:
            vals = set()
        valueset_cache[key] = vals
        return vals

    all_queries = set(gt.keys())

    for q in sorted(all_queries):
        gt_set = gt.get(q, set())
        dj_list = dj.get(q, [])
        ss_list = ss.get(q, [])

        dj_rank = to_rank_map(dj_list)
        ss_rank = to_rank_map(ss_list)

        # Candidates DeepJoin got correct (in GT)
        dj_correct = {cand for cand, _ in dj_list if cand in gt_set}
        ss_correct = {cand for cand, _ in ss_list if cand in gt_set}
        
        # False positives: predicted by method but not in ground truth
        dj_false_positives = {cand for cand, _ in dj_list if cand not in gt_set}
        ss_false_positives = {cand for cand, _ in ss_list if cand not in gt_set}

        # DeepJoin right, Semantic wrong (Semantic missed these correct candidates)
        for cand in sorted(dj_correct - ss_correct):
            q_table, q_col = split_key(q)
            c_table, c_col = split_key(cand)
            dj_r, dj_s = dj_rank.get(cand, (None, None))
            ss_r, ss_s = ss_rank.get(cand, (None, None))
            q_samples = load_column_samples(q_table, q_col, args.datalake_dir, args.sample_count)
            c_samples = load_column_samples(c_table, c_col, args.datalake_dir, args.sample_count)
            # Compute value overlap (Jaccard) at full-column level
            set_q = get_value_set(q_table, q_col, args.datalake_dir)
            set_c = get_value_set(c_table, c_col, args.datalake_dir)
            overlap = round((len(set_q & set_c) / len(set_q | set_c)), 3) if (set_q or set_c) else 0.0
            rows_dj_right.append({
                'query_table': q_table,
                'query_column': q_col,
                'candidate_table': c_table,
                'candidate_column': c_col,
                'deepjoin_rank': dj_r,
                'deepjoin_score': dj_s,
                'semantic_rank': ss_r,
                'semantic_score': ss_s,
                'query_samples': ','.join(q_samples),
                'candidate_samples': ','.join(c_samples),
                'overlapping_values': ','.join(list(set_q & set_c)[:args.sample_count]),
                'Jaccard_similarity': overlap,
            })

        # Semantic right, DeepJoin wrong (DeepJoin missed these correct candidates)
        for cand in sorted(ss_correct - dj_correct):
            q_table, q_col = split_key(q)
            c_table, c_col = split_key(cand)
            dj_r, dj_s = dj_rank.get(cand, (None, None))
            ss_r, ss_s = ss_rank.get(cand, (None, None))
            q_samples = load_column_samples(q_table, q_col, args.datalake_dir, args.sample_count)
            c_samples = load_column_samples(c_table, c_col, args.datalake_dir, args.sample_count)
            set_q = get_value_set(q_table, q_col, args.datalake_dir)
            set_c = get_value_set(c_table, c_col, args.datalake_dir)
            overlap = round((len(set_q & set_c) / len(set_q | set_c)), 3) if (set_q or set_c) else 0.0
            rows_ss_right.append({
                'query_table': q_table,
                'query_column': q_col,
                'candidate_table': c_table,
                'candidate_column': c_col,
                'deepjoin_rank': dj_r,
                'deepjoin_score': dj_s,
                'semantic_rank': ss_r,
                'semantic_score': ss_s,
                'query_samples': ','.join(q_samples),
                'candidate_samples': ','.join(c_samples),
                'overlapping_values': ','.join(list(set_q & set_c)[:args.sample_count]),
                'Jaccard_similarity': overlap,
            })

        # DeepJoin false positives (predicted by DeepJoin but not in ground truth)
        for cand in sorted(dj_false_positives):
            q_table, q_col = split_key(q)
            c_table, c_col = split_key(cand)
            dj_r, dj_s = dj_rank.get(cand, (None, None))
            ss_r, ss_s = ss_rank.get(cand, (None, None))
            q_samples = load_column_samples(q_table, q_col, args.datalake_dir, args.sample_count)
            c_samples = load_column_samples(c_table, c_col, args.datalake_dir, args.sample_count)
            set_q = get_value_set(q_table, q_col, args.datalake_dir)
            set_c = get_value_set(c_table, c_col, args.datalake_dir)
            overlap = round((len(set_q & set_c) / len(set_q | set_c)), 3) if (set_q or set_c) else 0.0
            rows_dj_false_positives.append({
                'query_table': q_table,
                'query_column': q_col,
                'candidate_table': c_table,
                'candidate_column': c_col,
                'deepjoin_rank': dj_r,
                'deepjoin_score': dj_s,
                'semantic_rank': ss_r,
                'semantic_score': ss_s,
                'query_samples': ','.join(q_samples),
                'candidate_samples': ','.join(c_samples),
                'overlapping_values': ','.join(list(set_q & set_c)[:args.sample_count]),
                'Jaccard_similarity': overlap,
            })

        # Semantic false positives (predicted by Semantic but not in ground truth)
        for cand in sorted(ss_false_positives):
            q_table, q_col = split_key(q)
            c_table, c_col = split_key(cand)
            dj_r, dj_s = dj_rank.get(cand, (None, None))
            ss_r, ss_s = ss_rank.get(cand, (None, None))
            q_samples = load_column_samples(q_table, q_col, args.datalake_dir, args.sample_count)
            c_samples = load_column_samples(c_table, c_col, args.datalake_dir, args.sample_count)
            set_q = get_value_set(q_table, q_col, args.datalake_dir)
            set_c = get_value_set(c_table, c_col, args.datalake_dir)
            overlap = round((len(set_q & set_c) / len(set_q | set_c)), 3) if (set_q or set_c) else 0.0
            rows_ss_false_positives.append({
                'query_table': q_table,
                'query_column': q_col,
                'candidate_table': c_table,
                'candidate_column': c_col,
                'deepjoin_rank': dj_r,
                'deepjoin_score': dj_s,
                'semantic_rank': ss_r,
                'semantic_score': ss_s,
                'query_samples': ','.join(q_samples),
                'candidate_samples': ','.join(c_samples),
                'overlapping_values': ','.join(list(set_q & set_c)[:args.sample_count]),
                'Jaccard_similarity': overlap,
            })

    df_dj_right = pd.DataFrame(rows_dj_right)
    if not df_dj_right.empty and 'Jaccard_similarity' in df_dj_right.columns:
        df_dj_right = df_dj_right.sort_values(by='Jaccard_similarity', ascending=False)
    df_dj_right.to_csv(out_dir / 'deepjoin_right_semantic_wrong.csv', index=False)

    df_ss_right = pd.DataFrame(rows_ss_right)
    if not df_ss_right.empty and 'Jaccard_similarity' in df_ss_right.columns:
        df_ss_right = df_ss_right.sort_values(by='Jaccard_similarity', ascending=False)
    df_ss_right.to_csv(out_dir / 'semantic_right_deepjoin_wrong.csv', index=False)

    df_dj_false_positives = pd.DataFrame(rows_dj_false_positives)
    if not df_dj_false_positives.empty and 'Jaccard_similarity' in df_dj_false_positives.columns:
        df_dj_false_positives = df_dj_false_positives.sort_values(by='Jaccard_similarity', ascending=False)
    df_dj_false_positives.to_csv(out_dir / 'deepjoin_false_positives.csv', index=False)

    df_ss_false_positives = pd.DataFrame(rows_ss_false_positives)
    if not df_ss_false_positives.empty and 'Jaccard_similarity' in df_ss_false_positives.columns:
        df_ss_false_positives = df_ss_false_positives.sort_values(by='Jaccard_similarity', ascending=False)
    df_ss_false_positives.to_csv(out_dir / 'semantic_false_positives.csv', index=False)

    # Also write simple counts per query
    per_query = []
    for q in sorted(all_queries):
        dj_list = dj.get(q, [])
        ss_list = ss.get(q, [])
        gt_set = gt.get(q, set())
        dj_correct_cnt = sum(1 for cand, _ in dj_list if cand in gt_set)
        ss_correct_cnt = sum(1 for cand, _ in ss_list if cand in gt_set)
        per_query.append({
            'query_key': q,
            'deepjoin_correct': dj_correct_cnt,
            'semantic_correct': ss_correct_cnt,
            'deepjoin_total_preds': len(dj_list),
            'semantic_total_preds': len(ss_list),
            'ground_truth_size': len(gt_set),
        })
    pd.DataFrame(per_query).to_csv(out_dir / 'per_query_counts.csv', index=False)

    print(f"Wrote: {out_dir / 'deepjoin_right_semantic_wrong.csv'}")
    print(f"Wrote: {out_dir / 'semantic_right_deepjoin_wrong.csv'}")
    print(f"Wrote: {out_dir / 'deepjoin_false_positives.csv'}")
    print(f"Wrote: {out_dir / 'semantic_false_positives.csv'}")
    print(f"Wrote: {out_dir / 'per_query_counts.csv'}")


if __name__ == '__main__':
    main()


