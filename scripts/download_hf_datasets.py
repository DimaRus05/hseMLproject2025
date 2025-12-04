import os
import argparse
from datasets import load_dataset
import pandas as pd


DATASET_IDS = [
    'AlexSham/Toxic_Russian_Comments',
    'marriamaslova/toxic_dvach'
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_text_column(df: pd.DataFrame):
    candidates = ['text', 'comment', 'comments', 'sentence', 'content', 'post', 'body', 'comment_text']
    cols = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand in cols:
            return df.columns[cols.index(cand)]
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return None


def find_label_column(df: pd.DataFrame):
    candidates = ['label', 'labels', 'toxic', 'target', 'is_toxic', 'annotation']
    cols = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand in cols:
            return df.columns[cols.index(cand)]
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
            uniq = df[c].dropna().unique()
            if len(uniq) <= 3:
                return c
    return None


def standardize_df(df: pd.DataFrame):
    text_col = find_text_column(df)
    label_col = find_label_column(df)
    if text_col is None:
        return None
    if label_col is None:
        return None
    out = pd.DataFrame()
    out['text'] = df[text_col].astype(str)
    lab = df[label_col]
    if pd.api.types.is_float_dtype(lab) or pd.api.types.is_integer_dtype(lab):
        if lab.max() <= 1.0 and lab.min() >= 0.0:
            out['label'] = (lab >= 0.5).astype(int)
        else:
            uniques = sorted([u for u in pd.unique(lab) if pd.notna(u)])
            if len(uniques) == 2:
                mapping = {uniques[0]: 0, uniques[1]: 1}
                out['label'] = lab.map(mapping).astype(int)
            else:
                return None
    else:
        uniques = pd.unique(lab.dropna())
        if len(uniques) == 2:
            mapping = {uniques[0]: 0, uniques[1]: 1}
            out['label'] = lab.map(mapping).astype(int)
        else:
            return None
    return out


def save_dataset(dataset_id: str, outdir: str):
    print(f'Loading {dataset_id} ...')
    try:
        ds = load_dataset(dataset_id)
    except Exception as e:
        print(f'Failed to load {dataset_id}:', e)
        return

    base = os.path.join(outdir, dataset_id.replace('/', '_'))
    ensure_dir(base)

    if isinstance(ds, dict):
        for split, d in ds.items():
            df = pd.DataFrame(d)
            p = os.path.join(base, f'{split}.csv')
            df.to_csv(p, index=False)
            print('Saved', p, 'rows=', len(df))
            std = standardize_df(df)
            if std is not None:
                std.to_csv(os.path.join(base, 'standardized.csv'), index=False)
    else:
        df = pd.DataFrame(ds)
        p = os.path.join(base, 'data.csv')
        df.to_csv(p, index=False)
        print('Saved', p, 'rows=', len(df))
        std = standardize_df(df)
        if std is not None:
            std.to_csv(os.path.join(base, 'standardized.csv'), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='data/hf_raw', help='Output directory for downloaded datasets')
    parser.add_argument('--which', nargs='*', default=None, help='Optional list of dataset IDs to download (overrides default list)')
    args = parser.parse_args()

    ensure_dir(args.outdir)
    ids = args.which if args.which else DATASET_IDS
    for did in ids:
        try:
            save_dataset(did, args.outdir)
        except Exception as e:
            print('Error processing', did, e)


if __name__ == '__main__':
    main()
