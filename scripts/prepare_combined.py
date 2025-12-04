import argparse
import os
from datasets import load_dataset
import pandas as pd


HF_IDS = [
    "AlexSham/Toxic_Russian_Comments",
    "marriamaslova/toxic_dvach"
]


def detect_text_column(df):
    for col in ['text', 'comment_text', 'comment', 'content', 'message', 'post']:
        if col in df.columns:
            return col
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            return col
    raise ValueError('No text column found')


def detect_label_column(df):
    for col in ['label', 'toxicity', 'toxic', 'target', 'is_toxic']:
        if col in df.columns:
            return col
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
            return col
    raise ValueError('No label column found')


def to_binary_label(series):
    def convert(x):
        if pd.isna(x):
            return 0
        if isinstance(x, (list, tuple)):
            for v in x:
                try:
                    if int(v) == 1:
                        return 1
                except Exception:
                    pass
            return 0
        if isinstance(x, dict):
            for v in x.values():
                try:
                    if int(v) == 1:
                        return 1
                except Exception:
                    pass
            return 0
        if isinstance(x, str):
            if x.strip().lower() in ('1', 'true', 'yes', 'toxic', 'tox'):
                return 1
            try:
                return 1 if float(x) > 0.5 else 0
            except Exception:
                return 0
        try:
            return 1 if int(x) == 1 else 0
        except Exception:
            try:
                return 1 if float(x) > 0.5 else 0
            except Exception:
                return 0

    return series.apply(convert)


def load_and_concat(hf_ids, input_dir=None):
    parts = []
    if input_dir:
        print('Loading local CSVs from', input_dir)
        for root, _, files in os.walk(input_dir):
            for fn in files:
                if not fn.lower().endswith('.csv'):
                    continue
                p = os.path.join(root, fn)
                try:
                    df = pd.read_csv(p)
                except Exception as e:
                    print('Failed to read', p, e)
                    continue
                try:
                    text_col = detect_text_column(df)
                    label_col = detect_label_column(df)
                except Exception as e:
                    print('Skipping', p, 'due to', e)
                    continue
                df2 = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label_raw'})
                df2['label'] = to_binary_label(df2['label_raw'])
                parts.append(df2[['text', 'label']])
    else:
        for hid in hf_ids:
            print('Loading', hid)
            ds = load_dataset(hid)
            if hasattr(ds, 'keys') and 'train' in ds.keys():
                df = ds['train'].to_pandas()
            else:
                if hasattr(ds, 'keys'):
                    first = list(ds.keys())[0]
                    df = ds[first].to_pandas()
                else:
                    df = ds.to_pandas()
            try:
                text_col = detect_text_column(df)
                label_col = detect_label_column(df)
            except Exception as e:
                print('Skipping', hid, 'due to', e)
                continue
            df2 = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label_raw'})
            df2['label'] = to_binary_label(df2['label_raw'])
            parts.append(df2[['text', 'label']])

    if not parts:
        raise RuntimeError('No datasets loaded')
    combined = pd.concat(parts, ignore_index=True)
    combined['text'] = combined['text'].fillna('').astype(str)
    combined = combined[combined['text'].str.strip() != '']
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='data/ru_toxic', help='Output directory')
    parser.add_argument('--sample_size', type=int, default=20000, help='Size of small sample')
    parser.add_argument('--input_dir', default=None, help='Optional: directory with locally downloaded HF CSVs (use this instead of loading from the Hub)')
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    combined = load_and_concat(HF_IDS, input_dir=args.input_dir)
    print('Total rows in combined:', len(combined))
    combined_path = os.path.join(out_dir, 'combined.csv')
    combined.to_csv(combined_path, index=False)
    print('Saved combined to', combined_path)

    n = min(args.sample_size, len(combined))
    sample = combined.sample(n, random_state=42)
    sample_path = os.path.join(out_dir, 'sample_small.csv')
    sample.to_csv(sample_path, index=False)
    print('Saved sample (%d rows) to %s' % (n, sample_path))


if __name__ == '__main__':
    main()
