import os
import argparse
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
import joblib


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/ru_toxic/combined.csv')
    parser.add_argument('--fallback', default='data/ru_toxic/sample_small.csv')
    parser.add_argument('--oof_out', default='data/ru_toxic/combined_oof.csv')
    parser.add_argument('--model_out', default='models/calibrated_model.joblib')
    parser.add_argument('--n_splits', type=int, default=5)
    args = parser.parse_args()

    inp = args.input if os.path.exists(args.input) else args.fallback
    if not os.path.exists(inp):
        raise FileNotFoundError(f'No input CSV found at {args.input} or fallback {args.fallback}')

    df = pd.read_csv(inp)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise RuntimeError('Input CSV must contain `text` and `label` columns')

    X = df['text'].astype(str).values
    y = df['label'].astype(int).values

    base_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=2000, solver='lbfgs'))
    ])

    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    calibrator = CalibratedClassifierCV(base_pipe, method='sigmoid', cv=5)

    print('Computing OOF probabilities with cross-validation...')
    oof_probs = cross_val_predict(calibrator, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:,1]

    df_out = df.copy()
    df_out['soft_label'] = oof_probs

    ensure_dir(args.oof_out)
    df_out.to_csv(args.oof_out, index=False)
    print('Saved OOF csv to', args.oof_out)

    print('Fitting calibrated model on full data...')
    calibrator.fit(X, y)
    ensure_dir(args.model_out)
    joblib.dump(calibrator, args.model_out)
    print('Saved calibrated model to', args.model_out)

    try:
        brier = brier_score_loss(y, oof_probs)
        auc = roc_auc_score(y, oof_probs) if len(np.unique(y))>1 else float('nan')
        print(f'OOF Brier score: {brier:.4f}')
        print(f'OOF ROC AUC: {auc:.4f}')
    except Exception as e:
        print('Failed to compute metrics:', e)


if __name__ == '__main__':
    main()
