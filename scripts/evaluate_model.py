import argparse
import sys
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, classification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to joblib model (supports predict_proba or predict)')
    parser.add_argument('test_csv', help='CSV file with columns `text` and `label`')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold for converting probs to labels')
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    df = pd.read_csv(args.test_csv)
    if 'text' not in df.columns or 'label' not in df.columns:
        print('Test CSV must contain `text` and `label` columns')
        sys.exit(2)

    texts = df['text'].astype(str).tolist()
    y = df['label'].astype(int).values

    try:
        probs = model.predict_proba(texts)[:,1]
    except Exception:
        preds_raw = model.predict(texts)
        if hasattr(preds_raw[0], '__iter__'):
            try:
                probs = [p[1] for p in preds_raw]
            except Exception:
                probs = [1.0 if p==1 else 0.0 for p in preds_raw]
        else:
            probs = [1.0 if p==1 else 0.0 for p in preds_raw]

    preds = [1 if pr >= args.threshold else 0 for pr in probs]

    print('Accuracy:', accuracy_score(y, preds))
    print('Precision:', precision_score(y, preds, zero_division=0))
    print('Recall:', recall_score(y, preds, zero_division=0))
    print('F1:', f1_score(y, preds, zero_division=0))
    try:
        print('ROC AUC:', roc_auc_score(y, probs))
    except Exception:
        pass
    print('Brier score:', brier_score_loss(y, probs))
    print('\nClassification report:\n')
    print(classification_report(y, preds, zero_division=0))


if __name__ == '__main__':
    main()
