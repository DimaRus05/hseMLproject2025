import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oof_csv', required=True, help='Path to CSV with `text`, `label`, `soft_label`')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for converting soft_label to predicted labels')
    args = parser.parse_args()

    df = pd.read_csv(args.oof_csv)
    if not {'label','soft_label'}.issubset(df.columns):
        raise SystemExit('CSV must contain columns `label` and `soft_label`')

    y_true = df['label'].astype(int).values
    probs = df['soft_label'].astype(float).values
    preds = (probs >= args.threshold).astype(int)

    print('Accuracy:', accuracy_score(y_true, preds))
    print('Precision:', precision_score(y_true, preds, zero_division=0))
    print('Recall:', recall_score(y_true, preds, zero_division=0))
    print('F1:', f1_score(y_true, preds, zero_division=0))
    try:
        print('ROC AUC:', roc_auc_score(y_true, probs))
    except Exception:
        pass
    print('Brier Score:', brier_score_loss(y_true, probs))


if __name__ == '__main__':
    main()
