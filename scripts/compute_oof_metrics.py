import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'data/ru_toxic/combined_oof.csv'
print('Reading', path)
df = pd.read_csv(path)
if 'soft_label' not in df.columns or 'label' not in df.columns:
    raise RuntimeError('CSV must contain columns: label, soft_label')

p = df['soft_label'].astype(float).values
y = df['label'].astype(int).values
th = 0.5
pred = (p >= th).astype(int)

acc = accuracy_score(y, pred)
prec = precision_score(y, pred, zero_division=0)
rec = recall_score(y, pred, zero_division=0)
f1 = f1_score(y, pred, zero_division=0)
auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')
brier = brier_score_loss(y, p)

print(f'Accuracy: {acc:.4f}')
print(f'Precision: {prec:.4f}')
print(f'Recall: {rec:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'ROC AUC: {auc:.4f}')
print(f'Brier score: {brier:.4f}')
