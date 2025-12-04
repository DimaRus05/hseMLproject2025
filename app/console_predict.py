"""Simple console application to classify Russian text as toxic / not toxic.

Usage:
    python app\console_predict.py --model models/baseline_tfidf_logreg.joblib

The script loads a saved sklearn pipeline (TF-IDF + classifier) and interacts via stdin.
"""
import argparse
import joblib


def interactive(model_path):
    print('Loading model from', model_path)
    model = joblib.load(model_path)
    print('Model loaded. Enter text lines (empty line to exit).')
    while True:
        try:
            text = input('> ')
        except EOFError:
            break
        if not text or text.strip() == '':
            print('Exiting.')
            break
        prob = None
        try:
            prob = model.predict_proba([text])[0][1]
        except Exception:
            pred = model.predict([text])[0]
            prob = 1.0 if pred == 1 else 0.0
        pct = round(100 * prob, 1)
        label = 'TOXIC' if prob >= 0.5 else 'NOT_TOXIC'
        print(f'{label} (prob={pct}%)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/baseline_tfidf_logreg.joblib', help='Path to saved model')
    args = parser.parse_args()
    interactive(args.model)
