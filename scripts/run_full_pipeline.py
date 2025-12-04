import os
import subprocess
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def run(cmd, desc=None, check=True):
    if desc:
        print('---', desc)
    print('>',' '.join(cmd))
    res = subprocess.run(cmd, check=check)
    return res.returncode


def main():
    py = sys.executable

    download_script = os.path.join(REPO_ROOT, 'scripts', 'download_hf_datasets.py')
    if os.path.exists(download_script):
        run([py, download_script, '--outdir', os.path.join('data','hf_raw')], desc='Downloading HF datasets')
    else:
        print('No download_hf_datasets.py found, skipping download step')

    prepare_script = os.path.join(REPO_ROOT, 'scripts', 'prepare_combined.py')
    if os.path.exists(prepare_script):
        run([py, prepare_script], desc='Preparing combined CSVs')
    else:
        print('No prepare_combined.py found, ensure combined CSVs exist under data/ru_toxic')

    train_script = os.path.join(REPO_ROOT, 'scripts', 'train_baseline.py')
    if os.path.exists(train_script):
        run([py, train_script], desc='Training baseline and saving calibrated model')
    else:
        print('No train_baseline.py found, aborting')

    print('\nPipeline finished. Check data/ru_toxic/combined_oof.csv and models/calibrated_model.joblib')


if __name__ == '__main__':
    main()
import argparse
import subprocess
import sys
import os


def run_cmd(cmd):
    print('\n>>> Running:', ' '.join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(f'Command failed: {cmd}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='data/ru_toxic')
    parser.add_argument('--sample_size', type=int, default=5000)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--model_path', default='models/baseline_tfidf_logreg.joblib')
    parser.add_argument('--calibrated_model', default='models/calibrated_model.joblib')
    args = parser.parse_args()

    python = sys.executable

    cmd_prepare = [python, 'scripts/prepare_combined.py', '--out_dir', args.out_dir, '--sample_size', str(args.sample_size)]
    run_cmd(cmd_prepare)

    combined_csv = os.path.join(args.out_dir, 'combined.csv')
    sample_csv = os.path.join(args.out_dir, 'sample_small.csv')

    out_oof = os.path.join(args.out_dir, 'combined_oof.csv')
    cmd_oof = [python, 'scripts/create_oof_soft_labels.py', '--input_csv', combined_csv, '--out_csv', out_oof, '--n_splits', str(args.n_splits), '--sample_size', str(args.sample_size)]
    run_cmd(cmd_oof)
    cmd_train = [python, 'scripts/train_baseline.py', '--model_path', args.model_path]
    run_cmd(cmd_train)
    cmd_cal = [python, 'scripts/calibrate.py', '--model', args.model_path, '--val_csv', sample_csv, '--out_model', args.calibrated_model, '--method', 'isotonic']
    run_cmd(cmd_cal)

    print('\nPipeline completed. Calibrated model saved to', args.calibrated_model)


if __name__ == '__main__':
    main()
