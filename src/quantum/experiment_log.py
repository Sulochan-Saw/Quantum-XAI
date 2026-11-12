"""Experiment log helpers.

results/experiment_logs/LOG.md is the TEAM's shared table (handbook 6.9). train.py and sanity_check.py add one
row per run through log_run(); existing rows are never touched.

    python -m src.quantum.experiment_log      # writes results/experiment_logs/member4_setup.md
                                              # (environment, data, settings, rules: your own file, no conflicts)
"""
import datetime
import json
import os
import platform
import subprocess
from importlib import metadata

from src.config import CFG

LOG_PATH = 'results/experiment_logs/LOG.md'
SETUP_PATH = 'results/experiment_logs/member4_setup.md'
HEADER = ('| Date | Who | Exp ID | Command | Git commit | Seed | Output file | Notes |\n'
          '| --- | --- | --- | --- | --- | --- | --- | --- |\n')


def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M')


def git_commit():
    try:
        out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL, text=True)
        return out.strip() or 'n/a'
    except Exception:
        return 'n/a (not a git repo)'


def log_run(exp_id, command, seed, output_file, notes, who='Member 4'):
    """Add ONE row to the team's LOG.md table (handbook 6.9). Existing rows are never changed."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    exists = os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0
    needs_newline = False
    if exists:
        with open(LOG_PATH, 'rb') as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b'\n'
    cells = [now(), who, exp_id, '`' + command + '`', git_commit(), str(seed), output_file,
             str(notes).replace('|', '/').replace('\n', ' ')]
    with open(LOG_PATH, 'a') as f:
        if not exists:
            f.write(HEADER)
        elif needs_newline:
            f.write('\n')
        f.write('| ' + ' | '.join(cells) + ' |\n')


def _ver(pkg):
    try:
        return metadata.version(pkg)
    except Exception:
        return 'not installed'


def header():
    meta_path = 'data/processed/meta.json'
    names_path = 'data/processed/feature_names.json'
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    names = json.load(open(names_path)) if os.path.exists(names_path) else []
    nq = CFG['n_qubits']
    lines = [
        '# Member 4 setup: quantum hybrid model',
        '',
        'Created ' + now() + '. One row per run is in the team table results/experiment_logs/LOG.md.',
        '',
        '## Environment',
        '- Platform: ' + platform.platform() + ' | CPUs: ' + str(os.cpu_count()),
        '- Python ' + platform.python_version() + ' | PennyLane ' + _ver('pennylane') + ' | torch ' + _ver('torch')
        + ' | numpy ' + _ver('numpy') + ' | scikit-learn ' + _ver('scikit-learn'),
        '- Simulator only (PennyLane `default.qubit`, exact expectation values). No hardware, no noise in these runs.',
        '',
        '## Data (real data, produced by `src.preprocessing.clean_data`)',
        '- Drug: ' + str(meta.get('drug', 'unknown')) + ' | target: LN_IC50',
        '- Cell lines: total ' + str(meta.get('n_total', '?')) + ' = train ' + str(meta.get('n_train', '?'))
        + ' / val ' + str(meta.get('n_val', '?')) + ' / test ' + str(meta.get('n_test', '?'))
        + ' (split seed ' + str(meta.get('seed', CFG['seed'])) + ')',
        '- Expression: ' + str(meta.get('expression', 'unknown')),
        '- 8 selected genes (ranked by F-score on train): ' + ', '.join(names),
        '- Genes fed to the circuit (first ' + str(nq) + ', one per qubit): ' + ', '.join(names[:nq]),
        '',
        '## Model and training settings (from config.yaml)',
        '- Angle embedding RY(pi*x), ' + str(CFG['n_layers']) + ' StronglyEntanglingLayers, ' + str(nq)
        + ' qubits, Z expectation on every qubit, linear head (' + str(nq) + ' -> 1)',
        '- Adam, lr ' + str(CFG['lr']) + ', batch ' + str(CFG['batch_size']) + ', epochs ' + str(CFG['epochs'])
        + ', loss MSE on train-standardised LN_IC50',
        '- Training seeds: ' + str(CFG['train_seeds']),
        '',
        '## Rules followed',
        '- Best checkpoint chosen by validation loss only. Test set used once per run, after training.',
        '- Synthetic data is for debugging only and never appears in a report or slide.',
        '- No hyper-parameter was changed after seeing test scores.',
        '',
        '## Open items',
        '- Drug choice must be agreed with Member 2 and written into config.yaml (`drug_name`).',
        '- Classical baselines (Member 3) use 8 genes; the circuit uses the first ' + str(nq)
        + '. Ask Member 1 / 3 for a ' + str(nq) + '-gene classical comparison.',
    ]
    return '\n'.join(lines) + '\n\n'


if __name__ == '__main__':
    os.makedirs(os.path.dirname(SETUP_PATH), exist_ok=True)
    open(SETUP_PATH, 'w').write(header())
    print('wrote', SETUP_PATH)
