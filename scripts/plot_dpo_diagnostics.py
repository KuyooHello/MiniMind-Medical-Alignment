"""Plot fixed-pair probes and training-window diagnostics from the recorded CSV."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/dpo_diagnostics.json')
    parser.add_argument('--csv', default='results/dpo_diagnostic_metrics.csv')
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text())
    with Path(args.csv).open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    labels = {'diag_dpo_sum': 'Sum', 'diag_dpo_mean': 'Mean', 'diag_dpo_sum_nll': 'Sum + NLL',
              'diag_dpo_mean_nll': 'Mean + NLL', 'diag_dpo_mean_nll_filtered': 'Mean + NLL, filtered'}
    colors = ['#6489aa', '#bd554f', '#b89048', '#397966', '#795f9b']
    panels = [
        ('probe', 'chosen_nll', 'Chosen NLL on the fixed 32 pairs', 'Mean response NLL (nats/token)'),
        ('probe', 'rejected_nll', 'Rejected NLL on the fixed 32 pairs', 'Mean response NLL (nats/token)'),
        ('probe', 'chosen_token_kl_policy_ref', 'Chosen-context token KL(policy || ref)', 'Mean conditional token KL'),
        ('train_window', 'preference_loss', 'Preference loss: pair-weighted windows', 'Preference loss'),
        ('train_window', 'grad_norm_before_clip', 'Gradient norm before clipping', 'Mean norm per update (log scale)'),
        ('probe', 'preference_accuracy', 'Relative reward ordering on fixed training pairs', 'Fraction with reward margin > 0'),
    ]
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 170, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for axis, (event, metric, title, ylabel) in zip(axes.flat, panels):
        for result, color in zip(report['results'], colors):
            selected = [row for row in rows if row['run'] == result['id'] and row['event'] == event]
            axis.plot([int(row['step']) for row in selected], [float(row[metric]) for row in selected],
                      color=color, label=labels[result['id']], marker='o' if event == 'probe' else None,
                      markersize=3, linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel('Training step')
        axis.set_ylabel(ylabel)
        axis.grid(axis='y', alpha=.18)
    first_probe = report['results'][0]['initial_probe']['metrics']
    for axis, metric in zip(axes[0, :2], ['ref_chosen_nll', 'ref_rejected_nll']):
        axis.axhline(first_probe[metric], linestyle='--', color='#777777', linewidth=1, label='Frozen reference')
    axes[1, 1].set_yscale('log')
    axes[1, 1].axhline(1.0, linestyle='--', color='#777777', linewidth=1, label='Clip threshold')
    axes[1, 2].set_ylim(-.03, 1.03)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for extension in ('png', 'svg'):
        fig.savefig(f'images/dpo_diagnostics.{extension}')
    plt.close(fig)
    svg = Path('images/dpo_diagnostics.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print('Figures -> images/dpo_diagnostics.png and .svg')


if __name__ == '__main__':
    main()
