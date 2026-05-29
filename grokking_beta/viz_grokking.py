"""Visualize grokking β trajectory : is β jump SHARP at transition?"""
from __future__ import annotations
import json, subprocess, sys
import matplotlib.pyplot as plt

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#000000', 'axes.facecolor': '#000000',
    'savefig.facecolor': '#000000', 'axes.edgecolor': '#444444',
    'axes.labelcolor': '#cccccc', 'xtick.color': '#999999',
    'ytick.color': '#999999', 'text.color': '#ffffff', 'font.family': 'monospace',
})
COL = {'train': '#00ff9f', 'val': '#ff00ff', 'beta_a': '#ffaa00', 'beta_w': '#88ccff'}


def fetch_results() -> list:
    """Pull results JSON via modal fetch_all (dumps to stdout as repr won't work; use volume get)."""
    # Use modal to run a tiny dump
    out = subprocess.run(
        ["modal", "run", "grokking_beta/modal_grokking_beta.py::dump_json"],
        capture_output=True, text=True, cwd="/Users/zac/claude-code-projects/metamorphic-systems",
    )
    # Parse JSON line
    for line in out.stdout.splitlines():
        if line.startswith("JSON_DUMP:"):
            return json.loads(line[len("JSON_DUMP:"):])
    return []


def plot(results):
    n = len(results)
    fig, axs = plt.subplots(n, 1, figsize=(12, 4 * n))
    if n == 1:
        axs = [axs]
    for ax, r in zip(axs, results):
        traj = r['trajectory']
        steps = [t['step'] for t in traj]
        tr = [t['train_acc'] for t in traj]
        va = [t['val_acc'] for t in traj]
        ba = [t['beta_act'] for t in traj]
        gs = r['grok_step']

        ax.plot(steps, tr, color=COL['train'], lw=1.5, label='train_acc')
        ax.plot(steps, va, color=COL['val'], lw=1.5, label='val_acc')
        ax.set_xscale('log')
        ax.set_ylabel('accuracy')
        ax.set_xlabel('step (log)')
        if gs:
            ax.axvline(gs, color='#ffffff', ls='--', alpha=0.5, label=f'grok @ {gs}')

        ax2 = ax.twinx()
        ax2.plot(steps, ba, color=COL['beta_a'], lw=2, label='β_act (eigenspectrum)')
        ax2.set_ylabel('β_act', color=COL['beta_a'])

        ax.set_title(f"seed{r['seed']} wd{r['weight_decay']} — grokking β signature",
                     color='#ffffff', pad=10)
        l1, lab1 = ax.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, loc='center right',
                  facecolor='#111111', edgecolor='#333333', fontsize=9)
        ax.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig('grokking_beta/fig_grokking_beta.png', dpi=140, bbox_inches='tight')
    print("Saved grokking_beta/fig_grokking_beta.png")


if __name__ == "__main__":
    res = fetch_results()
    if not res:
        print("No results")
        sys.exit(1)
    plot(res)
