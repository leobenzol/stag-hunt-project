from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.load_runs import load_train_runs
from analysis.plot_learning_curves import _binned_curve, ALGO_LABEL, PROSOCIAL_LABEL


ALGOS = ["idqn", "ippo", "mappo"]
PROSOCIAL_COLOR = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}


def plot_env(df: pd.DataFrame, env: str, out_dir: Path, per_seed: bool = False) -> Path:
    sub = df[df["env"] == env].copy()
    if sub.empty:
        return Path()
    algos = [a for a in ALGOS if a in set(sub["algo"].unique())]
    if not algos:
        return Path()
    fig, axes = plt.subplots(1, len(algos), figsize=(5 * len(algos), 4), sharey=True)
    if len(algos) == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        sa = sub[sub["algo"] == algo]
        for n in sorted(sa["n_prosocial"].unique()):
            sg = sa[sa["n_prosocial"] == n]
            if sg.empty:
                continue
            color = PROSOCIAL_COLOR.get(n, None)
            label = PROSOCIAL_LABEL.get(n, f"{n} prosocial")
            if per_seed:
                # One thin, semi-transparent line per seed so bimodality across
                # seeds is visible rather than averaged away.
                for i, (_seed, ss) in enumerate(sg.groupby("seed")):
                    steps, mean, _, _ = _binned_curve(ss, "joint_return")
                    ax.plot(
                        steps, mean,
                        label=label if i == 0 else None,
                        color=color, linewidth=0.9, alpha=0.5,
                    )
                continue
            steps, mean, lo, hi = _binned_curve(sg, "joint_return")
            ax.plot(steps, mean, label=label, color=color, linewidth=2)
            ax.fill_between(steps, lo, hi, color=color, alpha=0.15)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_title(ALGO_LABEL.get(algo, algo))
        ax.set_xlabel("env steps")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Train joint return")
    axes[-1].legend(loc="lower right", fontsize=9)
    subtitle = "per-seed" if per_seed else "mean ± 95 % CI over seeds"
    fig.suptitle(f"{env} — train joint return by prosociality ({subtitle})")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"prosociality_curves_{env}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("figures"))
    p.add_argument("--envs", nargs="+", default=None, help="Subset of envs to plot.")
    p.add_argument("--per-seed", action="store_true",
                   help="Skip mean/CI; draw one thin line per seed to reveal bimodality.")
    args = p.parse_args()

    df = load_train_runs(args.results)
    if df.empty:
        print("No train data yet; nothing to plot.")
        return
    envs = args.envs or sorted(df["env"].unique())
    for env in envs:
        path = plot_env(df, env, args.out, per_seed=args.per_seed)
        if path != Path():
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
