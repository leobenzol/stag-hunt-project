from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.load_runs import load_train_runs


ALGO_COLOR = {"idqn": "#1f77b4", "ippo": "#d62728", "mappo": "#9467bd"}
ALGO_LABEL = {"idqn": "IDQN", "ippo": "IPPO", "mappo": "MAPPO"}
PROSOCIAL_LABEL = {0: "both selfish", 1: "one prosocial", 2: "both prosocial"}
N_TRAIN_BINS = 60


def _mean_ci(values: np.ndarray, ci: float = 0.95) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(values))
    if n == 1:
        return (mean, mean, mean)
    std = float(np.std(values, ddof=1))
    # Normal approximation; with seeds=3 this is rough but report should still note
    # error bars. For more rigour switch to t-distribution.
    z = 1.96 if ci == 0.95 else 1.0
    half = z * std / np.sqrt(n)
    return (mean, mean - half, mean + half)


def _binned_curve(sg: pd.DataFrame, y_col: str, n_bins: int = N_TRAIN_BINS) -> tuple[list, list, list, list]:
    """Mean +/- CI across seeds within env_steps bins.

    Train logs are written once per episode, so each seed's env_steps values
    land at different points (episode lengths vary) and can't be grouped
    exactly; bin first, then aggregate across seeds per bin.
    """
    if sg.empty:
        return [], [], [], []
    x_min, x_max = sg["env_steps"].min(), sg["env_steps"].max()
    edges = np.linspace(x_min, x_max, n_bins + 1) if x_max > x_min else np.array([x_min, x_min + 1])
    sg = sg.copy()
    sg["_bin"] = pd.cut(sg["env_steps"], bins=edges, include_lowest=True)
    per_seed = sg.groupby(["seed", "_bin"], observed=True)[y_col].mean().reset_index()
    steps, mean, lo, hi = [], [], [], []
    for b, grp in per_seed.groupby("_bin", observed=True):
        if grp.empty:
            continue
        m, l, h = _mean_ci(grp[y_col].to_numpy())
        steps.append(b.mid)
        mean.append(m)
        lo.append(l)
        hi.append(h)
    order = np.argsort(steps)
    return ([float(steps[i]) for i in order], [mean[i] for i in order],
            [lo[i] for i in order], [hi[i] for i in order])


def plot_env(df: pd.DataFrame, env: str, out_dir: Path, per_seed: bool = False) -> Path:
    sub = df[df["env"] == env].copy()
    if sub.empty:
        return Path()
    ns = sorted(sub["n_prosocial"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(5 * len(ns), 4), sharey=True)
    if len(ns) == 1:
        axes = [axes]

    for ax, n in zip(axes, ns):
        sa = sub[sub["n_prosocial"] == n]
        for algo, color in ALGO_COLOR.items():
            sg = sa[sa["algo"] == algo]
            if sg.empty:
                continue
            if per_seed:
                # One thin, semi-transparent line per seed so bimodality across
                # seeds is visible rather than averaged away.
                for i, (_seed, ss) in enumerate(sg.groupby("seed")):
                    steps, mean, _, _ = _binned_curve(ss, "joint_return")
                    ax.plot(
                        steps, mean,
                        label=ALGO_LABEL[algo] if i == 0 else None,
                        color=color, linewidth=0.9, alpha=0.5,
                    )
                continue
            steps, mean, lo, hi = _binned_curve(sg, "joint_return")
            ax.plot(steps, mean, label=ALGO_LABEL[algo], color=color, linewidth=2)
            ax.fill_between(steps, lo, hi, color=color, alpha=0.15)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_title(PROSOCIAL_LABEL.get(n, f"{n} prosocial"))
        ax.set_xlabel("env steps")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Train joint return")
    axes[-1].legend(loc="lower right", fontsize=9)
    subtitle = "per-seed" if per_seed else "mean ± 95 % CI over seeds"
    fig.suptitle(f"{env} — train joint return ({subtitle})")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"learning_curves_{env}.png"
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
