from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.load_runs import load_train_runs


ALGO_ORDER = ["idqn", "ippo", "mappo"]
ALGO_LABEL = {"idqn": "IDQN", "ippo": "IPPO", "mappo": "MAPPO"}

# Sweep-axis tick labels: how many of the two agents are prosocial (α=0.5).
PROSOCIAL_LABEL = {0: "both\nselfish", 1: "one\nprosocial", 2: "both\nprosocial"}


def converged_returns(df: pd.DataFrame, last_frac: float = 0.2) -> pd.DataFrame:
    """For each (env, algo, n_prosocial, seed), take the mean of joint_return over the
    last ``last_frac`` of env_steps. Returns one row per (env, algo, n_prosocial, seed)."""
    out = []
    for (env, algo, n_prosocial, seed), g in df.groupby(["env", "algo", "n_prosocial", "seed"]):
        g_sorted = g.sort_values("env_steps")
        cutoff = g_sorted["env_steps"].max() * (1 - last_frac)
        tail = g_sorted[g_sorted["env_steps"] >= cutoff]
        if tail.empty:
            continue
        out.append({
            "env": env,
            "algo": algo,
            "n_prosocial": n_prosocial,
            "seed": seed,
            "final_return": tail["joint_return"].mean(),
        })
    return pd.DataFrame(out)


def plot_env_heatmap(df: pd.DataFrame, env: str, out_dir: Path) -> Path:
    sub = df[df["env"] == env]
    if sub.empty:
        return Path()
    ns = sorted(sub["n_prosocial"].unique())
    algos = [a for a in ALGO_ORDER if a in sub["algo"].unique()]
    matrix = np.full((len(algos), len(ns)), np.nan, dtype=np.float32)
    for i, algo in enumerate(algos):
        for j, n in enumerate(ns):
            vals = sub[(sub["algo"] == algo) & (sub["n_prosocial"] == n)]["final_return"].values
            if len(vals) > 0:
                matrix[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(0.9 * len(ns) + 4, 0.6 * len(algos) + 2))
    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanmax(matrix))
    # symmetric colormap around 0, Stag-Hunt returns are signed.
    vabs = max(abs(vmin), abs(vmax), 1e-6)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([PROSOCIAL_LABEL.get(n, str(n)) for n in ns])
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels([ALGO_LABEL[a] for a in algos])
    for i in range(len(algos)):
        for j in range(len(ns)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        color="white" if abs(v) > vabs * 0.45 else "black", fontsize=9)
    ax.set_title(f"{env} — converged joint return (mean over last 20 % of training, then over seeds)")
    fig.colorbar(im, ax=ax, label="joint return")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"alpha_heatmap_{env}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("figures"))
    p.add_argument("--last-frac", type=float, default=0.2)
    args = p.parse_args()

    df = load_train_runs(args.results)
    if df.empty:
        print("No train data yet.")
        return
    conv = converged_returns(df, last_frac=args.last_frac)
    for env in sorted(conv["env"].unique()):
        path = plot_env_heatmap(conv, env, args.out)
        if path != Path():
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
