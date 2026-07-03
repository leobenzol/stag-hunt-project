from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.load_runs import load_train_runs
from analysis.plot_alpha_heatmap import converged_returns

_ALGOS = ["idqn", "ippo", "mappo"]
_COLORS = {"idqn": "#4C72B0", "ippo": "#C44E52", "mappo": "#8172B3"}
_PROSOCIAL_LABEL = {0: "both selfish", 1: "one prosocial", 2: "both prosocial"}


def plot_escape_fraction(conv, env: str, out_dir: Path, threshold: float) -> Path:
    sub = conv[conv["env"] == env]
    ns = sorted(sub["n_prosocial"].unique())
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    width = 0.8 / len(_ALGOS)
    x = np.arange(len(ns))
    for j, algo in enumerate(_ALGOS):
        fracs, labels = [], []
        for n in ns:
            vals = sub[(sub["algo"] == algo) & (sub["n_prosocial"] == n)]["final_return"].values
            n_seeds = len(vals)
            n_coop = int(np.sum(vals >= threshold))
            fracs.append(n_coop / n_seeds if n_seeds else 0.0)
            labels.append(f"{n_coop}/{n_seeds}" if n_seeds else "")
        xpos = x + (j - (len(_ALGOS) - 1) / 2) * width
        bars = ax.bar(xpos, fracs, width, label=algo.upper(), color=_COLORS[algo])
        for rect, lab in zip(bars, labels):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02, lab,
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([_PROSOCIAL_LABEL.get(n, str(n)) for n in ns])
    ax.set_xlabel("prosociality (number of prosocial agents)")
    ax.set_ylabel(f"escape fraction (return $\\geq$ {threshold:.0f})")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"{env}: stag-coordination escape fraction")
    ax.legend(ncol=4, loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"escape_fraction_{env}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("figures"))
    p.add_argument("--env", default="Hunt")
    p.add_argument("--coop-threshold", type=float, default=400.0)
    args = p.parse_args()

    conv = converged_returns(load_train_runs(args.results))
    if conv.empty:
        print("No train data yet.")
        return
    plot_escape_fraction(conv, args.env, args.out, args.coop_threshold)


if __name__ == "__main__":
    main()
