from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis.load_runs import load_train_runs
from analysis.plot_alpha_heatmap import converged_returns


def summary_table(conv: pd.DataFrame, coop_threshold: float = 400.0) -> pd.DataFrame:
    rows = []
    for (env, algo, n_prosocial), g in conv.groupby(["env", "algo", "n_prosocial"]):
        vals = g["final_return"].values
        n = len(vals)
        mean = float(vals.mean()) if n else float("nan")
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        ci = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        frac = float(np.mean(vals >= coop_threshold)) if n else float("nan")
        rows.append({
            "env": env, "algo": algo, "n_prosocial": n_prosocial,
            "n_seeds": n, "mean": mean, "std": std, "ci95_half": ci,
            "frac_coop": frac, "n_coop": int(np.sum(vals >= coop_threshold)),
        })
    return pd.DataFrame(rows).sort_values(["env", "algo", "n_prosocial"]).reset_index(drop=True)


def il_vs_ctde(conv: pd.DataFrame) -> pd.DataFrame:
    pairs = [("ippo", "mappo", "PPO-family")]
    rows = []
    for env in sorted(conv["env"].unique()):
        for n_prosocial in sorted(conv[conv["env"] == env]["n_prosocial"].unique()):
            base = conv[(conv["env"] == env) & (conv["n_prosocial"] == n_prosocial)]
            for il, ctde, family in pairs:
                a = base[base["algo"] == il]["final_return"].values
                b = base[base["algo"] == ctde]["final_return"].values
                if len(a) < 2 or len(b) < 2:
                    p = float("nan")
                    u = float("nan")
                else:
                    res = stats.mannwhitneyu(a, b, alternative="two-sided")
                    p = float(res.pvalue)
                    u = float(res.statistic)
                rows.append({
                    "env": env, "n_prosocial": n_prosocial, "family": family,
                    "il_algo": il, "ctde_algo": ctde,
                    "n_il": len(a), "n_ctde": len(b),
                    "il_mean": float(a.mean()) if len(a) else float("nan"),
                    "ctde_mean": float(b.mean()) if len(b) else float("nan"),
                    "U": u, "p_value": p,
                })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("tables"))
    p.add_argument("--coop-threshold", type=float, default=400.0,
                   help="Return above which a seed counts as having reached the cooperative "
                        "(stag) equilibrium; used for the frac_coop escape fraction on Hunt.")
    args = p.parse_args()

    df = load_train_runs(args.results)
    if df.empty:
        print("No train data yet.")
        return
    conv = converged_returns(df)

    summ = summary_table(conv, coop_threshold=args.coop_threshold)
    print("=== Per-(env, algo, n_prosocial) converged joint return ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    print("\n=== IL vs CTDE (Mann–Whitney U, two-sided) ===")
    test = il_vs_ctde(conv)
    print(test.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    args.out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(args.out / "summary.csv", index=False)
    test.to_csv(args.out / "il_vs_ctde.csv", index=False)
    print(f"\nWritten to {args.out / 'summary.csv'} and {args.out / 'il_vs_ctde.csv'}.")


if __name__ == "__main__":
    main()
