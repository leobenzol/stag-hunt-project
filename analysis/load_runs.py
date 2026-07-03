from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RUN_RE = re.compile(
    r"^(?P<env>[A-Za-z]+)_(?P<algo>[a-z]+)_alpha(?P<a0>[0-9.]+)(?:-(?P<a1>[0-9.]+))?_seed(?P<seed>\d+)$"
)


def parse_run_name(name: str) -> dict | None:
    m = RUN_RE.match(name)
    if not m:
        return None
    g = m.groupdict()
    a0 = float(g["a0"])
    a1 = float(g["a1"]) if g["a1"] is not None else a0
    return {
        "env": g["env"],
        "algo": g["algo"],
        "alpha0": a0,
        "alpha1": a1,
        "n_prosocial": int(a0 > 0) + int(a1 > 0),
        "seed": int(g["seed"]),
    }



def load_train_runs(results_dir: str | Path = "results") -> pd.DataFrame:
    results_dir = Path(results_dir)
    rows = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = parse_run_name(run_dir.name)
        if meta is None:
            continue
        log = run_dir / "train_log.csv"
        if not log.exists() or log.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(log)
        except pd.errors.EmptyDataError:
            continue
        for k, v in meta.items():
            df[k] = v
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)