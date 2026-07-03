from __future__ import annotations

import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

ENV_BUDGETS = {
    "Hunt": 400_000,
    "Harvest": 1_000_000,
    "Escalation": 500_000,
}

# Prosociality sweep values from the paper
PROSOCIAL_ALPHAS: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),      # both selfish
    1: (0.0, 0.5),      # one selfish, one fully prosocial
    2: (0.5, 0.5),      # both fully prosocial
}

WORKERS = 15

# Exploration values found with a grid search
PPO_EXPLORE: dict[str, dict] = {
    "Hunt":    dict(n_envs=8,  entropy_coef=0.045, entropy_coef_final=0.0, entropy_hold_frac=0.0),
    "Harvest": dict(n_envs=16, entropy_coef=0.05,  entropy_coef_final=0.0, entropy_hold_frac=0.5),
}

DQN_EXPLORE: dict[str, dict] = {
    "Hunt":       dict(epsilon_decay_steps=150_000),
    "Escalation": dict(epsilon_decay_steps=300_000, epsilon_final=0.01),
}

# Reward values from the paper. Middle values from the sweeps were chosen for:
#    mauling_punishment, chance_to_mature and streak_break_punishment_factor
# which let the prosociality effects show by not being too extreme with punishments
# or so easy that all models always find the optimal strategy.
#
# Paper: "Each plant is born ‘young’, then every time step turns ‘mature’
# with probability rmature. While a plant is mature it can die on each time step with probability rdeath.
# The probabilities are always selected such that each plant lives for 20 time steps in expectation"
# 1/mature + 1/die = 20 ==> mature+die=20*mature*die ==> die = mature/(20*mature-1) = 1/(20-1/mature)
ENV_REWARD_ARGS: dict[str, dict] = {
    "Hunt": {"max_timesteps": 250, "stag_reward": 5, "forage_reward": 1, "mauling_punishment": -1},
    "Harvest": {"max_timesteps": 250, "young_reward": 1, "mature_reward": 2, "chance_to_mature": 0.3, "chance_to_die": 1/(20 - 1/0.3)},
    "Escalation": {"max_timesteps": 50, "streak_break_punishment_factor": 0.5},
}

def run_one(env: str, algo: str, alpha: tuple[float, float], seed: int, log_dir_str: str) -> tuple[str, float]:
    # Given the high number of runs and the very simple networks, limit all runs to one thread each to avoid contention.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    import tf_cpu_only  # noqa: F401

    from experiments.runner import RunConfig, train
    steps = ENV_BUDGETS[env]
    cfg = RunConfig(
        env_name=env, algo=algo, alpha=alpha, seed=seed,
        total_env_steps=steps, log_dir=Path(log_dir_str),
        env_kwargs=dict(ENV_REWARD_ARGS[env]),
        agent_kwargs=dict((DQN_EXPLORE if algo == "idqn" else PPO_EXPLORE).get(env, {})),
    )

    t0 = time.perf_counter()
    train(cfg)
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", nargs="+", default=list(ENV_BUDGETS), choices=list(ENV_BUDGETS))
    p.add_argument("--algos", nargs="+", default=["idqn", "ippo", "mappo"],
                   choices=["idqn", "ippo", "mappo"])
    p.add_argument("--prosocial", nargs="+", type=int, default=[0, 1, 2],
                   choices=[0, 1, 2],
                   help="How many of the two agents are fully prosocial")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--log-dir", type=Path, default=Path("results"))
    args = p.parse_args()

    alphas = [PROSOCIAL_ALPHAS[n] for n in args.prosocial]
    combos = list(itertools.product(args.envs, args.algos, alphas, args.seeds))
    total = len(combos)

    print(f"Matrix: {len(args.envs)} envs x {len(args.algos)} algos x "
          f"{len(alphas)} prosocial-config x {len(args.seeds)} seeds = {total} runs")

    t_global = time.perf_counter()

    # Parallel runs
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        futures = {
            pool.submit(run_one, 
                env, algo, alpha, seed, str(args.log_dir)
            ): (env, algo, alpha, seed)
            for env, algo, alpha, seed in combos
        }
        done = 0
        for fut in as_completed(futures):
            env, algo, alpha, seed = futures[fut]
            run_name = f"{env}_{algo}_{f"alpha{alpha[0]:.2f}-{alpha[1]:.2f}"}_seed{seed}"

            try:
                run_time = fut.result()
            except Exception as e:
                done += 1
                print(f"[FAIL {done}/{total}] {run_name}: {e}")
                continue
            done += 1

            # very crude eta, not accounting the different step budgets
            eta = (time.perf_counter() - t_global) / done * (total - done)
            print(f"[{done}/{total}] {run_name} done in {run_time/60:.1f} min | "
                  f"elapsed {(time.perf_counter()-t_global)/3600:.2f} h | "
                  f"ETA {eta/3600:.2f} h")

if __name__ == "__main__":
    main()
