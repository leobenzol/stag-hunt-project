from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import gymnasium as gym

from agents.base import Agent
from envs.factory import make_env


@dataclass
class RunConfig:
    env_name: str
    algo: str
    alpha: float | tuple[float, float]
    seed: int
    total_env_steps: int
    log_dir: Path
    env_kwargs: dict
    # Extra per-run agent hyperparameters (e.g. per-env PPO exploration settings). Merged
    # into the agent constructor; env-var overrides (tuning sweeps) still take precedence.
    agent_kwargs: dict = field(default_factory=dict)
    progress_every: int = 5_000  # how often to print a one-line progress update


def train(cfg: RunConfig):
    run_dir = cfg.log_dir / _run_id(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    if _completed(run_dir, cfg.total_env_steps):
        print(f"[SKIP] {run_dir.name}: already completed")
        return run_dir

    probe = make_env(cfg.env_name, alpha=cfg.alpha, seed=cfg.seed, env_kwargs=cfg.env_kwargs)
    obs_dim = int(probe.observation_space.shape[1])
    action_dim = int(probe.action_space.n)
    agent_factory = _build_factory(cfg.algo, cfg.seed, cfg.total_env_steps, cfg.agent_kwargs)
    agent = agent_factory(obs_dim, action_dim)

    # DQN does not gain anything from vectorized envs, which actually make
    # the replay buffer episodes more correlated. https://github.com/openai/baselines/issues/711
    if agent.n_envs > 1:
        probe.close()
        _train_vectorized(cfg, agent, run_dir)
        return
    _train_single_stream(cfg, agent, probe, run_dir)


def _train_single_stream(cfg: RunConfig, agent: Agent, env: gym.Env, run_dir: Path):
    train_f, train_w = _open_csv(run_dir / "train_log.csv")

    obs, _ = env.reset(seed=cfg.seed)
    episode_idx = 0
    next_log = cfg.progress_every
    t0 = time.perf_counter()
    try:
        for step in range(1, cfg.total_env_steps + 1):
            a0, a1 = agent.act(obs, greedy=False)
            next_obs, step_rewards, term, trunc, info = env.step((a0, a1))
            # Bootstrapping at a time-limit cutoff is "partial-episode bootstrapping".
            # Pardo et al. 2018, "Time Limits in Reinforcement Learning"
            # Stag hunt never terminates but truncates at the time-limit.
            truncated = bool(term or trunc)
            metrics = agent.observe(obs, (a0, a1), tuple(step_rewards), next_obs,
                                    False, truncated)
            obs = next_obs

            if truncated:
                # Added by EpisodeStatsRecorder wrapper
                ep = info["episode"]
                train_w.writerow({
                    "env_steps": step, "episode": episode_idx,
                    "return_0": ep["return_0"], "return_1": ep["return_1"],
                    "joint_return": ep["joint_return"], "length": ep["length"],
                    **metrics,
                })
                train_f.flush()
                episode_idx += 1
                obs, _ = env.reset()

            if step >= next_log:
                dt = time.perf_counter() - t0
                stats = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
                print(f"[{_run_id(cfg)}] step={step:>7d} ep={episode_idx:>5d} "
                      f"{stats} ({step/dt:.0f} steps/s)", flush=True)
                next_log += cfg.progress_every
    finally:
        train_f.close()
        env.close()
        agent.save(run_dir / "weights")


def _train_vectorized(cfg: RunConfig, agent: Agent, run_dir: Path):
    from envs.vec_env import SyncVecEnv

    n = agent.n_envs
    vec = SyncVecEnv(cfg.env_name, n_envs=n, alpha=cfg.alpha, seed=cfg.seed, env_kwargs=cfg.env_kwargs)
    train_f, train_w = _open_csv(run_dir / "train_log.csv")

    obs = vec.reset(seed=cfg.seed)  # (n, 2, D)
    episode_idx = 0
    t0 = time.perf_counter()
    next_log = cfg.progress_every
    ticks = cfg.total_env_steps // n
    try:
        # Inspired by the pseudo code in the 1st of the 37 details, instead of the double loop,
        # the inner one has been moved into observe to keep the same signature as other agents and hide the rollout buffer
        for tick in range(1, ticks + 1):
            actions = agent.act(obs, greedy=False)            # (n, 2)
            next_obs, rewards, terms, truncs, infos = vec.step(actions)
            done = np.logical_or(terms > 0, truncs > 0)
            boot_obs = next_obs.copy()
            # when a sub-env is done it's reset immediately to not wait for the others,
            # so the true final obs must be stored for bootstrapping 
            for i in range(n):
                if done[i]:
                    boot_obs[i] = infos[i]["final_obs"]
            metrics = agent.observe(obs, actions, rewards, boot_obs,
                                    np.zeros(n, dtype=np.float32), done.astype(np.float32))
            obs = next_obs
            step = tick * n

            wrote = False
            for i in range(n):
                if done[i]:
                    ep = infos[i]["episode"]
                    train_w.writerow({
                        "env_steps": step, "episode": episode_idx,
                        "return_0": ep["return_0"], "return_1": ep["return_1"],
                        "joint_return": ep["joint_return"], "length": ep["length"],
                        **metrics,
                    })
                    episode_idx += 1
                    wrote = True
            if wrote:
                train_f.flush()

            if step >= next_log:
                dt = time.perf_counter() - t0
                stats = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
                print(f"[{_run_id(cfg)}] step={step:>7d} ep={episode_idx:>5d} "
                      f"{stats} ({step/dt:.0f} steps/s)", flush=True)
                next_log += cfg.progress_every
    finally:
        train_f.close()
        vec.close()
        agent.save(run_dir / "weights")


def _build_factory(algo: str, seed: int, total_env_steps: int, agent_kwargs: dict | None = None):
    algo = algo.lower()
    if algo == "idqn":
        from agents.dqn import IDQNAgent
        dqn_kwargs = dict(agent_kwargs or {})
        return lambda obs_dim, action_dim: IDQNAgent(
            obs_dim=obs_dim, action_dim=action_dim, seed=seed, **dqn_kwargs)
    ppo_kwargs = dict(agent_kwargs or {})
    if algo == "ippo":
        from agents.ppo import IPPOAgent
        return lambda obs_dim, action_dim: IPPOAgent(
            obs_dim=obs_dim, action_dim=action_dim, seed=seed,
            total_env_steps=total_env_steps, **ppo_kwargs)
    if algo == "mappo":
        from agents.mappo import MAPPOAgent
        return lambda obs_dim, action_dim: MAPPOAgent(
            obs_dim=obs_dim, action_dim=action_dim, seed=seed,
            total_env_steps=total_env_steps, **ppo_kwargs)
    raise ValueError(f"unknown algorithm {algo!r}")

def _completed(run_dir: Path, expected_steps: int) -> bool:
    log = run_dir / "train_log.csv"
    if not log.exists():
        return False
    try:
        with log.open() as f:
            reader = csv.DictReader(f)
            last_step = 0
            for row in reader:
                last_step = max(last_step, int(float(row["env_steps"])))
        return last_step >= expected_steps
    except Exception:
        return False

def _run_id(cfg: RunConfig) -> str:
    return f"{cfg.env_name}_{cfg.algo}_{f"alpha{cfg.alpha[0]:.2f}-{cfg.alpha[1]:.2f}"}_seed{cfg.seed}"

TRAIN_FIELDS = ["env_steps", "episode", "return_0", "return_1", "joint_return", "length",
                 "loss", "epsilon", "entropy", "approx_kl", "clip_fraction",
                 "actor_loss", "critic_loss"]

def _open_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="")
    w = csv.DictWriter(f, fieldnames=TRAIN_FIELDS, extrasaction="ignore")
    w.writeheader()
    return f, w
