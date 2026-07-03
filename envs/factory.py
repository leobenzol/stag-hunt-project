from __future__ import annotations

from typing import Any

import gymnasium as gym
import gymnasium_stag_hunt  # noqa: F401 — registers StagHunt-*-v0 envs

from envs.wrappers import (
    EpisodeStatsRecorder,
    NormalizedCoordObs,
    ProsocialReward,
)


_ENV_IDS = {
    "Hunt": "StagHunt-Hunt-v0",
    "Harvest": "StagHunt-Harvest-v0",
    "Escalation": "StagHunt-Escalation-v0",
}


def _binary_obs_cols(name: str, feat_dim: int) -> tuple[int, ...]:
    # Columns of the coord observation that are binary flags, not coordinates.
    if name == "Harvest":
        n_plants = (feat_dim - 4) // 3
        return tuple(6 + 3 * k for k in range(n_plants))
    return ()


def make_env(
    name: str,
    *,
    alpha: float | tuple[float, float] = 0.0,
    seed: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> gym.Env:
    if name not in _ENV_IDS:
        raise ValueError(f"unknown env {name!r}; expected one of {sorted(_ENV_IDS)}")
    env_kwargs = dict(env_kwargs or {})
    env_kwargs.setdefault("obs_type", "coords")
    env_kwargs.setdefault("enable_multiagent", True)
    env_kwargs.setdefault("flip_obs", True)

    env = gym.make(_ENV_IDS[name], **env_kwargs)
    binary_cols = _binary_obs_cols(name, env.observation_space.shape[-1])
    env = NormalizedCoordObs(env, binary_cols=binary_cols)
    env = ProsocialReward(env, alpha=alpha)
    env = EpisodeStatsRecorder(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
