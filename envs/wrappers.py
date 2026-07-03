from __future__ import annotations

from collections import Counter

import numpy as np
from gymnasium import Wrapper
from gymnasium.spaces import Box

# Stndard normalization to [0, 1] floats to help the NN.
class NormalizedCoordObs(Wrapper):
    def __init__(self, env, binary_cols: tuple[int, ...] = ()):
        super().__init__(env)
        assert env.observation_space.dtype == np.uint8, "expected uint8 coord obs"
        high = float(env.observation_space.high.max())
        scale = high if high > 0 else 1.0
        feat_dim = int(env.observation_space.shape[-1])
        divisor = np.full(feat_dim, scale, dtype=np.float32)
        for c in binary_cols:
            divisor[c] = 1.0
        self._divisor = divisor
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def _convert(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32) / self._divisor

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._convert(obs), info

    def step(self, action):
        obs, rewards, term, trunc, info = self.env.step(action)
        return self._convert(obs), rewards, term, trunc, info


class ProsocialReward(Wrapper):
    """From the prosociality paper: 
        U_i = r'_i = (1 - alpha_i) · r_i + alpha_i · r_j

        The original (r0, r1) tuple is preserved in info['raw_rewards'] 
        so evaluation can always report the raw returns
    """

    def __init__(self, env, alpha: float | tuple[float, float]):
        super().__init__(env)
        a0, a1 = (alpha, alpha) if np.isscalar(alpha) else alpha
        for a in (a0, a1):
            if not 0.0 <= a <= 1.0:
                raise ValueError(f"alpha must be in [0, 1], got {a}")
        self.alphas = (float(a0), float(a1))

    def step(self, action):
        obs, rewards, term, trunc, info = self.env.step(action)
        r0, r1 = float(rewards[0]), float(rewards[1])
        a0, a1 = self.alphas
        shaped = (
            (1.0 - a0) * r0 + a0 * r1,
            (1.0 - a1) * r1 + a1 * r0,
        )
        info = dict(info) if info else {}
        info["raw_rewards"] = np.array([r0, r1], dtype=np.float32)
        return obs, shaped, term, trunc, info


class EpisodeStatsRecorder(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._return_0 = 0.0
        self._return_1 = 0.0
        self._length = 0
        self._reward_counts: Counter[float] = Counter()

    def _reset_accum(self) -> None:
        self._return_0 = 0.0
        self._return_1 = 0.0
        self._length = 0
        self._reward_counts = Counter()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._reset_accum()
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, rewards, term, trunc, info = self.env.step(action)
        info = dict(info) if info else {}

        # Statistics are always on the un-shaped environmental reward (set by
        # ProsocialReward); if no shaping wrapper is present the step's own reward tuple
        # already carries the raw values.
        raw = info.get("raw_rewards")
        if raw is not None:
            r0, r1 = float(raw[0]), float(raw[1])
        else:
            r0, r1 = float(rewards[0]), float(rewards[1])

        self._return_0 += r0
        self._return_1 += r1
        self._length += 1
        self._reward_counts[round(r0, 6)] += 1

        if term or trunc:
            info["episode"] = {
                "return_0": self._return_0,
                "return_1": self._return_1,
                "joint_return": self._return_0 + self._return_1,
                "length": self._length,
                "reward_counts": dict(self._reward_counts),
            }
        return obs, rewards, term, trunc, info
