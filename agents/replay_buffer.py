from __future__ import annotations

import numpy as np


# DQN samples independently per agent and trains each agent on its own column
# Agents do not share parameters, just a memory layout convenience.
class JointReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], rng: np.random.Generator):
        self._cap = int(capacity)
        self._rng = rng
        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((capacity, 2), dtype=np.int32)
        self._rewards = np.zeros((capacity, 2), dtype=np.float32)
        self._terminated = np.zeros((capacity,), dtype=np.float32)
        self._size = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        obs: np.ndarray,
        actions: tuple[int, int],
        rewards: tuple[float, float],
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        i = self._ptr
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._actions[i] = actions
        self._rewards[i] = rewards
        # Store the true terminal flag (not time limit truncation) so the TD target
        # bootstraps correctly: V(next) is dropped only at episode ends.
        self._terminated[i] = float(terminated)
        self._ptr = (i + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Uniform random batch with keys ``obs, actions, rewards, next_obs, terminated``."""
        idx = self._rng.integers(0, self._size, size=batch_size)
        return {
            "obs": self._obs[idx],                  # (B, 2, D)
            "actions": self._actions[idx],          # (B, 2)
            "rewards": self._rewards[idx],          # (B, 2)
            "next_obs": self._next_obs[idx],        # (B, 2, D)
            "terminated": self._terminated[idx],    # (B,)
        }
