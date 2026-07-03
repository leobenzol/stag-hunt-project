from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Agent(ABC):
    # True for DQN, false for PPO.
    eval_greedy: bool = True
    # 1 for DQN, >1 for PPO.
    n_envs: int = 1

    @abstractmethod
    def act(self, obs: np.ndarray, *, greedy: bool = False, rng=None) -> tuple[int, int]:
        pass

    @abstractmethod
    def observe(
        self,
        obs: np.ndarray,
        actions: tuple[int, int],
        rewards: tuple[float, float],
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, float]:
        """All three envs are continuing tasks, so every episode ends by
        truncation and terminated is always False.
        Returns metrics for logging. May be empty between gradient steps.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save learned weights to path."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load learned weights from path."""
