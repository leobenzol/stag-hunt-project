from __future__ import annotations

import numpy as np
from tensorflow import keras

from agents.ppo import IPPOAgent, ValueNorm
from agents.heads import build_critic_net


class MAPPOAgent(IPPOAgent):
    def _critic_input_dim(self) -> int:
        return 2 * self.obs_dim

    def _build_critic_side(self) -> None:
        # A single centralized critic and value-norm, shared by both agents. Its update
        # goes over 2*B samples, so it takes 2x the gradient steps an IPPO critic does,
        # LR anneal sized accordingly.
        self.central_critic = build_critic_net(self._critic_input_dim(), self.hidden_dims,
                                               name="central_critic")
        self.central_critic_opt = self._make_adam(2 * self._actor_gsteps)
        self.central_critic_opt.build(self.central_critic.trainable_variables)
        self.central_value_norm = ValueNorm()

    def _critic_input(self, obs: np.ndarray, i: int) -> np.ndarray:
        # Both agents see the same joint observation (..., 2, D) -> (..., 2D)."""
        lead = obs.shape[:-2]
        return obs.reshape(*lead, 2 * self.obs_dim)

    def _critic_for(self, i: int) -> keras.Model:
        return self.central_critic

    def _critic_opt_for(self, i: int) -> keras.optimizers.Optimizer:
        return self.central_critic_opt

    def _value_norm_for(self, i: int) -> ValueNorm:
        return self.central_value_norm

    def _weight_paths(self, path):
        b = self._base(path)
        return [
            (self.actors[0], b.parent / f"{b.name}.actor0.weights.h5"),
            (self.actors[1], b.parent / f"{b.name}.actor1.weights.h5"),
            (self.central_critic, b.parent / f"{b.name}.central_critic.weights.h5"),
        ]
