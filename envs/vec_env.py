from __future__ import annotations

import numpy as np

from envs.factory import make_env

# Can't use gymnasium "SyncVectorEnv" directly because our envs return a tuple of
# two rewards per step
class SyncVecEnv:
    """Shapes: "reset": (n_envs, 2, obs_dim)
               "step":  (n_envs, 2, obs_dim) obs,  (n_envs, 2) rewards,  (n_envs,) terminated and truncated
    """

    def __init__(self, env_name, n_envs, *, alpha, seed, env_kwargs):
        self.n_envs = int(n_envs)
        # Distinct seeds so the envs are actually different streams
        self.envs = [make_env(env_name, alpha=alpha, seed=seed + i * 10, env_kwargs=env_kwargs)
                     for i in range(self.n_envs)]
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

    def reset(self, seed=None):
        obs = []
        for i, e in enumerate(self.envs):
            o, _ = e.reset(seed=None if seed is None else seed + i)
            obs.append(o)
        return np.stack(obs).astype(np.float32)

    def step(self, actions):
        # Any env that finishes is auto-reset. its pre-reset observation is stored in
        # infos[i]['final_obs'] and the returned obs[i] is the new episode's first obs.
        obs = np.zeros((self.n_envs, *self.single_observation_space.shape), dtype=np.float32)
        rewards = np.zeros((self.n_envs, 2), dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=np.float32)
        truncated = np.zeros(self.n_envs, dtype=np.float32)
        infos = []
        for i, e in enumerate(self.envs):
            o, r, term, trunc, info = e.step((int(actions[i, 0]), int(actions[i, 1])))
            rewards[i] = r
            terminated[i] = float(term)
            truncated[i] = float(trunc)
            if term or trunc:
                info = dict(info)
                info["final_obs"] = np.asarray(o, dtype=np.float32)
                o, _ = e.reset()
            obs[i] = o
            infos.append(info)
        return obs, rewards, terminated, truncated, infos

    def close(self):
        for e in self.envs:
            e.close()
