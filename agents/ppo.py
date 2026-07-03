from __future__ import annotations

from pathlib import Path
import warnings as _warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras

from agents.base import Agent
from agents.heads import build_actor_net, build_critic_net


class ValueNorm:
    """From MAPPO implementation, applied to standard PPO as well

    The critic predicts in a normalized space; targets are normalized before the loss and
    predictions are de-normalized whenever a real value is needed (GAE, bootstrap). Stats
    use a debiased exponential moving average (Adam-style bias correction via _debias)
    """

    def __init__(self, beta: float = 0.99, epsilon: float = 1e-5):
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self._mean = 0.0
        self._mean_sq = 0.0
        self._debias = 0.0

    def update(self, x: np.ndarray) -> None:
        self._mean = self._mean * self.beta + float(np.mean(x)) * (1.0 - self.beta)
        self._mean_sq = self._mean_sq * self.beta + float(np.mean(np.square(x))) * (1.0 - self.beta)
        self._debias = self._debias * self.beta + (1.0 - self.beta)

    @property
    def mean(self) -> float:
        return self._mean / max(self._debias, self.epsilon)

    @property
    def std(self) -> float:
        var = self._mean_sq / max(self._debias, self.epsilon) - self.mean ** 2
        return float(np.sqrt(max(var, self.epsilon)))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class RolloutBuffer:
    def __init__(self, rollout_length: int, n_envs: int, obs_dim: int):
        self.T = int(rollout_length)
        self.N = int(n_envs)
        self.D = int(obs_dim)
        self.obs = np.zeros((self.T, self.N, 2, self.D), dtype=np.float32)
        self.boot_obs = np.zeros((self.T, self.N, 2, self.D), dtype=np.float32)
        self.actions = np.zeros((self.T, self.N, 2), dtype=np.int32)
        self.log_probs = np.zeros((self.T, self.N, 2), dtype=np.float32)
        self.values = np.zeros((self.T, self.N, 2), dtype=np.float32)
        self.rewards = np.zeros((self.T, self.N, 2), dtype=np.float32)
        self.terminated = np.zeros((self.T, self.N), dtype=np.float32)
        self.truncated = np.zeros((self.T, self.N), dtype=np.float32)
        self.t = 0

    def push(self, obs, actions, log_probs, values, rewards,
             terminated, truncated, boot_obs) -> None:
        i = self.t
        self.obs[i] = obs
        self.actions[i] = actions
        self.log_probs[i] = log_probs
        self.values[i] = values
        self.rewards[i] = rewards
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.boot_obs[i] = boot_obs
        self.t += 1

    @property
    def full(self) -> bool:
        return self.t >= self.T

    def reset(self) -> None:
        self.t = 0


class IPPOAgent(Agent):
    eval_greedy = False

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        total_env_steps: int = 200_000,
        n_envs: int = 8,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        rollout_length: int = 128,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        entropy_coef: float = 0.01,
        entropy_coef_final: float = 0.01,
        entropy_hold_frac: float = 0.0,
        max_grad_norm: float = 0.5,
        adam_epsilon: float = 1e-5,
        target_kl: float | None = 0.03,
        hidden_dims: tuple[int, ...] = (64, 64),
        seed: int = 0,
        device: str = "/CPU:0",
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_envs = int(n_envs)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.lr = lr
        self.rollout_length = rollout_length
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef_init = entropy_coef
        self.entropy_coef_final = entropy_coef_final
        self.entropy_hold_frac = float(entropy_hold_frac)
        self.total_env_steps = total_env_steps
        self.max_grad_norm = max_grad_norm
        self.adam_epsilon = adam_epsilon
        self.target_kl = target_kl
        self.hidden_dims = hidden_dims
        self.device = device

        self._rng = np.random.default_rng(seed)
        keras.utils.set_random_seed(seed)

        # One actor grad step per minibatch per epoch per update, LR anneal must match.
        batch = self.n_envs * rollout_length
        updates = max(1, total_env_steps // batch)
        self._actor_gsteps = max(1, updates * ppo_epochs * max(1, batch // minibatch_size))
        with tf.device(device):
            self.actors = [build_actor_net(obs_dim, action_dim, hidden_dims, name=f"actor{i}")
                           for i in range(2)]
            self.actor_opts = [self._make_adam(self._actor_gsteps) for _ in range(2)]
            for i in range(2):
                self.actor_opts[i].build(self.actors[i].trainable_variables)
            self._build_critic_side()

        self.buffer = RolloutBuffer(rollout_length, self.n_envs, obs_dim)
        self._env_steps = 0
        self._pending = None  # (log_probs, values, entropy) saved by act() for observe()

    def _make_adam(self, decay_steps: int) -> keras.optimizers.Optimizer:
        schedule = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.lr, decay_steps=decay_steps,
            end_learning_rate=0.0, power=1.0,
        )
        return keras.optimizers.Adam(learning_rate=schedule, epsilon=self.adam_epsilon)

    # critic overridables for mappo

    def _critic_input_dim(self) -> int:
        return self.obs_dim

    def _build_critic_side(self) -> None:
        self.critics = [build_critic_net(self._critic_input_dim(), self.hidden_dims, name=f"critic{i}")
                        for i in range(2)]
        self.critic_opts = [self._make_adam(self._actor_gsteps) for _ in range(2)]
        for i in range(2):
            self.critic_opts[i].build(self.critics[i].trainable_variables)
        self.value_norms = [ValueNorm() for _ in range(2)]

    def _critic_input(self, obs: np.ndarray, i: int) -> np.ndarray:
        return obs[..., i, :]

    def _critic_for(self, i: int) -> keras.Model:
        return self.critics[i]

    def _critic_opt_for(self, i: int) -> keras.optimizers.Optimizer:
        return self.critic_opts[i]

    def _value_norm_for(self, i: int) -> ValueNorm:
        return self.value_norms[i]

    @property
    def entropy_coef(self) -> float:
        frac = min(1.0, self._env_steps / max(1, self.total_env_steps))
        if frac <= self.entropy_hold_frac:
            return self.entropy_coef_init
        p = (frac - self.entropy_hold_frac) / max(1e-8, 1.0 - self.entropy_hold_frac)
        return self.entropy_coef_init + p * (self.entropy_coef_final - self.entropy_coef_init)


    def act(self, obs: np.ndarray, *, greedy: bool = False, rng=None):
        # Single is true in eval mode.
        single = obs.ndim == 2
        sampler = rng if rng is not None else self._rng
        batch = obs[None] if single else obs            # (N, 2, D)
        N = batch.shape[0]
        actions = np.zeros((N, 2), dtype=np.int32)
        log_probs = np.zeros((N, 2), dtype=np.float32)
        values = np.zeros((N, 2), dtype=np.float32)
        ent = np.zeros(2, dtype=np.float32)
        for i in range(2):
            logits = _tf_forward(self.actors[i], tf.convert_to_tensor(batch[:, i, :])).numpy()
            if greedy:
                a = logits.argmax(axis=-1)
            else:
                g = -np.log(-np.log(sampler.uniform(size=logits.shape) + 1e-12) + 1e-12)
                a = (logits + g).argmax(axis=-1)
            shifted = logits - logits.max(axis=-1, keepdims=True)
            log_softmax = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
            probs = np.exp(log_softmax)
            actions[:, i] = a
            log_probs[:, i] = log_softmax[np.arange(N), a]
            ent[i] = float((-(probs * log_softmax).sum(axis=-1)).mean())
            ci = self._critic_input(batch, i)           # (N, c_dim)
            v = _tf_forward(self._critic_for(i), tf.convert_to_tensor(ci)).numpy()[:, 0]
            values[:, i] = self._value_norm_for(i).denormalize(v)

        if single:
            return int(actions[0, 0]), int(actions[0, 1])
        self._pending = (log_probs, values, float(ent.mean()))
        return actions


    def observe(self, obs, actions, rewards, boot_obs, terminated, truncated) -> dict[str, float]:
        # Push one vectorized timestep (N, ...) and update when the rollout is full.

        assert self._pending is not None, "act() must be called before observe() each step"
        log_probs, values, entropy = self._pending
        self.buffer.push(obs, actions, log_probs, values, rewards, terminated, truncated, boot_obs)
        self._env_steps += self.n_envs
        metrics = {"entropy": entropy}
        if self.buffer.full:
            metrics.update(self._update())
            self.buffer.reset()
        return metrics

    def _bootstrap_all(self) -> np.ndarray:
        T, N = self.rollout_length, self.n_envs
        out = np.zeros((T, N, 2), dtype=np.float32)
        for i in range(2):
            ci = self._critic_input(self.buffer.boot_obs, i).reshape(T * N, -1)
            v = _tf_forward(self._critic_for(i), tf.convert_to_tensor(ci)).numpy()[:, 0]
            out[:, :, i] = self._value_norm_for(i).denormalize(v).reshape(T, N)
        return out

    def _compute_gae(self, boot_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """trace cuts the recursion (A_{t+1}->0) at episode boundaries so advantages
        never leak across episodes. Bootstrap V(s_{t+1}):
          - true terminal: next_value=0, trace=0
          - truncation: bootstrap the stored pre-reset terminal
            value, but still cut the trace (the next step belongs to a fresh episode);
          - last rollout step, not done: bootstrap the value of the next (un-stored) obs;
          - interior step: use the stored value of the actually-observed next state.
        """
        T, N = self.rollout_length, self.n_envs
        buf = self.buffer
        adv = np.zeros((T, N, 2), dtype=np.float32)
        for i in range(2):
            for n in range(N):
                gae = 0.0
                for t in reversed(range(T)):
                    if buf.terminated[t, n]:
                        next_value, trace = 0.0, 0.0
                    elif buf.truncated[t, n]:
                        next_value, trace = boot_values[t, n, i], 0.0
                    elif t == T - 1:
                        next_value, trace = boot_values[t, n, i], 1.0
                    else:
                        next_value, trace = buf.values[t + 1, n, i], 1.0
                    delta = buf.rewards[t, n, i] + self.gamma * next_value - buf.values[t, n, i]
                    gae = delta + self.gamma * self.gae_lambda * trace * gae
                    adv[t, n, i] = gae
        returns = adv + buf.values
        return adv, returns

    def _distinct(self, getter):
        # Return distinct objects by identity so that the mappo critic is only updated once
        out: list = []
        for i in range(2):
            o = getter(i)
            if not any(o is s for s in out):
                out.append(o)
        return out

    def _update(self) -> dict[str, float]:
        T, N = self.rollout_length, self.n_envs
        B = T * N
        boot_values = self._bootstrap_all()
        advantages, returns = self._compute_gae(boot_values)
        ent_coef = tf.constant(self.entropy_coef, dtype=tf.float32)

        # Refresh each distinct value normalizer exactly once.
        for vn in self._distinct(self._value_norm_for):
            agents = [j for j in range(2) if self._value_norm_for(j) is vn]
            vn.update(np.concatenate([returns[:, :, j].reshape(B) for j in agents]))

        # Actor update
        per_agent: list[dict[str, float]] = []
        for i in range(2):
            obs_i = self.buffer.obs[:, :, i, :].reshape(B, self.obs_dim)
            actions_i = self.buffer.actions[:, :, i].reshape(B)
            old_logp_i = self.buffer.log_probs[:, :, i].reshape(B)
            adv_i = advantages[:, :, i].reshape(B)
            actor, actor_opt = self.actors[i], self.actor_opts[i]

            idx = np.arange(B)
            a_losses, ents, kls, clips = [], [], [], []
            epochs_used = self.ppo_epochs
            for epoch in range(self.ppo_epochs):
                self._rng.shuffle(idx)
                epoch_kls = []
                for start in range(0, B, self.minibatch_size):
                    mb = idx[start:start + self.minibatch_size]
                    if mb.size == 0:
                        continue
                    # Advantage normalization at the MINI-BATCH level — PPO37 #7.
                    adv_mb = adv_i[mb]
                    adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                    a_loss, entropy, kl, clipf = _actor_grad_step(
                        actor, actor_opt,
                        tf.convert_to_tensor(obs_i[mb]),
                        tf.convert_to_tensor(actions_i[mb]),
                        tf.convert_to_tensor(old_logp_i[mb]),
                        tf.convert_to_tensor(adv_mb, dtype=tf.float32),
                        clip_eps=self.clip_eps, entropy_coef=ent_coef,
                        max_grad_norm=self.max_grad_norm,
                    )
                    a_losses.append(float(a_loss)); ents.append(float(entropy))
                    kls.append(float(kl)); clips.append(float(clipf)); epoch_kls.append(float(kl))
                # Early stopping, from PPO37 to avoid steps too large
                if self.target_kl is not None and epoch_kls and np.mean(epoch_kls) > self.target_kl:
                    epochs_used = epoch + 1
                    break
            per_agent.append({
                "actor_loss": float(np.mean(a_losses)), "entropy_train": float(np.mean(ents)),
                "approx_kl": float(np.mean(kls)), "clip_fraction": float(np.mean(clips)),
                "epochs_used": float(epochs_used),
            })

        # Critic update
        v_losses: list[float] = []
        for critic in self._distinct(self._critic_for):
            agents = [j for j in range(2) if self._critic_for(j) is critic]
            vn = self._value_norm_for(agents[0])
            critic_opt = self._critic_opt_for(agents[0])
            cin = np.concatenate(
                [self._critic_input(self.buffer.obs, j).reshape(B, -1) for j in agents])
            ret_norm = vn.normalize(
                np.concatenate([returns[:, :, j].reshape(B) for j in agents]))
            old_v_norm = vn.normalize(
                np.concatenate([self.buffer.values[:, :, j].reshape(B) for j in agents]))
            idx = np.arange(cin.shape[0])
            for epoch in range(self.ppo_epochs):
                self._rng.shuffle(idx)
                for start in range(0, idx.size, self.minibatch_size):
                    mb = idx[start:start + self.minibatch_size]
                    if mb.size == 0:
                        continue
                    v_loss = _critic_grad_step(
                        critic, critic_opt,
                        tf.convert_to_tensor(cin[mb]),
                        tf.convert_to_tensor(ret_norm[mb], dtype=tf.float32),
                        tf.convert_to_tensor(old_v_norm[mb], dtype=tf.float32),
                        clip_eps=self.clip_eps, max_grad_norm=self.max_grad_norm,
                    )
                    v_losses.append(float(v_loss))

        out = {k: float(np.mean([d[k] for d in per_agent])) for k in per_agent[0]}
        out["critic_loss"] = float(np.mean(v_losses))
        out["loss"] = out["actor_loss"] + out["critic_loss"]
        return out

    # io

    @staticmethod
    def _base(path: str | Path) -> Path:
        p = Path(path)
        while p.suffix:
            p = p.with_suffix("")
        return p

    def _weight_paths(self, path: str | Path) -> list[tuple[keras.Model, Path]]:
        b = self._base(path)
        pairs = [(self.actors[i], b.parent / f"{b.name}.actor{i}.weights.h5") for i in range(2)]
        pairs += [(self._critic_for(i), b.parent / f"{b.name}.critic{i}.weights.h5") for i in range(2)]
        return pairs

    def save(self, path: str | Path) -> None:
        for model, p in self._weight_paths(path):
            model.save_weights(str(p))

    def load(self, path: str | Path) -> None:
        for model, p in self._weight_paths(path):
            if p.exists():
                model.load_weights(str(p))


# kernels

@tf.function(reduce_retracing=True)
def _tf_forward(model, x):
    return model(x, training=False)


@tf.function(reduce_retracing=True)
def _actor_grad_step(actor, optimizer, obs, actions, old_log_probs, advantages,
                     *, clip_eps: float, entropy_coef, max_grad_norm: float):
    with tf.GradientTape() as tape:
        logits = actor(obs, training=True)
        log_softmax = tf.nn.log_softmax(logits, axis=-1)
        log_probs = tf.gather_nd(
            log_softmax, tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1))
        # Clipped surrogate PPO37 #8.
        ratio = tf.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        # Entropy bonus to encourage exploration PPO37 #10
        probs = tf.nn.softmax(logits, axis=-1)
        entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_softmax, axis=-1))
        loss = actor_loss - entropy_coef * entropy
    # k3 approx_kl estimator E[(r-1) - log r] and clip_fraction, PPO37 #12 debug vars.
    log_ratio = log_probs - old_log_probs
    approx_kl = tf.reduce_mean(ratio - 1.0 - log_ratio)
    clip_fraction = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), clip_eps), tf.float32))
    # Global-norm gradient clip PPO37 #11 
    grads = tape.gradient(loss, actor.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    optimizer.apply_gradients(zip(grads, actor.trainable_variables))
    return actor_loss, entropy, approx_kl, clip_fraction


@tf.function(reduce_retracing=True)
def _critic_grad_step(critic, optimizer, critic_in, returns, old_values,
                      *, clip_eps: float, max_grad_norm: float):
    with tf.GradientTape() as tape:
        # Clipped value loss PPO37 #9
        values = tf.squeeze(critic(critic_in, training=True), axis=-1)
        v_clipped = old_values + tf.clip_by_value(values - old_values, -clip_eps, clip_eps)
        v_loss = 0.5 * tf.reduce_mean(
            tf.maximum(tf.square(returns - values), tf.square(returns - v_clipped)))
    grads = tape.gradient(v_loss, critic.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    optimizer.apply_gradients(zip(grads, critic.trainable_variables))
    return v_loss
