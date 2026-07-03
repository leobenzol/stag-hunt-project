from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from agents.base import Agent
from agents.replay_buffer import JointReplayBuffer
from agents.heads import build_dqn_net


class IDQNAgent(Agent):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        gamma: float = 0.99,
        lr: float = 5e-4,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay_steps: int = 60_000,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        learning_starts: int = 1_000,
        train_every: int = 4,
        target_sync_every: int = 1_000,
        double_q: bool = True,
        hidden_dims: tuple[int, ...] = (64, 64),
        seed: int = 0,
        device: str = "/CPU:0",
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_every = train_every
        self.target_sync_every = target_sync_every
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay_steps = epsilon_decay_steps
        self.double_q = double_q
        self.device = device

        self._rng = np.random.default_rng(seed)
        keras.utils.set_random_seed(seed)

        # One online + target network per agent.
        with tf.device(device):
            self.qnets = [build_dqn_net(obs_dim, action_dim, hidden_dims, name=f"q{i}")
                          for i in range(2)]
            self.target_qnets = [build_dqn_net(obs_dim, action_dim, hidden_dims, name=f"tq{i}")
                                 for i in range(2)]
            for i in range(2):
                self.target_qnets[i].set_weights(self.qnets[i].get_weights())
            self.optimizers = [keras.optimizers.Adam(learning_rate=lr) for _ in range(2)]
            for i in range(2):
                self.optimizers[i].build(self.qnets[i].trainable_variables)

        self.buffer = JointReplayBuffer(buffer_capacity, (2, obs_dim), self._rng)
        self._env_steps = 0
        self._grad_steps = 0
        self._train_fns = [self._make_agent_train_fn(i) for i in range(2)]

    # compilation
    def _make_agent_train_fn(self, i: int):
        qnet, target_qnet, optimizer = self.qnets[i], self.target_qnets[i], self.optimizers[i]
        gamma, double_q = self.gamma, self.double_q

        @tf.function(reduce_retracing=True)
        def train(obs, actions, rewards, next_obs, terminated):
            # Bootstrapped TD target r + gamma*max(next) with the target net theta' forming
            # the bootstrap (DQN2013 sec 2 eq.2-3); Double-DQN selection/evaluation split is
            # inside _next_state_value (DDQN2016 eq.4).
            next_q = _next_state_value(qnet, target_qnet, next_obs, double_q)
            target = rewards + gamma * next_q * (1.0 - terminated)
            with tf.GradientTape() as tape:
                q_taken = _taken_q(qnet, obs, actions)
                # smooth-L1 loss on the TD error. 
                loss = tf.reduce_mean(keras.losses.huber(target[:, None], q_taken[:, None]))
            grads = tape.gradient(loss, qnet.trainable_variables)
            # stabilization inspired by PPO 
            grads, _ = tf.clip_by_global_norm(grads, 10.0)
            optimizer.apply_gradients(zip(grads, qnet.trainable_variables))
            return loss

        return train

    # ------------------------------------------------------------------ acting

    @property
    def epsilon(self) -> float:
        # epsilon linear decay
        frac = min(1.0, self._env_steps / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_final - self.epsilon_start)

    def act(self, obs: np.ndarray, *, greedy: bool = False, rng=None) -> tuple[int, int]:
        eps = 0.0 if greedy else self.epsilon
        sampler = rng if rng is not None else self._rng
        actions = []
        for i in range(2):
            if eps > 0.0 and sampler.random() < eps:
                actions.append(int(sampler.integers(0, self.action_dim)))
            else:
                q = self._forward_q(i, tf.convert_to_tensor(obs[i:i + 1])).numpy()[0]
                actions.append(int(np.argmax(q)))
        return actions[0], actions[1]

    def _forward_q(self, i: int, obs: tf.Tensor) -> tf.Tensor:
        return _eval_forward(self.qnets[i], obs)

    # ----------------------------------------------------------------- learning

    def observe(self, obs, actions, rewards, next_obs, terminated, truncated) -> dict[str, float]:
        # Only a true terminal zeros the bootstrap, truncation keeps it.
        self.buffer.push(obs, actions, rewards, next_obs, terminated)
        self._env_steps += 1

        metrics: dict[str, float] = {"epsilon": self.epsilon}
        if len(self.buffer) >= self.learning_starts and self._env_steps % self.train_every == 0:
            metrics["loss"] = float(self._train_step())
            self._grad_steps += 1
            # Periodic target-network sync: copy online -> target every target_sync_every
            # gradient steps so the bootstrap target is held fixed between syncs 
            if self._grad_steps % self.target_sync_every == 0:
                for i in range(2):
                    self.target_qnets[i].set_weights(self.qnets[i].get_weights())
        return metrics

    def _train_step(self) -> float:
        return float(np.mean([self._train_agent(i) for i in range(2)]))

    def _train_agent(self, i: int) -> float:
        batch = self.buffer.sample(self.batch_size)  # each agent samples independently
        loss = self._train_fns[i](
            tf.convert_to_tensor(batch["obs"][:, i]),
            tf.convert_to_tensor(batch["actions"][:, i]),
            tf.convert_to_tensor(batch["rewards"][:, i]),
            tf.convert_to_tensor(batch["next_obs"][:, i]),
            tf.convert_to_tensor(batch["terminated"]),
        )
        return float(loss)


    @classmethod
    def _weight_paths(cls, path: str | Path) -> list[Path]:
        base = Path(path)
        while base.suffix:
            base = base.with_suffix("")
        return [base.parent / f"{base.name}.q{i}.weights.h5" for i in range(2)]

    def save(self, path: str | Path) -> None:
        for i, p in enumerate(self._weight_paths(path)):
            self.qnets[i].save_weights(str(p))

    def load(self, path: str | Path) -> None:
        for i, p in enumerate(self._weight_paths(path)):
            self.qnets[i].load_weights(str(p))
            self.target_qnets[i].set_weights(self.qnets[i].get_weights())


# kernels

@tf.function(reduce_retracing=True)
def _eval_forward(net, obs):
    return net(obs, training=False)


def _next_state_value(qnet, target_qnet, next_obs, double_q):
    """Double-DQN (DDQN2016 eq.4): the ONLINE net qnet (theta) selects the greedy next
    action argmax_a Q(s',a;theta), and the TARGET net target_qnet (theta') *evaluates* that
    action Q(s', a*; θ')
    """
    if double_q:
        next_actions = tf.argmax(qnet(next_obs, training=False), axis=-1, output_type=tf.int32)
        idx = tf.stack([tf.range(tf.shape(next_actions)[0]), next_actions], axis=1)
        return tf.gather_nd(target_qnet(next_obs, training=False), idx)
    return tf.reduce_max(target_qnet(next_obs, training=False), axis=-1)


def _taken_q(qnet, obs, actions):
    # Q-values of the actions actually taken
    q_all = qnet(obs, training=True)
    idx = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
    return tf.gather_nd(q_all, idx)
