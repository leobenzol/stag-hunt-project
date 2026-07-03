from __future__ import annotations

import math
from collections.abc import Sequence

from tensorflow import keras
from tensorflow.keras import layers


def build_mlp_layers(
    hidden_dims: Sequence[int] = (64, 64),
    activation: str = "tanh",
    kernel_initializer=None,
    init_scale: float = math.sqrt(2.0),
    name: str = "trunk",
) -> list[layers.Dense]:
    # PPO37 #2 and #13 
    if kernel_initializer is None:
        kernel_initializer = keras.initializers.Orthogonal(gain=init_scale)
    return [
        layers.Dense(
            units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer="zeros",
            name=f"{name}_dense_{i}",
        )
        for i, units in enumerate(hidden_dims)
    ]


def build_dqn_net(
    obs_dim: int,
    action_dim: int,
    hidden_dims: Sequence[int] = (64, 64),
    name: str = "qnet",
) -> keras.Model:
    inputs = keras.Input(shape=(obs_dim,), name=f"{name}_obs")
    h = inputs
    for layer in build_mlp_layers(
        hidden_dims, activation="relu", kernel_initializer="he_normal", name=f"{name}_trunk"):
        h = layer(h)
    q = layers.Dense(
        action_dim,
        activation=None,
        kernel_initializer="he_normal",
        name=f"{name}_q",
    )(h)
    return keras.Model(inputs, q, name=name)


def build_actor_net(
    obs_dim: int,
    action_dim: int,
    hidden_dims: Sequence[int] = (64, 64),
    name: str = "actor",
) -> keras.Model:
    inputs = keras.Input(shape=(obs_dim,), name=f"{name}_obs")
    h = inputs
    for layer in build_mlp_layers(hidden_dims, activation="tanh", name=f"{name}_trunk"):
        h = layer(h)
    logits = layers.Dense(
        action_dim,
        activation=None,
        kernel_initializer=keras.initializers.Orthogonal(gain=0.01),
        bias_initializer="zeros",
        name=f"{name}_logits",
    )(h)
    return keras.Model(inputs, logits, name=name)


def build_critic_net(
    state_dim: int,
    hidden_dims: Sequence[int] = (64, 64),
    name: str = "critic",
) -> keras.Model:
    inputs = keras.Input(shape=(state_dim,), name=f"{name}_state")
    h = inputs
    for layer in build_mlp_layers(hidden_dims, activation="tanh", name=f"{name}_trunk"):
        h = layer(h)
    v = layers.Dense(
        1,
        activation=None,
        kernel_initializer=keras.initializers.Orthogonal(gain=1.0),
        bias_initializer="zeros",
        name=f"{name}_value",
    )(h)
    return keras.Model(inputs, v, name=name)
