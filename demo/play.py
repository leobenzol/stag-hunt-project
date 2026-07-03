from __future__ import annotations

import argparse
import time
from pathlib import Path

import tf_cpu_only  # noqa: F401
import gymnasium as gym
import gymnasium_stag_hunt  # noqa: F401 — register StagHunt-*-v0
import pygame

from envs.wrappers import NormalizedCoordObs


_SCORE_COLOR_A0 = (40, 90, 220)   # blue — first agent
_SCORE_COLOR_A1 = (220, 40, 40)   # red — second agent
_SCORE_BG = (0, 0, 0)             # opaque black panel behind the numbers


def _draw_scores(score_a0: float, score_a1: float, font: pygame.font.Font) -> None:
    surface = pygame.display.get_surface()
    if surface is None:
        return
    text_a0 = font.render(f"P0: {score_a0:+.0f}", True, _SCORE_COLOR_A0)
    text_a1 = font.render(f"P1: {score_a1:+.0f}", True, _SCORE_COLOR_A1)
    w = max(text_a0.get_width(), text_a1.get_width()) + 12
    h = text_a0.get_height() + text_a1.get_height() + 12
    x = surface.get_width() - w - 6
    y = 6
    pygame.draw.rect(surface, _SCORE_BG, (x, y, w, h))
    surface.blit(text_a0, (x + 6, y + 4))
    surface.blit(text_a1, (x + 6, y + 6 + text_a0.get_height()))
    pygame.display.flip()


_ENV_IDS = {
    "Hunt": "StagHunt-Hunt-v0",
    "Harvest": "StagHunt-Harvest-v0",
    "Escalation": "StagHunt-Escalation-v0",
}


def _build_agent(algo: str, obs_dim: int, action_dim: int, seed: int):
    algo = algo.lower()
    if algo == "idqn":
        from agents.dqn import IDQNAgent
        return IDQNAgent(obs_dim=obs_dim, action_dim=action_dim, seed=seed)
    if algo == "ippo":
        from agents.ppo import IPPOAgent
        return IPPOAgent(obs_dim=obs_dim, action_dim=action_dim, seed=seed)
    if algo == "mappo":
        from agents.mappo import MAPPOAgent
        return MAPPOAgent(obs_dim=obs_dim, action_dim=action_dim, seed=seed)
    raise ValueError(algo)


_PROSOCIAL_ALPHAS: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),  # both selfish
    1: (0.0, 0.5),  # one selfish, one fully prosocial
    2: (0.5, 0.5),  # both fully prosocial
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=list(_ENV_IDS), required=True)
    p.add_argument("--algo", choices=["idqn", "ippo", "mappo"], required=True)
    p.add_argument("--n-prosocial", type=int, choices=[0, 1, 2], default=2,
                   help="Number of prosocial agents: 0=both selfish, 1=one prosocial, "
                        "2=both prosocial.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--steps-per-episode", type=int, default=250)
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--reward-pause-factor", type=float, default=3.0,
                   help="On any frame where some agent gets a non-zero raw reward, "
                        "sleep this many times the normal frame delay to draw attention.")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    args = p.parse_args()

    a0, a1 = _PROSOCIAL_ALPHAS[args.n_prosocial]
    run_dir = args.results_dir / f"{args.env}_{args.algo}_alpha{a0:.2f}-{a1:.2f}_seed{args.seed}"
    if not run_dir.exists():
        raise SystemExit(f"No run at {run_dir}. Train first or fix flags.")

    # Build env with the pygame renderer enabled
    env = gym.make(
        _ENV_IDS[args.env],
        obs_type="coords",
        enable_multiagent=True,
        flip_obs=True,
        load_renderer=True,
        max_timesteps=args.steps_per_episode,
    )
    env = NormalizedCoordObs(env)
    obs_dim = int(env.observation_space.shape[1])
    action_dim = int(env.action_space.n)

    agent = _build_agent(args.algo, obs_dim, action_dim, args.seed)
    agent.load(run_dir / "weights")

    if not pygame.font.get_init():
        pygame.font.init()
    score_font = pygame.font.SysFont("DejaVu Sans Mono, monospace", 22, bold=True)

    delay = 1.0 / args.fps
    pause_delay = delay * args.reward_pause_factor
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep * 100)
        env.render()
        ep_return = [0.0, 0.0]
        _draw_scores(ep_return[0], ep_return[1], score_font)
        for _ in range(args.steps_per_episode):
            a0, a1 = agent.act(obs, greedy=True)
            obs, rewards, term, trunc, info = env.step((a0, a1))
            raw = info.get("raw_rewards", rewards)
            r0, r1 = float(raw[0]), float(raw[1])
            ep_return[0] += r0
            ep_return[1] += r1
            env.render()
            _draw_scores(ep_return[0], ep_return[1], score_font)
            # Pause longer on frames where something happened, makes the demo readable.
            if r0 != 0.0 or r1 != 0.0:
                print(f"  step reward: agent0={r0:+.1f}, agent1={r1:+.1f}")
                time.sleep(pause_delay)
            else:
                time.sleep(delay)
            if term or trunc:
                break
        print(f"Episode {ep}: returns = {ep_return[0]:+.1f}, {ep_return[1]:+.1f} "
              f"(joint {sum(ep_return):+.1f})")
    env.close()


if __name__ == "__main__":
    main()
