"""SIMPLIFIED stand-in for the 'Sarsa' baseline of Table 1 (Bellemare et al. 2013).

Linear Q(s,a) = w_a . features(s), trained with on-policy Sarsa(0). Features are hand-engineered
(the very thing DQN avoids): the screen is cut into 15x10 blocks and a binary feature marks which
palette colours occur in each block after dropping the background colour (= background subtraction
+ one channel per colour, as described in Sec. 5.3). This is NOT the exact BASS feature set of [3].

  python sarsa_linear.py --game pong --frames 200000
"""
import argparse
import json
import os

import gymnasium as gym
import numpy as np
import ale_py

from dqn.env import GAMES

gym.register_envs(ale_py)
ROWS, COLS = np.meshgrid(np.arange(210) // 14, np.arange(160) // 16, indexing="ij")
BLOCK = (ROWS * 10 + COLS).astype(np.int32)           # 150 blocks
N_FEAT = 150 * 256


def features(gray):
    bg = np.bincount(gray.ravel()).argmax()           # most frequent colour = background
    codes = BLOCK * 256 + gray.astype(np.int32)
    return np.unique(codes[gray != bg])


class LinearSarsa:
    def __init__(self, n_actions, alpha=0.2, gamma=0.99, seed=0):
        self.w = np.zeros((n_actions, N_FEAT), np.float32)
        self.n, self.alpha, self.gamma = n_actions, alpha, gamma
        self.rng = np.random.default_rng(seed)

    def q(self, f):
        return self.w[:, f].sum(1)

    def act(self, f, eps):
        return int(self.rng.integers(self.n)) if self.rng.random() < eps else int(self.q(f).argmax())

    def update(self, f, a, r, f2, a2, done):
        target = r + (0.0 if done else self.gamma * self.q(f2)[a2])
        delta = target - self.q(f)[a]
        self.w[a, f] += self.alpha / max(len(f), 1) * delta


def make_env(game, seed):
    e = gym.make(f"ALE/{GAMES[game]}-v5", obs_type="grayscale", frameskip=3 if game == "spaceinvaders" else 4,
                 repeat_action_probability=0.0, full_action_space=False)
    e.reset(seed=seed)
    return e


def evaluate(agent, game, steps, eps=0.05, seed=7):
    env, scores, ep = make_env(game, seed), [], 0.0
    obs, _ = env.reset()
    for _ in range(steps):
        obs, r, te, tr, _ = env.step(agent.act(features(obs), eps)); ep += r
        if te or tr:
            scores.append(ep); ep = 0.0; obs, _ = env.reset()
    return float(np.mean(scores or [ep])), scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="pong", choices=list(GAMES))
    p.add_argument("--frames", type=int, default=200_000)
    p.add_argument("--eval-steps", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    env = make_env(a.game, a.seed)
    ag = LinearSarsa(env.action_space.n, seed=a.seed)
    k = 3 if a.game == "spaceinvaders" else 4
    obs, _ = env.reset(); f = features(obs); act = ag.act(f, 1.0); frames = 0
    while frames < a.frames:
        eps = max(0.1, 1.0 - 0.9 * frames / (a.frames / 2))
        obs, r, te, tr, _ = env.step(act); frames += k
        f2 = features(obs); act2 = ag.act(f2, eps)
        ag.update(f, act, float(np.sign(r)), f2, act2, te or tr)
        if te or tr:
            obs, _ = env.reset(); f2 = features(obs); act2 = ag.act(f2, eps)
        f, act = f2, act2
    mean, scores = evaluate(ag, a.game, a.eval_steps)
    print(f"{a.game} [sarsa]: mean score {mean:.2f} over {len(scores)} episode(s)")
    os.makedirs(a.out, exist_ok=True)
    json.dump({"game": a.game, "agent": "sarsa", "mean": mean, "scores": scores},
              open(os.path.join(a.out, f"{a.game}_sarsa.json"), "w"))


if __name__ == "__main__":
    main()
