"""Evaluate a policy the way the paper does (Sec. 5.3): eps-greedy with eps=0.05 for a fixed number of steps.

  python evaluate.py --game pong --agent random
  python evaluate.py --game pong --agent dqn --checkpoint runs/pong/best.pt
  python evaluate.py --game seaquest --agent dqn --checkpoint runs/seaquest/best.pt --viz-q   # Figure-3 style plot
"""
import argparse
import json
import os

import cv2
import numpy as np

from dqn.agent import DQNAgent
from dqn.env import AtariEnv, GAMES
from dqn.utils import run_eval, set_seed


def visualize_q(env, agent, out_dir, steps=300, eps=0.05):
    """Analogue of Figure 3: track max_a Q(s,a) along an episode and save screenshots at the peak / trough."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s, qs, shots = env.reset(), [], []
    for _ in range(steps):
        qs.append(float(agent.q_values(s[None]).max())); shots.append(env.last_raw.copy())
        s, _, term, trunc = env.step(agent.act(s, eps))
        if term or trunc:
            break
    lo, hi = int(np.argmin(qs)), int(np.argmax(qs))
    plt.figure(figsize=(7, 3)); plt.plot(qs); plt.scatter([lo, hi], [qs[lo], qs[hi]], c="r", zorder=3)
    plt.xlabel("agent step"); plt.ylabel("max Q"); plt.title("Predicted value along an episode (red: shown frames)")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "q_value_trace.png"), dpi=120); plt.close()
    for name, i in (("low_q", lo), ("high_q", hi)):
        cv2.imwrite(os.path.join(out_dir, f"{name}_step{i}.png"), cv2.cvtColor(shots[i], cv2.COLOR_RGB2BGR))
    print(f"saved Q-value trace + screenshots to {out_dir} (min Q {qs[lo]:.2f} @ {lo}, max Q {qs[hi]:.2f} @ {hi})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="pong", choices=list(GAMES))
    p.add_argument("--agent", default="dqn", choices=["dqn", "random"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--eval-steps", type=int, default=10_000)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--viz-q", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    set_seed(a.seed)
    env = AtariEnv(a.game, seed=a.seed)
    if a.agent == "random":
        rng = np.random.default_rng(a.seed)
        act = lambda s: int(rng.integers(env.n_actions))
        agent = None
    else:
        assert a.checkpoint, "--checkpoint required for --agent dqn"
        agent = DQNAgent(env.n_actions, a.device, seed=a.seed); agent.load(a.checkpoint)
        act = lambda s: agent.act(s, a.epsilon)
    mean, scores = run_eval(env, act, a.eval_steps)
    print(f"{a.game} [{a.agent}]: mean score {mean:.2f} over {len(scores)} episode(s) -> {np.round(scores, 1).tolist()}")
    os.makedirs(a.out, exist_ok=True)
    json.dump({"game": a.game, "agent": a.agent, "mean": mean, "scores": scores, "eval_steps": a.eval_steps},
              open(os.path.join(a.out, f"{a.game}_{a.agent}.json"), "w"))
    if a.viz_q and agent is not None:
        visualize_q(env, agent, a.out)


if __name__ == "__main__":
    main()
