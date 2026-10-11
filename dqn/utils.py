import csv
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def linear_epsilon(frame, start=1.0, end=0.1, anneal_frames=1_000_000):
    """Paper: epsilon annealed linearly 1 -> 0.1 over the first million frames, then fixed."""
    return end if frame >= anneal_frames else start + (end - start) * frame / anneal_frames


def run_eval(env, act_fn, n_steps):
    """Paper protocol: run the policy for a fixed number of steps, average the (unclipped) episode scores."""
    scores, ep, state = [], 0.0, env.reset()
    for _ in range(n_steps):
        state, r, term, trunc = env.step(act_fn(state))
        ep += r
        if term or trunc:
            scores.append(ep); ep, state = 0.0, env.reset()
    if not scores:                      # no episode finished inside the budget -> report partial
        scores.append(ep)
    return float(np.mean(scores)), scores


def collect_heldout_states(env, n=500, seed=0):
    """Fixed set of states gathered with a random policy before training (Sec. 5.1)."""
    rng = np.random.default_rng(seed)
    states, s = [], env.reset()
    while len(states) < n:
        s, _, term, trunc = env.step(int(rng.integers(env.n_actions)))
        states.append(s)
        if term or trunc:
            s = env.reset()
    return np.stack(states)


class CSVLogger:
    def __init__(self, path, fields):
        self.path, self.fields = path, fields
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    def log(self, **kw):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([kw[k] for k in self.fields])


def plot_training(csv_path, out_path, title=""):
    """Analogue of Figure 2: noisy average reward vs. smooth average max-Q on held-out states."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ep, [float(r["avg_reward"]) for r in rows], marker="o"); ax[0].set_title(f"Average reward per episode {title}")
    ax[1].plot(ep, [float(r["avg_max_q"]) for r in rows], marker="o", color="C1"); ax[1].set_title("Average max Q (held-out states)")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
