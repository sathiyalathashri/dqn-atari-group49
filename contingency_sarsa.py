#!/usr/bin/env python
"""
Simplified "Contingency" baseline (Table 1 of Mnih et al. 2013, Bellemare et al. 2012).

The paper describes Contingency as: the same linear Sarsa learner as the "Sarsa"
baseline, but with the feature set augmented by a *learned* representation of the
parts of the screen that are under the agent's control.

This file is a SIMPLIFIED re-implementation of that idea (not Bellemare et al.'s
exact method), in three phases:

  Phase 1  Contingency learning (random policy, `--contingency-steps` agent steps)
           For every screen block, measure how much its pixel changes depend on the
           action taken (between-action std of the mean change). Blocks whose
           change statistics depend strongly on the action form the "control mask".
           The mode colour of every pixel over these frames is the background
           (background subtraction, as the paper says both baselines use).

  Phase 2  Linear Sarsa(lambda) with sparse binary features
           (a) base features      : (block, colour) presence, 14x16 blocks x 128 colours
           (b) avatar position    : one-hot of the block the controlled object is in
           (c) avatar-relative    : (relative block offset, colour) presence in a
                                    (2R+1)x(2R+1) window centred on the avatar
           The avatar is located each step as the centroid of newly changed
           foreground pixels inside the control mask.
           (b)+(c) are the "contingency" features; `--no-contingency` drops them,
           giving a plain Sarsa with identical (a)-features for an ablation.

  Phase 3  Evaluation with eps = 0.05 for a fixed number of agent steps
           (average episode score, raw/unclipped, as in Sec. 5.3 of the paper).

Known limitations: the avatar locator is a heuristic (other objects that move
inside the control mask can disturb it); no previous-frame features; the exact
numbers will not match the paper.
"""
import argparse
import csv
import json
import math
import os
import time
from collections import deque

import numpy as np

# ----------------------------------------------------------------------------
# Screen geometry / feature layout
# ----------------------------------------------------------------------------
H, W = 210, 160          # Atari screen
GH, GW = 14, 16          # block grid
BH, BW = H // GH, W // GW  # 15 x 10 pixels per block
NC = 128                 # NTSC palette size
BLOCK_ID = (np.arange(H)[:, None] // BH * GW + np.arange(W)[None, :] // BW).astype(np.int32)

GAME_IDS = {
    "breakout": "Breakout", "pong": "Pong", "enduro": "Enduro", "qbert": "Qbert",
    "seaquest": "Seaquest", "space_invaders": "SpaceInvaders",
    "spaceinvaders": "SpaceInvaders", "beam_rider": "BeamRider", "beamrider": "BeamRider",
}


# ----------------------------------------------------------------------------
# Environment wrapper: returns palette-index screens (210x160, values 0..127)
# ----------------------------------------------------------------------------
class AtariPalette:
    def __init__(self, game, frame_skip=None, max_frames=108000):
        import gymnasium as gym
        try:
            import ale_py
            if hasattr(gym, "register_envs"):
                gym.register_envs(ale_py)
        except Exception:
            pass
        self.name = GAME_IDS.get(game.lower(), game)
        self.env = gym.make(
            f"ALE/{self.name}-v5", frameskip=1, repeat_action_probability=0.0,
            full_action_space=False, max_num_frames_per_episode=max_frames,
        )
        self.ale = self.env.unwrapped.ale
        self.k = frame_skip or (3 if self.name == "SpaceInvaders" else 4)
        self.n_actions = int(self.env.action_space.n)

    def _screen(self):
        s = None
        try:
            s = np.asarray(self.ale.getScreen())
            if s.ndim == 3:
                s = s[..., 0]
        except Exception:
            s = None
        if s is None or s.shape != (H, W):
            s = np.asarray(self.ale.getScreenGrayscale())
            if s.ndim == 3:
                s = s[..., 0]
        return (s >> 1).astype(np.uint8) & (NC - 1)

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        return self._screen()

    def step(self, a):
        total, done = 0.0, False
        for _ in range(self.k):
            _, r, term, trunc, _ = self.env.step(a)
            total += r
            if term or trunc:
                done = True
                break
        return self._screen(), float(total), done


# ----------------------------------------------------------------------------
# Phase 1: contingency model (control mask + background)
# ----------------------------------------------------------------------------
class ContingencyModel:
    def __init__(self, n_actions, thresh=0.3, bg_sub=True):
        self.nA, self.thresh, self.bg_sub = n_actions, thresh, bg_sub
        self._chg = np.zeros((n_actions, GH, GW), np.float64)
        self._n = np.zeros(n_actions, np.int64)
        self._bgc = np.zeros((NC, H * W), np.uint16)
        self._pix = np.arange(H * W)
        self.bg = np.full((H, W), 255, np.uint8)     # 255 never matches a colour
        self.score = np.zeros((GH, GW))
        self.mask_blocks = np.zeros((GH, GW), bool)
        self.mask_pix = np.zeros((H, W), bool)

    def observe(self, prv, nxt, a):
        changed = prv != nxt
        self._chg[a] += changed.reshape(GH, BH, GW, BW).mean(axis=(1, 3))
        self._n[a] += 1
        self._bgc[nxt.ravel(), self._pix] += 1

    def fit(self):
        used = self._n > 0
        if used.sum() < 2:
            raise RuntimeError("Need samples for at least 2 actions to learn contingency.")
        mean_change = self._chg[used] / self._n[used][:, None, None]
        self.score = mean_change.std(axis=0)               # how much action matters per block
        mx = self.score.max()
        self.mask_blocks = (self.score >= self.thresh * mx) if mx > 1e-9 else np.zeros((GH, GW), bool)
        self.mask_pix = np.kron(self.mask_blocks, np.ones((BH, BW), bool)).astype(bool)
        if self.bg_sub:
            self.bg = self._bgc.argmax(axis=0).reshape(H, W).astype(np.uint8)
        return self

    def locate(self, prv, nxt, last):
        """Avatar position (y, x) in pixels, or `last` if nothing moved inside the mask."""
        if not self.mask_blocks.any():
            return last
        ch = (prv != nxt) & self.mask_pix
        if self.bg_sub:
            new = ch & (nxt != self.bg)       # pixels the avatar has just moved onto
            if new.any():
                ch = new
        if not ch.any():
            return last
        ys, xs = np.nonzero(ch)
        return (float(ys.mean()), float(xs.mean()))

    def save(self, path):
        np.savez_compressed(path, score=self.score, mask=self.mask_blocks, bg=self.bg,
                            thresh=self.thresh)


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
class Features:
    def __init__(self, model, window=3, use_contingency=True):
        self.m, self.R, self.use_c = model, window, use_contingency
        self.n_base = GH * GW * NC
        self.n_pos = GH * GW + 1                      # last one = "avatar not located yet"
        self.n_rel = (2 * window + 1) ** 2 * NC
        self.n_features = self.n_base + (self.n_pos + self.n_rel if use_contingency else 0)

    def __call__(self, frame, pos):
        fg = frame != self.m.bg if self.m.bg_sub else np.ones_like(frame, bool)
        present = np.zeros(GH * GW * NC, bool)
        present[BLOCK_ID[fg] * NC + frame[fg]] = True
        base = np.flatnonzero(present)
        if not self.use_c:
            return base
        if pos is None:
            return np.concatenate([base, [self.n_base + GH * GW]]).astype(np.int64)
        by = min(max(int(pos[0]) // BH, 0), GH - 1)
        bx = min(max(int(pos[1]) // BW, 0), GW - 1)
        R = self.R
        padded = np.pad(present.reshape(GH, GW, NC), ((R, R), (R, R), (0, 0)))
        rel = np.flatnonzero(padded[by:by + 2 * R + 1, bx:bx + 2 * R + 1, :])
        return np.concatenate([
            base, [self.n_base + by * GW + bx], self.n_base + self.n_pos + rel
        ]).astype(np.int64)


# ----------------------------------------------------------------------------
# Phase 2: linear Sarsa(lambda) (truncated accumulating traces over sparse features)
# ----------------------------------------------------------------------------
class SarsaLambda:
    def __init__(self, n_actions, n_features, alpha=0.1, gamma=0.99, lam=0.9):
        self.nA, self.alpha, self.gamma, self.lam = n_actions, alpha, gamma, lam
        self.W = np.zeros((n_features, n_actions), np.float32)
        gl = gamma * lam
        trace_len = 1 if gl <= 0 else int(np.clip(math.ceil(math.log(0.02) / math.log(gl)), 1, 50))
        self.trace = deque(maxlen=trace_len)

    def q(self, idx):
        return self.W[idx].sum(axis=0)

    def reset_trace(self):
        self.trace.clear()

    def update(self, idx, a, r, idx2, a2, done):
        target = r if done else r + self.gamma * self.W[idx2, a2].sum()
        delta = target - self.W[idx, a].sum()
        self.trace.appendleft((idx, a))
        c, gl = 1.0, self.gamma * self.lam
        for i_k, a_k in self.trace:
            self.W[i_k, a_k] += self.alpha * delta * c / len(i_k)
            c *= gl
        return delta


def eps_greedy(learner, idx, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(learner.nA))
    q = learner.q(idx)
    return int(rng.choice(np.flatnonzero(q >= q.max() - 1e-8)))


# ----------------------------------------------------------------------------
# Loops
# ----------------------------------------------------------------------------
def collect(env, model, steps, rng, seed=None):
    obs = env.reset(seed=seed)
    for _ in range(steps):
        a = int(rng.integers(env.n_actions))
        nxt, _, done = env.step(a)
        model.observe(obs, nxt, a)
        obs = env.reset() if done else nxt
    return obs


def train(env, model, fx, learner, total_frames, eps_cfg, rng, log_path=None, log_every=50000,
          seed=None):
    eps0, eps1, anneal = eps_cfg
    eps_at = lambda f: eps1 if f >= anneal else eps0 + (eps1 - eps0) * f / anneal
    obs, pos = env.reset(seed=seed), None
    idx = fx(obs, pos)
    a = eps_greedy(learner, idx, eps_at(0), rng)
    frames, ep_ret, scores, t0, next_log = 0, 0.0, [], time.time(), log_every
    writer = None
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        f = open(log_path, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["frames", "episodes", "eps", "avg_score_last20"])
    while frames < total_frames:
        nxt, r, done = env.step(a)
        frames += env.k
        ep_ret += r
        pos = model.locate(obs, nxt, pos)
        eps = eps_at(frames)
        if done:
            learner.update(idx, a, float(np.sign(r)), idx, a, True)
            scores.append(ep_ret)
            ep_ret = 0.0
            obs, pos = env.reset(), None
            learner.reset_trace()
            idx = fx(obs, pos)
            a = eps_greedy(learner, idx, eps, rng)
        else:
            idx2 = fx(nxt, pos)
            a2 = eps_greedy(learner, idx2, eps, rng)
            learner.update(idx, a, float(np.sign(r)), idx2, a2, False)
            obs, idx, a = nxt, idx2, a2
        if frames >= next_log:
            avg = float(np.mean(scores[-20:])) if scores else float("nan")
            print(f"frames {frames:>9d}  eps {eps:.3f}  episodes {len(scores):>5d}  "
                  f"avg(last20) {avg:8.2f}  {frames / (time.time() - t0):.0f} fps")
            if writer:
                writer.writerow([frames, len(scores), f"{eps:.4f}", f"{avg:.3f}"])
                f.flush()
            next_log += log_every
    if writer:
        f.close()
    return scores


def evaluate(env, model, fx, learner, steps, eps, rng, seed=None):
    obs, pos = env.reset(seed=seed), None
    scores, ep = [], 0.0
    for _ in range(steps):
        a = eps_greedy(learner, fx(obs, pos), eps, rng)
        nxt, r, done = env.step(a)
        ep += r
        pos = model.locate(obs, nxt, pos)
        if done:
            scores.append(ep)
            ep, obs, pos = 0.0, env.reset(), None
        else:
            obs = nxt
    if not scores:                      # no finished episode within the step budget
        scores.append(ep)
    return float(np.mean(scores)), len(scores)


# ----------------------------------------------------------------------------
# Diagnostics plot
# ----------------------------------------------------------------------------
def save_diagnostic(model, frame, pos, path, title=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(model.score, cmap="viridis")
    ax[0].set_title("action-dependence score per block")
    ax[1].imshow(frame, cmap="nipy_spectral")
    ax[1].imshow(np.ma.masked_where(~model.mask_pix, model.mask_pix), cmap="autumn", alpha=0.35)
    if pos is not None:
        ax[1].plot([pos[1]], [pos[0]], "w+", ms=14, mew=2)
    ax[1].set_title("control mask (red) + located avatar (+)")
    fg = np.where(frame != model.bg, frame, 0)
    ax[2].imshow(fg, cmap="nipy_spectral")
    ax[2].set_title("foreground after background subtraction")
    for a_ in ax:
        a_.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--game", default="breakout")
    p.add_argument("--frames", type=int, default=2_000_000, help="training frames (same as sarsa_linear.py)")
    p.add_argument("--contingency-steps", type=int, default=20000,
                   help="random-policy agent steps used to learn the control mask / background (<=60000)")
    p.add_argument("--thresh", type=float, default=0.3, help="mask threshold, fraction of max action-dependence")
    p.add_argument("--window", type=int, default=3, help="R: avatar-relative window is (2R+1)^2 blocks")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-anneal-frames", type=int, default=500_000)
    p.add_argument("--eval-steps", type=int, default=10000)
    p.add_argument("--no-contingency", action="store_true",
                   help="drop avatar features (plain Sarsa with the same base features; ablation)")
    p.add_argument("--no-bg-sub", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main():
    args = parse()
    assert args.contingency_steps <= 60000, "uint16 background counter limit"
    rng = np.random.default_rng(args.seed)
    env = AtariPalette(args.game)
    tag = "sarsa_samefeat" if args.no_contingency else "contingency"
    run_dir = os.path.join(args.runs_dir, f"{args.game}_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"[1/3] learning control mask from {args.contingency_steps} random steps ...")
    model = ContingencyModel(env.n_actions, args.thresh, not args.no_bg_sub)
    last = collect(env, model, args.contingency_steps, rng, seed=args.seed)
    model.fit()
    print(f"      control mask covers {int(model.mask_blocks.sum())} / {GH * GW} blocks")
    model.save(os.path.join(run_dir, "contingency_model.npz"))

    fx = Features(model, args.window, use_contingency=not args.no_contingency)
    learner = SarsaLambda(env.n_actions, fx.n_features, args.alpha, args.gamma, args.lam)
    # one more frame pair for the diagnostic picture
    a0 = int(rng.integers(env.n_actions))
    nxt, _, _ = env.step(a0)
    save_diagnostic(model, nxt, model.locate(last, nxt, None),
                    os.path.join(args.results_dir, f"contingency_mask_{args.game}.png"),
                    f"{args.game}: learned contingency")

    print(f"[2/3] training Sarsa({args.lam}) with {fx.n_features} features for {args.frames} frames ...")
    train(env, model, fx, learner, args.frames,
          (args.eps_start, args.eps_end, args.eps_anneal_frames), rng,
          log_path=os.path.join(run_dir, "log.csv"), seed=args.seed + 1)
    np.save(os.path.join(run_dir, "weights.npy"), learner.W)

    print(f"[3/3] evaluating eps=0.05 for {args.eval_steps} agent steps ...")
    avg, n_ep = evaluate(env, model, fx, learner, args.eval_steps, 0.05, rng, seed=args.seed + 2)
    print(f"      average score: {avg:.2f} over {n_ep} episode(s)")

    out = {
        "agent": tag, "game": args.game, "avg_score": avg, "episodes": n_ep,
        "eval_steps": args.eval_steps, "eval_epsilon": 0.05, "train_frames": args.frames,
        "contingency_pretrain_steps": args.contingency_steps,
        "mask_blocks": int(model.mask_blocks.sum()), "seed": args.seed,
        "hyperparams": {k: getattr(args, k) for k in
                        ["thresh", "window", "alpha", "gamma", "lam", "eps_start", "eps_end",
                         "eps_anneal_frames", "no_bg_sub"]},
    }
    path = os.path.join(args.results_dir, f"{tag}_{args.game}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
