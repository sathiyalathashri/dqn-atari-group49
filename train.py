"""Train a DQN (Algorithm 1: deep Q-learning with experience replay) on one Atari game.

Paper-scale run  :  python train.py --game breakout            (10M frames, ~days on CPU, hours on GPU)
Quick smoke test :  python train.py --game pong --total-frames 20000 --replay-size 5000 \
                        --learn-start 1000 --epoch-steps 1000 --eval-steps 500 --eps-anneal-frames 10000
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from dqn.agent import DQNAgent
from dqn.env import AtariEnv, GAMES
from dqn.replay import ReplayMemory
from dqn.utils import CSVLogger, collect_heldout_states, linear_epsilon, plot_training, run_eval, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="pong", choices=list(GAMES))
    p.add_argument("--total-frames", type=int, default=10_000_000)        # paper: 10M frames
    p.add_argument("--replay-size", type=int, default=250_000)            # = 1M frames at frame-skip 4
    p.add_argument("--batch-size", type=int, default=32)                  # paper: 32
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.1)
    p.add_argument("--eps-anneal-frames", type=int, default=1_000_000)    # paper: first 1M frames
    p.add_argument("--learn-start", type=int, default=10_000)             # transitions before first update
    p.add_argument("--update-every", type=int, default=1)
    p.add_argument("--epoch-steps", type=int, default=50_000)             # agent steps per "epoch"
    p.add_argument("--eval-steps", type=int, default=10_000)              # paper: 10000 steps, eps=0.05
    p.add_argument("--eval-eps", type=float, default=0.05)
    p.add_argument("--heldout", type=int, default=500)
    # optional deviations from the 2013 paper (all OFF by default)
    p.add_argument("--target-update", type=int, default=0)
    p.add_argument("--huber", action="store_true")
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs")
    a = p.parse_args()

    set_seed(a.seed)
    out = os.path.join(a.out, a.game); os.makedirs(out, exist_ok=True)
    env, eval_env = AtariEnv(a.game, seed=a.seed), AtariEnv(a.game, seed=a.seed + 1000)
    agent = DQNAgent(env.n_actions, a.device, a.lr, a.gamma, a.target_update, a.huber, a.grad_clip, a.seed)
    replay = ReplayMemory(a.replay_size, seed=a.seed)
    heldout = collect_heldout_states(eval_env, a.heldout, a.seed)
    log = CSVLogger(os.path.join(out, "log.csv"), ["epoch", "frames", "avg_reward", "avg_max_q", "loss", "epsilon", "fps"])
    json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"{a.game}: {env.n_actions} actions, frame-skip {env.frame_skip}, device {a.device}")

    frames = steps = epoch = 0
    state, first, ep_ret, losses, t0 = env.reset(), True, 0.0, [], time.time()
    best = -1e9
    while frames < a.total_frames:
        eps = linear_epsilon(frames, a.eps_start, a.eps_end, a.eps_anneal_frames)
        act = agent.act(state, eps)                                    # eps-greedy action
        nstate, r, term, trunc = env.step(act)                         # execute in emulator
        frames += env.frame_skip; steps += 1; ep_ret += r
        done = term or trunc
        replay.push(state[-1], act, float(np.sign(r)), done, first)    # store transition; reward clipped to {-1,0,1}
        first = False
        if done:
            state, first, ep_ret = env.reset(), True, 0.0
        else:
            state = nstate
        if len(replay) >= max(a.learn_start, a.batch_size + 5) and steps % a.update_every == 0:
            losses.append(agent.learn(replay.sample(a.batch_size)))    # minibatch gradient step

        if steps % a.epoch_steps == 0:                                  # ---- end of epoch: evaluate ----
            epoch += 1
            avg_r, _ = run_eval(eval_env, lambda s: agent.act(s, a.eval_eps), a.eval_steps)
            avg_q = float(agent.q_values(heldout).max(1).values.mean())
            fps = frames / (time.time() - t0)
            ml = float(np.mean(losses)) if losses else float("nan"); losses = []
            log.log(epoch=epoch, frames=frames, avg_reward=avg_r, avg_max_q=avg_q, loss=ml, epsilon=eps, fps=fps)
            print(f"epoch {epoch:3d} | frames {frames:>9d} | eps {eps:.2f} | avg reward {avg_r:8.2f} | "
                  f"avg max-Q {avg_q:7.3f} | loss {ml:.4f} | {fps:.0f} fps", flush=True)
            agent.save(os.path.join(out, "last.pt"), a.game)
            if avg_r > best:
                best = avg_r; agent.save(os.path.join(out, "best.pt"), a.game)
            plot_training(log.path, os.path.join(out, "training_curves.png"), a.game)

    agent.save(os.path.join(out, "last.pt"), a.game)
    print("done. checkpoints + log + curves in", out)


if __name__ == "__main__":
    main()
