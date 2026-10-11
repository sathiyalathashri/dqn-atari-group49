"""Compare Bellman value iteration with model-free Q-learning and replay.

This deterministic gridworld needs only NumPy (no Atari or neural network).
Rewards are +1 at the goal, -1 at the pit, and -0.01 otherwise. Terminal
states have zero continuation value. Finite training approximates Q*;
accuracy depends on exploration, the training budget, and the random seed.

Run ``python tabular_q_demo.py --help`` for training options.
"""
import argparse

import numpy as np

SIZE, GOAL, PIT, GAMMA = 5, (4, 4), (2, 2), 0.9
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
STATES = [(r, c) for r in range(SIZE) for c in range(SIZE)
          if (r, c) not in (GOAL, PIT)]


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def step(s, a):
    """Take an action; boundaries clip movement and terminals are absorbing."""
    if len(s) != 2 or any(not isinstance(x, (int, np.integer)) or not 0 <= x < SIZE for x in s):
        raise ValueError(f"state must contain two integer coordinates in [0, {SIZE})")
    if not isinstance(a, (int, np.integer)) or not 0 <= a < len(MOVES):
        raise ValueError("action must be an integer in [0, 4)")
    if tuple(s) in (GOAL, PIT):
        return tuple(s), 0.0, True
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    s2 = (min(max(r, 0), SIZE - 1), min(max(c, 0), SIZE - 1))
    if s2 == GOAL:
        return s2, 1.0, True
    if s2 == PIT:
        return s2, -1.0, True
    return s2, -0.01, False


def value_iteration(iters=200, *, tol=1e-10):
    """Apply synchronous Bellman backups until tolerance or the iteration cap."""
    _positive_integer("iters", iters)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    q = np.zeros((SIZE, SIZE, len(MOVES)))
    for _ in range(iters):
        updated = np.zeros_like(q)
        for s in STATES:
            for a in range(len(MOVES)):
                s2, reward, done = step(s, a)
                updated[s][a] = reward + (0.0 if done else GAMMA * q[s2].max())
        delta = np.max(np.abs(updated - q))
        q = updated
        if delta <= tol:
            break
    return q


def q_learning_with_replay(episodes=3000, eps=0.2, lr=0.1, batch=32, seed=0,
                           *, replay_capacity=10000, max_steps=100,
                           random_starts=True):
    """Learn with bounded replay and uniformly sampled minibatches.

    Random nonterminal starts improve coverage when comparing the whole table.
    Set random_starts=False to train only from (0, 0). Greedy action ties are
    broken randomly using the seeded generator. Updates are sequential so
    repeated state/action pairs in a minibatch each receive their own update.
    The step cap truncates an episode, but does not make its last state terminal.
    """
    for name, value in (("episodes", episodes), ("batch", batch),
                        ("replay_capacity", replay_capacity), ("max_steps", max_steps)):
        _positive_integer(name, value)
    if not np.isfinite(eps) or not 0 <= eps <= 1:
        raise ValueError("eps must be finite and in [0, 1]")
    if not np.isfinite(lr) or not 0 < lr <= 1:
        raise ValueError("lr must be finite and in (0, 1]")
    rng = np.random.default_rng(seed)
    q = np.zeros((SIZE, SIZE, len(MOVES)))
    memory = []
    position = 0
    for _ in range(episodes):
        s = STATES[int(rng.integers(len(STATES)))] if random_starts else (0, 0)
        for _ in range(max_steps):
            if rng.random() < eps:
                a = int(rng.integers(len(MOVES)))
            else:
                a = int(rng.choice(np.flatnonzero(q[s] == q[s].max())))
            s2, reward, done = step(s, a)
            transition = (s, a, reward, s2, done)
            if len(memory) < replay_capacity:
                memory.append(transition)
            else:
                memory[position] = transition
            position = (position + 1) % replay_capacity
            # Use the available transitions during warm-up, without replacement.
            indices = rng.choice(len(memory), size=min(batch, len(memory)), replace=False)
            for j in indices:
                sj, aj, rj, s2j, terminal = memory[j]
                target = rj + (0.0 if terminal else GAMMA * q[s2j].max())
                q[sj][aj] += lr * (target - q[sj][aj])
            s = s2
            if done:
                break
    return q


def optimal_action_count(reference, learned, *, atol=1e-8):
    """Count greedy actions that are optimal, allowing ties in the reference."""
    return sum(bool(np.isclose(reference[s][learned[s].argmax()],
                               reference[s].max(), rtol=0.0, atol=atol))
               for s in STATES)


def greedy_rollout(q, max_steps=100):
    """Evaluate deterministically from (0, 0); report loops instead of hanging."""
    _positive_integer("max_steps", max_steps)
    s, total, path = (0, 0), 0.0, [(0, 0)]
    for _ in range(max_steps):
        s, reward, done = step(s, int(q[s].argmax()))
        total += reward
        repeated = s in path
        path.append(s)
        if done:
            return path, total, "goal" if s == GOAL else "pit"
        if repeated:
            return path, total, "loop"
    return path, total, "step limit"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-capacity", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--fixed-start", action="store_true",
                        help="train from (0, 0) instead of random nonterminal states")
    args = parser.parse_args(argv)
    try:
        reference = value_iteration()
        learned = q_learning_with_replay(
            episodes=args.episodes, eps=args.eps, lr=args.lr, batch=args.batch,
            seed=args.seed, replay_capacity=args.replay_capacity,
            max_steps=args.max_steps, random_starts=not args.fixed_start)
    except ValueError as exc:
        parser.error(str(exc))
    error = np.array([np.abs(reference[s] - learned[s]) for s in STATES])
    print(f"Training: episodes={args.episodes}, seed={args.seed}, "
          f"starts={'fixed' if args.fixed_start else 'random'}")
    print(f"Nonterminal Q error: max={error.max():.6f}, mean={error.mean():.6f}")
    print("Greedy policy (G=goal, X=pit):")
    arrows = "^v<>"
    for r in range(SIZE):
        print(" ".join("G" if (r, c) == GOAL else "X" if (r, c) == PIT
                       else arrows[learned[r, c].argmax()] for c in range(SIZE)))
    print(f"Optimal greedy action in {optimal_action_count(reference, learned)}/{len(STATES)} states (ties allowed)")
    path, total, outcome = greedy_rollout(learned, args.max_steps)
    print(f"Greedy rollout: {outcome}, {len(path) - 1} steps, return={total:.2f}")
    print("Path: " + " -> ".join(map(str, path)))


if __name__ == "__main__":
    main()
