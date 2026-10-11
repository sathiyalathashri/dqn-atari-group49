"""Section 2 of the paper on a toy problem (no Atari, runs in seconds).

 1. Value iteration with the Bellman equation (Eq. 1):  Q_{i+1}(s,a) = E[r + gamma * max_a' Q_i(s',a')]
    converges to Q*.
 2. Model-free, off-policy Q-learning with an eps-greedy behaviour policy and experience replay
    (the same recipe as DQN, but with a table instead of a neural net) learns the same Q*.
"""
import numpy as np

SIZE, GOAL, PIT, GAMMA = 5, (4, 4), (2, 2), 0.9
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


def step(s, a):
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    s2 = (min(max(r, 0), SIZE - 1), min(max(c, 0), SIZE - 1))
    if s2 == GOAL: return s2, 1.0, True
    if s2 == PIT:  return s2, -1.0, True
    return s2, -0.01, False


def value_iteration(iters=200):
    Q = np.zeros((SIZE, SIZE, 4))
    for _ in range(iters):
        new = np.zeros_like(Q)
        for r in range(SIZE):
            for c in range(SIZE):
                if (r, c) in (GOAL, PIT): continue
                for a in range(4):
                    s2, rew, done = step((r, c), a)
                    new[r, c, a] = rew + (0 if done else GAMMA * Q[s2].max())
        Q = new
    return Q


def q_learning_with_replay(episodes=3000, eps=0.2, lr=0.1, batch=32, seed=0):
    rng, Q, memory = np.random.default_rng(seed), np.zeros((SIZE, SIZE, 4)), []
    for _ in range(episodes):
        s = (0, 0)
        for _ in range(100):
            a = rng.integers(4) if rng.random() < eps else int(Q[s].argmax())   # behaviour policy
            s2, r, done = step(s, a)
            memory.append((s, a, r, s2, done))                                    # store in replay memory D
            for j in rng.integers(len(memory), size=batch if len(memory) > batch else 1):  # random minibatch
                sj, aj, rj, s2j, dj = memory[j]
                y = rj + (0 if dj else GAMMA * Q[s2j].max())                      # target y_j
                Q[sj][aj] += lr * (y - Q[sj][aj])                                 # gradient step on (y - Q)^2
            s = s2
            if done: break
    return Q


if __name__ == "__main__":
    Qstar, Qhat = value_iteration(), q_learning_with_replay()
    mask = np.ones((SIZE, SIZE), bool); mask[GOAL] = mask[PIT] = False
    err = np.abs(Qstar - Qhat)[mask].max()
    print(f"max |Q* - Q_learned| over non-terminal states: {err:.4f}")
    arrows = np.array(list("^v<>"))
    print("greedy policy from Q-learning (G=goal, X=pit):")
    for r in range(SIZE):
        print(" ".join("G" if (r, c) == GOAL else "X" if (r, c) == PIT else arrows[Qhat[r, c].argmax()] for c in range(SIZE)))
    print("same greedy action as Q* in", int((Qstar.argmax(2) == Qhat.argmax(2))[mask].sum()), "of", int(mask.sum()), "states")
