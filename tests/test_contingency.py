"""
Sanity tests for contingency_sarsa.py that need NO Atari ROMs.

Run from the repo root:   python -m tests.test_contingency

A tiny synthetic "catch" game is used: a paddle (the only thing the agent controls)
moves left/right at the bottom, a ball falls from a random x. +1 reward if the paddle
catches it. A flickering HUD block changes every step regardless of the action, so the
test also checks that action-independent motion is NOT put in the control mask.
"""
import numpy as np

from contingency_sarsa import (
    GH, GW, BH, BW, ContingencyModel, Features, SarsaLambda,
    collect, train, evaluate, H, W,
)

PADDLE_C, BALL_C, WALL_C, HUD_C = 40, 90, 5, 77


class FakeCatch:
    """Same interface as AtariPalette: reset(seed) / step(a) / n_actions / k."""
    n_actions, k = 3, 4              # 0 noop, 1 left, 2 right

    def __init__(self, balls_per_episode=25, seed=0):
        self.rng = np.random.default_rng(seed)
        self.balls_per_episode = balls_per_episode

    def _new_ball(self):
        self.bx, self.by = int(self.rng.integers(4, W - 8)), 20

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.px, self.balls, self.hud = W // 2 - 8, 0, 0
        self._new_ball()
        return self._draw()

    def _draw(self):
        s = np.zeros((H, W), np.uint8)
        s[:, :2] = WALL_C
        s[:, -2:] = WALL_C
        s[2:8, 10 + 4 * (self.hud % 10):14 + 4 * (self.hud % 10)] = HUD_C      # flicker, action-independent
        s[190:194, self.px:self.px + 16] = PADDLE_C
        s[self.by:self.by + 4, self.bx:self.bx + 4] = BALL_C
        return s

    def step(self, a):
        reward, done = 0.0, False
        for _ in range(4):
            self.px = int(np.clip(self.px + {0: 0, 1: -3, 2: 3}[a], 2, W - 18))
            self.by += 4
            if self.by >= 186:
                if self.px - 4 <= self.bx <= self.px + 16:
                    reward += 1.0
                self.balls += 1
                if self.balls >= self.balls_per_episode:
                    done = True
                    break
                self._new_ball()
        self.hud = int(self.rng.integers(0, 10))
        return self._draw(), reward, done


def test_control_mask_and_localisation():
    env = FakeCatch(seed=1)
    rng = np.random.default_rng(1)
    model = ContingencyModel(env.n_actions, thresh=0.3)
    collect(env, model, 4000, rng, seed=1)
    model.fit()
    rows = np.nonzero(model.mask_blocks.any(axis=1))[0]
    assert len(rows) >= 1 and set(rows) <= {190 // BH}, f"mask rows {rows}"
    assert not model.mask_blocks[:1].any(), "HUD flicker must not be in the control mask"
    # localisation: the estimated x should follow the real paddle centre
    obs, pos, errs = env.reset(), None, []
    for _ in range(300):
        a = int(rng.integers(3))
        nxt, _, done = env.step(a)
        pos = model.locate(obs, nxt, pos)
        if pos is not None:
            errs.append(abs(pos[1] - (env.px + 8)))
        obs = env.reset() if done else nxt
        if done:
            pos = None
    assert np.median(errs) < 12, f"median x error {np.median(errs):.1f}px"
    print("ok  control mask rows:", rows.tolist(), " median avatar x error: %.1fpx" % np.median(errs))


def test_feature_indices():
    env = FakeCatch(seed=2)
    rng = np.random.default_rng(2)
    model = ContingencyModel(env.n_actions)
    collect(env, model, 1500, rng)
    model.fit()
    fx = Features(model, window=3, use_contingency=True)
    f = fx(env.reset(), (192.0, 80.0))
    assert f.min() >= 0 and f.max() < fx.n_features
    assert len(np.unique(f)) == len(f), "feature indices must be unique per step"
    f0 = fx(env.reset(), None)
    assert f0.max() < fx.n_features
    print("ok  features: %d total, %d active (with avatar), %d active (no avatar yet)"
          % (fx.n_features, len(f), len(f0)))


def run(use_c, steps_frames, seed=3):
    env = FakeCatch(seed=seed)
    rng = np.random.default_rng(seed)
    model = ContingencyModel(env.n_actions)
    collect(env, model, 3000, rng)
    model.fit()
    fx = Features(model, window=3, use_contingency=use_c)
    learner = SarsaLambda(env.n_actions, fx.n_features, alpha=0.2, gamma=0.95, lam=0.8)
    train(env, model, fx, learner, steps_frames, (1.0, 0.05, steps_frames // 3), rng,
          log_every=10**9, seed=seed)
    return evaluate(env, model, fx, learner, 3000, 0.05, rng, seed=seed + 10)[0]


def test_learning():
    env = FakeCatch(seed=5)
    rng = np.random.default_rng(5)
    scores = []
    obs = env.reset()
    for _ in range(3000):
        obs, r, done = env.step(int(rng.integers(3)))
        scores.append(r)
        if done:
            env.reset()
    random_score = float(np.sum(scores)) / max(1, len(scores) / (25 * 11)) if scores else 0.0
    with_c = run(True, 400_000)
    without_c = run(False, 400_000)
    print("ok  episode score  random~%.1f  sarsa(base features)=%.1f  contingency=%.1f"
          % (random_score, without_c, with_c))
    assert with_c > random_score * 1.3, "contingency agent should clearly beat random"


if __name__ == "__main__":
    test_control_mask_and_localisation()
    test_feature_indices()
    test_learning()
    print("all contingency tests passed")
