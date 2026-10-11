"""Fast sanity tests (no training). Run:  python -m tests.test_components"""
import numpy as np
import torch

from dqn.env import AtariEnv
from dqn.model import DQN
from dqn.replay import ReplayMemory


def test_model_shapes():
    out = DQN(6)(torch.zeros(2, 4, 84, 84))
    assert out.shape == (2, 6)


def test_env_preprocessing():
    env = AtariEnv("pong")
    s = env.reset(); assert s.shape == (4, 84, 84) and s.dtype == np.uint8
    s, r, te, tr = env.step(0); assert s.shape == (4, 84, 84)
    assert AtariEnv("spaceinvaders").frame_skip == 3 and env.frame_skip == 4


def test_replay_stacks_and_wraparound():
    N = 50
    m = ReplayMemory(N, seed=1)
    # fake episodes of length 7; frame value encodes (global step id) so we can verify stacking
    t, ep_len = 0, 7
    for t in range(130):                              # wraps the buffer ~2.6 times
        m.push(np.full((84, 84), t % 256, np.uint8), t % 3, 1.0, (t % ep_len) == ep_len - 1, (t % ep_len) == 0)
    for _ in range(300):
        s, a, r, s2, d = m.sample(16)
        for k in range(16):
            v = s[k, -1, 0, 0]                                         # id of current frame
            pos = int(v) % ep_len                                      # position inside its episode
            expected = [int(v) - min(pos, back) if True else 0 for back in (3, 2, 1, 0)]
            assert list(map(int, s[k, :, 0, 0])) == expected, (s[k, :, 0, 0], expected)   # never crosses an episode
            if not d[k]:
                assert int(s2[k, -1, 0, 0]) == (int(v) + 1) % 256       # next state's newest frame = next step
    print("replay OK")


if __name__ == "__main__":
    test_model_shapes(); test_env_preprocessing(); test_replay_stacks_and_wraparound(); print("all tests passed")
