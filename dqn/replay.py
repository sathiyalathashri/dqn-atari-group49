"""Experience replay memory D (Algorithm 1).

To fit in RAM we store each 84x84 frame ONCE and rebuild the 4-frame stacks phi(s) on demand
(1M stacked transitions would need ~28 GB; single frames need ~7 GB). Stacks never cross an
episode boundary: at an episode start the first frame is repeated, exactly as AtariEnv.reset() does.
Stored per step t:  frame x_t (last frame of s_t), a_t, r_t (clipped), terminal flag, first-of-episode flag.
"""
import numpy as np


class ReplayMemory:
    def __init__(self, capacity: int, history: int = 4, seed: int = 0):
        self.N, self.h = capacity, history
        self.frames = np.zeros((capacity, 84, 84), np.uint8)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.dones = np.zeros(capacity, np.bool_)
        self.first = np.zeros(capacity, np.bool_)
        self.ptr, self.size = 0, 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def push(self, frame, action, reward, done, first):
        i = self.ptr
        self.frames[i], self.actions[i], self.rewards[i] = frame, action, reward
        self.dones[i], self.first[i] = done, first
        self.ptr = (i + 1) % self.N
        self.size = min(self.size + 1, self.N)       # old transitions are overwritten when full

    def _stack(self, idx):
        out = np.empty((len(idx), self.h), np.int64)
        out[:, -1] = idx
        for k in range(self.h - 2, -1, -1):
            prev = out[:, k + 1]
            out[:, k] = np.where(self.first[prev], prev, (prev - 1) % self.N)
        return self.frames[out]                       # (B, h, 84, 84)

    def sample(self, batch_size: int):
        """Uniform random minibatch of (phi_j, a_j, r_j, phi_{j+1}, terminal_j)."""
        if self.size < self.N:                        # not yet wrapped: valid i in [0, size-2]
            idx = self.rng.integers(0, self.size - 1, batch_size)
        else:                                         # wrapped: oldest item is at ptr
            off = self.rng.integers(self.h - 1, self.N - 1, batch_size)
            idx = (self.ptr + off) % self.N
        nxt = (idx + 1) % self.N
        return (self._stack(idx), self.actions[idx], self.rewards[idx],
                self._stack(nxt), self.dones[idx])
