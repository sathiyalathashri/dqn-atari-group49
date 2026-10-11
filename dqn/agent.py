"""Deep Q-learning agent (Algorithm 1): epsilon-greedy acting + Q-learning minibatch update."""
import numpy as np
import torch
import torch.nn.functional as F

from .model import DQN


class DQNAgent:
    def __init__(self, n_actions, device="cpu", lr=2.5e-4, gamma=0.99, target_update=0,
                 huber=False, grad_clip=None, seed=0):
        self.n_actions, self.gamma, self.device = n_actions, gamma, torch.device(device)
        self.q = DQN(n_actions).to(self.device)
        # Paper-faithful default: NO target network (targets use the current weights).
        # target_update>0 enables the later Nature-2015 trick as an optional extra.
        self.target = None
        if target_update > 0:
            self.target = DQN(n_actions).to(self.device)
            self.target.load_state_dict(self.q.state_dict())
        self.target_update = target_update
        self.opt = torch.optim.RMSprop(self.q.parameters(), lr=lr, alpha=0.95, eps=0.01)  # paper: RMSProp
        self.huber, self.grad_clip, self.updates = huber, grad_clip, 0
        self.rng = np.random.default_rng(seed)

    def _t(self, x):  # uint8 numpy -> float tensor in [0,1]
        return torch.as_tensor(x, device=self.device).float().div_(255.0)

    @torch.no_grad()
    def q_values(self, states):
        return self.q(self._t(states))

   #Random action with probability ε, otherwise the argmax of Q. One forward pass scores every action.
    def act(self, state, epsilon: float) -> int:
        """epsilon-greedy behaviour policy."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(self.q_values(state[None]).argmax(1).item())

    def learn(self, batch) -> float:
        s, a, r, s2, done = batch
        s, s2 = self._t(s), self._t(s2)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        done = torch.as_tensor(done, device=self.device).float()
        with torch.no_grad():                                   # y_j  (Algorithm 1)
            net = self.target if self.target is not None else self.q
            y = r + self.gamma * (1.0 - done) * net(s2).max(1).values
        q_sa = self.q(s).gather(1, a[:, None]).squeeze(1)       # Q(phi_j, a_j; theta)
        loss = F.smooth_l1_loss(q_sa, y) if self.huber else F.mse_loss(q_sa, y)   # (y - Q)^2
        self.opt.zero_grad(set_to_none=True)
        loss.backward()                                         # gradient of Eq. 2/3
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()
        self.updates += 1
        if self.target is not None and self.updates % self.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())
        return loss.item()

    def save(self, path, game):
        torch.save({"model": self.q.state_dict(), "n_actions": int(self.n_actions), "game": game}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ck["model"])
