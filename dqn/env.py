"""Atari environment + the preprocessing function phi from Section 4.1 of the paper.

* raw frame 210x160 RGB  ->  grayscale  ->  resize to 110x84  ->  crop 84x84 playing area
* frame skipping: agent picks an action every k-th frame, action repeated in between (k=4,
  k=3 for Space Invaders so the blinking lasers stay visible)
* phi stacks the last 4 preprocessed frames  ->  state of shape (4, 84, 84), uint8
"""
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
import ale_py

gym.register_envs(ale_py)

GAMES = {  # the seven games used in the paper
    "beamrider": "BeamRider", "breakout": "Breakout", "enduro": "Enduro", "pong": "Pong",
    "qbert": "Qbert", "seaquest": "Seaquest", "spaceinvaders": "SpaceInvaders",
}


def preprocess(frame_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_AREA)  # (height=110, width=84)
    return small[18:102, :]                                            # crop to 84x84


class AtariEnv:
    def __init__(self, game: str, seed: int = 0, frame_skip: int | None = None,
                 history: int = 4, max_episode_frames: int = 108_000):
        game = game.lower()
        if game not in GAMES:
            raise ValueError(f"game must be one of {list(GAMES)}")
        self.game = game
        self.frame_skip = frame_skip or (3 if game == "spaceinvaders" else 4)
        self.history = history
        # frameskip=1 + no sticky actions -> we do the skipping ourselves, like the paper (ALE 2013)
        self.env = gym.make(f"ALE/{GAMES[game]}-v5", frameskip=1, repeat_action_probability=0.0,
                            full_action_space=False, max_num_frames_per_episode=max_episode_frames)
        self.n_actions = int(self.env.action_space.n)         # 4..18 depending on the game
        self._seed = seed
        self._first_reset = True
        self.frames = deque(maxlen=history)
        self.last_raw = None                              # last raw RGB frame (for screenshots)

    def _state(self):
        return np.stack(self.frames, axis=0)

    def reset(self):
        if self._first_reset:
            obs, _ = self.env.reset(seed=self._seed); self._first_reset = False
        else:
            obs, _ = self.env.reset()
        self.last_raw = obs
        f = preprocess(obs)
        for _ in range(self.history):
            self.frames.append(f)
        return self._state()

    def step(self, action: int):
        total, term, trunc = 0.0, False, False
        for _ in range(self.frame_skip):
            obs, r, term, trunc, _ = self.env.step(action)
            total += r
            if term or trunc:
                break
        self.last_raw = obs
        self.frames.append(preprocess(obs))
        return self._state(), total, term, trunc

    def close(self):
        self.env.close()
