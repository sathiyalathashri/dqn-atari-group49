# Playing Atari with Deep Reinforcement Learning — runnable implementation

Re-implementation (PyTorch) of Mnih et al. 2013 (arXiv:1312.5602): the original DQN.

```
dqn_atari/
├── dqn/
│   ├── env.py       # Atari env + preprocessing phi (gray, 110x84, crop 84x84, 4-frame stack, frame-skip, k=3 for Space Invaders)
│   ├── model.py     # Q-network: conv16 8x8/4 -> conv32 4x4/2 -> FC256 -> one linear output per action
│   ├── replay.py    # experience replay memory D (stores each frame once, rebuilds stacks on sampling)
│   ├── agent.py     # eps-greedy policy + Q-learning minibatch update with RMSProp  (Algorithm 1)
│   └── utils.py     # eps schedule, paper-style evaluation, held-out states, logging, plots
├── train.py             # Algorithm 1 end-to-end; logs avg reward + avg max-Q per epoch; saves checkpoints + curves
├── evaluate.py          # eps=0.05 evaluation (Sec 5.3), random baseline, Q-value visualization (Fig. 3 analogue)
├── sarsa_linear.py      # simplified linear Sarsa baseline with hand-made colour/block features (Table 1 "Sarsa" analogue)
├── tabular_q_demo.py    # Sec. 2 on a toy gridworld: Bellman value iteration vs. Q-learning + replay (seconds)
├── compare.py           # Table-1 style table: your results next to the paper's numbers
└── tests/test_components.py
```

## Setup
```bash
pip install -r requirements.txt
python -m tests.test_components      # shapes, preprocessing, replay-stack correctness
python tabular_q_demo.py             # understand the maths first (no Atari)
```

## Run the experiments
```bash
# 1. baselines
python evaluate.py --game breakout --agent random
python sarsa_linear.py --game breakout --frames 2000000

# 2. DQN, paper settings (10M frames, replay 1M frames, batch 32, eps 1->0.1 over 1M frames)
python train.py --game breakout                     # GPU strongly recommended
#    quick smoke run (minutes, will NOT learn much):
python train.py --game pong --total-frames 20000 --replay-size 5000 --learn-start 1000 \
       --epoch-steps 1000 --eval-steps 500 --eps-anneal-frames 10000

# 3. evaluate the trained network (eps = 0.05)
python evaluate.py --game breakout --agent dqn --checkpoint runs/breakout/best.pt --viz-q

# 4. compare with the paper
python compare.py
```
Outputs: `runs/<game>/{log.csv,training_curves.png,best.pt,last.pt,config.json}`, `results/*.json`,
`results/q_value_trace.png` plus screenshots at the lowest / highest predicted value.

## How to check that learning works
* `training_curves.png` (Fig. 2 analogue): average reward is noisy; **average max-Q on held-out states should rise smoothly**
  and the loss should stay finite (no divergence).
* Breakout/Pong typically show clear improvement over random after ~1-3M frames; paper-level numbers need the full 10M frames
  (and still vary by seed). Expect lower scores than Table 1 for short runs.
* `evaluate.py --viz-q`: Q should spike when a rewarding event is imminent and fall afterwards (Fig. 3).

## Faithful to the paper / deliberate choices
* Faithful: preprocessing, architecture, RMSProp, batch 32, eps-greedy 1->0.1 over 1M frames then 0.1, reward clipping to {-1,0,1}
  (training only, evaluation uses raw score), frame-skip 4 (3 for Space Invaders), no target network, squared error loss,
  evaluation with eps=0.05, same hyper-parameters for every game.
* Not specified in the paper (my choices): lr 2.5e-4, gamma 0.99, RMSProp decay 0.95 / eps 0.01, 10k warm-up transitions,
  replay = 250k transitions (= 1M frames at skip 4), "epoch" = 50k agent steps.
* Optional extras OFF by default: `--target-update N`, `--huber`, `--grad-clip X` (later DQN improvements, useful if training is unstable).
* Uses the modern `ALE/<Game>-v5` envs with sticky actions disabled and the minimal action set, so scores are comparable in spirit,
  not identical to the 2013 emulator setup. `sarsa_linear.py` is a simplified baseline, not the exact feature set of Bellemare et al.
