"""Print a Table-1 style comparison from results/*.json (+ the paper's published numbers for reference)."""
import glob
import json

PAPER = {  # Table 1 of the paper
    "Random": dict(beamrider=354, breakout=1.2, enduro=0, pong=-20.4, qbert=157, seaquest=110, spaceinvaders=179),
    "Sarsa [paper]": dict(beamrider=996, breakout=5.2, enduro=129, pong=-19, qbert=614, seaquest=665, spaceinvaders=271),
    "Contingency [paper]": dict(beamrider=1743, breakout=6, enduro=159, pong=-17, qbert=960, seaquest=723, spaceinvaders=268),
    "DQN [paper]": dict(beamrider=4092, breakout=168, enduro=470, pong=20, qbert=1952, seaquest=1705, spaceinvaders=581),
    "Human [paper]": dict(beamrider=7456, breakout=31, enduro=368, pong=-3, qbert=18900, seaquest=28010, spaceinvaders=3690),
}
G = ["beamrider", "breakout", "enduro", "pong", "qbert", "seaquest", "spaceinvaders"]
mine = {}
for f in glob.glob("results/*.json"):
    d = json.load(open(f)); mine.setdefault(f"{d['agent']} [yours]", {})[d["game"]] = d["mean"]
print(f"{'':22s}" + "".join(f"{g[:9]:>10s}" for g in G))
for name, row in {**mine, **PAPER}.items():
    print(f"{name:22s}" + "".join(f"{row[g]:>10.1f}" if g in row else f"{'-':>10s}" for g in G))
