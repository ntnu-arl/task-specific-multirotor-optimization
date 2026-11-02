"""Compare designs fairly by the best they can do: a design is trained --seeds times, each time by sequential halving
(rl.halving()), which picks a policy (on rl.EVAL_COPIES episodes, the same for every policy); that policy flies
rl.FINAL_EPISODES other episodes (its "final" fitness, as the paper's 40000). Prints per design the final fitness of
its --seeds halvings (mean, spread, best) and how much the episodes the policy was picked on flattered it, and writes
<out>/compare.json (rl.multistart()).

    python compare.py --out results/compare --seeds 8 --seed 1 \
        --designs planar franchi rajappa shen paper results/<run>/final_design.json

A design is a baseline name (design.BASELINE_NAMES), "paper" (the paper's optimized design for the task, TASK=A or B) or a JSON file
with {"x": [...]} (or a results file with a "design" entry).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import design
import rl

def load(arg):
    if arg in design.BASELINE_NAMES:
        return arg, {"name": arg}
    if arg == "paper":
        return f"paper task {design.TASK}", {"x": design.PAPER_DESIGN}
    d = json.loads(Path(arg).read_text())
    return Path(arg).parent.name + "/" + Path(arg).stem, {"x": d.get("design", d)["x"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--designs", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, required=True, help="independent halvings per design")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override constants of rl.py, e.g. LR_DECAY=False")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=", 1)
        assert hasattr(rl, k), k
        setattr(rl, k, eval(v))
        print(f"rl.{k} = {getattr(rl, k)}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    named = [load(x) for x in a.designs]
    phys = [design.physical(d) for _, d in named]
    assert all(p["feasible"] for p in phys), [n for (n, _), p in zip(named, phys) if not p["feasible"]]
    S = a.seeds
    tic = time.time()
    res = rl.multistart(phys, S, seed=a.seed, log=print)
    wall = time.time() - tic
    print(f"\n{S} halvings ({rl.HALVING}, {rl.ITERS} PPO iterations) per design in {wall / 60:.1f} min; "
          f"final: the policy a halving picked on {rl.FINAL_EPISODES} episodes")
    print(f"{'design':38s} {'mean':>6s} {'std':>5s} {'min':>6s} {'max':>6s} {'flatter':>7s}   final of each halving")
    rows = []
    for (n, d), ph, r in sorted(zip(named, phys, res), key=lambda z: -np.mean(z[2]["final"])):
        f = np.array(r["final"])
        print(f"{n:38s} {f.mean():6.2f} {f.std():5.2f} {f.min():6.2f} {f.max():6.2f} "
              f"{(np.array(r['picked_on']) - f).mean():+7.2f}   " + " ".join(f"{v:5.1f}" for v in f))
        rows.append({"name": n, "design": d, **{k: v for k, v in r.items() if k != "policies"},
                     "best_policy": r["policies"][int(f.argmax())], "mass": ph["mass"],
                     "thrust_to_weight": ph["thrust_to_weight"], "hover_tilt_deg": ph["hover_tilt_deg"]})
    (out / "compare.json").write_text(json.dumps({"args": vars(a), "wall_s": wall, "designs": rows}))


if __name__ == "__main__":
    main()
