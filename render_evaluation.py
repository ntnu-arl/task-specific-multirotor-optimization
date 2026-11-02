"""Evaluate one hexarotor design against the baselines of the paper and render the comparison video.

    python render_evaluation.py --design results/<run>/final_design.json --out results/<run> --prefix final_
    python render_evaluation.py --design baseline --out results/renders/baseline

--design: a JSON file with {"x": [15 numbers in [0, 1]]}, or a results JSON (best_design.json, final_design.json:
with the policy it was evaluated with), or a baseline name (design.BASELINE_NAMES; "baseline" is design.BASELINE).
A design without a policy first gets one (the same training as in the optimization: rl.halving()). The baselines
(design.BASELINE_NAMES, all but the design itself) fly with the policies in the JSON's "references" ({name:
policy}, as in final_design.json), or else with their cached ones (rl.baseline_policies(), trained the first time).
All fly the same episodes; the video shows them one on top of the other. --reevaluations (a run's
reevaluations.json): the report starts with a bar plot of its last re-evaluation (the design, the paper's design and
the baselines).
Writes <out>/<prefix>comparison.mp4 (no numbers, a name on every drone), <out>/<prefix>report.pdf (the design and
every baseline: pictures and all the data), <out>/<prefix>metrics.json and the pictures in <out>/images/. Called by
evolve.py at the end of a run (--out <run> --prefix final_).
"""
import argparse
import json
from pathlib import Path

import design
import render
import report
import rl

RECORD = 64  # episodes recorded for the video (the first ones of the evaluation, the same for all)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="", help="of the output files, e.g. final_")
    ap.add_argument("--label", default="Evaluated design", help="title of the design page in the report")
    ap.add_argument("--name", default="evaluated", help="name of the design in the video")
    ap.add_argument("--reevaluations", help="a run's reevaluations.json: a bar plot of its last re-evaluation")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    if a.design == "baseline" or a.design in design.BASELINE_NAMES:
        d = design.BASELINE if a.design == "baseline" else {"name": a.design}
        rec, name, label = {}, d["name"], f"Baseline ({d['name']})"
    else:
        rec = json.loads(Path(a.design).read_text())
        d, name, label = rec.get("design", rec), a.name, a.label
    policy = rec.get("policy")
    ph = design.physical(d)
    if not ph["feasible"]:
        raise SystemExit(f"this design cannot be flown: {ph['reason']}")
    if policy is None and "name" in d:
        policy = rl.baseline_policies([d["name"]])[d["name"]][0]
    elif policy is None:
        print(f"training the design's controller (sequential halving {rl.HALVING}, {rl.ITERS} PPO iterations)",
              flush=True)
        policy = rl.halving([ph], seed=0, log=print)[0]["policy"]
    refs = [n for n in design.BASELINE_NAMES if n != d.get("name") and design.physical({"name": n})["feasible"]]
    given = rec.get("references", {})
    cached = rl.baseline_policies([n for n in refs if n not in given])
    names = [name] + refs
    phys = [ph] + [design.physical({"name": n}) for n in refs]
    policies = [policy] + [given[n] if n in given else cached[n][0] for n in refs]
    res, recording = rl.evaluate(phys, policies, record=RECORD)
    results = [{"design": dd, **r} for dd, r in zip([d] + [{"name": n} for n in refs], res)]
    (out / f"{a.prefix}metrics.json").write_text(json.dumps(dict(zip(names, results)), indent=1))
    pages = []
    for k, (n, ph_k, r) in enumerate(zip(names, phys, results)):
        print(f"{n:10s} fitness {r['fitness']:+6.2f}  gates {r['gates_per_episode']:5.2f}  missed "
              f"{r['missed_per_episode']:.3f}  crashes {100 * r['crash_rate']:.1f} %")
        image = out / "images" / f"{a.prefix}{'design' if k == 0 else n}.png"
        highlight = k == 0 and "name" not in d
        render.still(ph_k, image, highlight, level=highlight)
        pages.append((label if k == 0 else f"Baseline ({n})", highlight, image, r))
    comparison = None
    if a.reevaluations:  # the design (if it was re-evaluated), the paper's design and the baselines
        rows = json.loads(Path(a.reevaluations).read_text())[-1][1]
        comparison = [(label if r["design"] == d else r["name"], r["design"] == d, r) for r in rows
                      if r["design"] == d or not r["name"].startswith("#")]
    report.evaluation_pdf(out / f"{a.prefix}report.pdf", pages, comparison)
    video = out / f"{a.prefix}comparison.mp4"
    render.render(phys, names, recording, recording["crashed"], video, highlight=None if "name" in d else 0)
    print(f"video -> {video}, report -> {out / f'{a.prefix}report.pdf'}")


if __name__ == "__main__":
    main()
