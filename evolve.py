"""Design optimization of the hexarotor for task B (or task A, with the environment variable TASK=A), as in the paper: Bayesian optimization (Ax/BoTorch) for global
search, then CMA-ES for local refinement around the best design found by BO.

Every design's controller is trained by sequential halving (rl.halving(), as the paper: 16 trainings, the best 8 go
on after 150 PPO iterations and the best 4 after 300, to 900) and its fitness is its best survivor; the designs of a
batch (a BO batch or a CMA-ES generation) are trained and evaluated together on the GPU. Training is still noisy: a
design with lucky trainings can look better than it is, and the optimizer would follow the luck. So the 5 best
designs are re-evaluated, with what counts being the best a design can do: rl.REEVAL_MULTISTARTS (8) independent
halvings, each one's picked policy flying rl.FINAL_EPISODES (40960) other episodes as the paper's 40000; the design's
fitness is the mean of these 8 (rl.multistart()). This
happens after the BO (CMA-ES starts from the best of them) and at the end (with the paper's design and its baselines,
design.BASELINE_NAMES, for reference; the best of the 5 is the final design). A design that cannot hover is not trained and gets the
fitness of an immediate crash (sim: -10). At the end, render_evaluation.py renders the final design against the
baselines (each with the policy of its re-evaluation) into the run folder.

    python evolve.py --run-dir results/<name> --bo-evals 400 --cma-evals 400 --batch 20 --seed 242689

Outputs in --run-dir: log (stdout), progress.csv (one line per evaluation), evaluations.jsonl, progress.pdf (updated
after every batch: progress against time, baseline and best design with their data, the re-evaluations),
baseline.json (the evaluation of the baseline of the reports),
best_design.json (the best single evaluation: design, results and policy), reevaluations.json, final_design.json (the
final design, with the policies of the baselines' re-evaluations as "references"), final_comparison.mp4,
final_report.pdf, final_metrics.json, images/, DONE when finished.

An interrupted run (after its BO) goes on in its own folder with:

    python evolve.py --run-dir results/<name> --resume
"""
import argparse
import csv as csvlib
import json
import logging
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import cma  # noqa: E402
import numpy as np  # noqa: E402

import design  # noqa: E402
import render  # noqa: E402
import report  # noqa: E402
import rl  # noqa: E402

INFEASIBLE = -10.0  # fitness of a design that cannot hover: the score of an immediate crash
NAMES = [f"x{k:02d}" for k in range(design.N_X)]


def evaluate_batch(xs, seed):
    """Train (rl.halving()) and evaluate the designs xs ([0, 1]^15 each) at once. Returns one record per design
    (fitness: its best surviving training, with that policy; fitness_runs: all its survivors)."""
    phys = [design.physical({"x": [float(v) for v in x]}) for x in xs]
    ok = [i for i, ph in enumerate(phys) if ph["feasible"]]
    recs = [None] * len(xs)
    if ok:
        res = rl.halving([phys[i] for i in ok], seed=seed)
        for r, i in zip(res, ok):
            recs[i] = {"design": phys[i]["design"], **r, "feasible": True, "reason": "", "mass_kg": phys[i]["mass"],
                       "thrust_to_weight": phys[i]["thrust_to_weight"], "hover_tilt_deg": phys[i]["hover_tilt_deg"],
                       "repair_m": phys[i]["repair_m"]}
    for k, ph in enumerate(phys):
        if recs[k] is None:
            recs[k] = {"design": ph["design"], "fitness": INFEASIBLE, "feasible": False, "reason": ph["reason"],
                       "gates_per_episode": 0.0, "missed_per_episode": 0.0, "crash_rate": 1.0, "episodes": 0,
                       "fitness_runs": []}
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=f"results/run_{time.strftime('%Y%m%d_%H%M%S')}",
                    help="output folder (default: results/run_<date>_<time>)")
    ap.add_argument("--bo-evals", type=int, default=400, help="evaluations of the Bayesian optimization")
    ap.add_argument("--cma-evals", type=int, default=400, help="evaluations of CMA-ES (a multiple of --batch)")
    ap.add_argument("--batch", type=int, default=20,
                    help="designs trained at once: the BO batch size and the CMA-ES population size")
    ap.add_argument("--cma-sigma", type=float, default=0.1, help="initial CMA-ES step size (the box is [0, 1]^15)")
    ap.add_argument("--seed", type=int, default=242689)
    ap.add_argument("--resume", action="store_true",
                    help="resume the interrupted run in --run-dir (interrupted after its BO: during CMA-ES, the final "
                         "re-evaluation or the rendering) in that same folder: the other arguments are taken from its "
                         "args.json, the CMA-ES state is rebuilt by replaying its generations and the run goes on")
    a = ap.parse_args()
    run = Path(a.run_dir)
    if a.resume:
        saved = json.loads((run / "args.json").read_text())
        assert saved.get("task", "B") == design.TASK, f"the run is of task {saved.get('task', 'B')}: set TASK"
        for k in ("bo_evals", "cma_evals", "batch", "cma_sigma", "seed"):
            setattr(a, k, saved[k])
    assert a.cma_evals % a.batch == 0, "--cma-evals must be a multiple of --batch (whole CMA-ES generations)"
    run.mkdir(parents=True, exist_ok=True)
    if not a.resume:
        (run / "args.json").write_text(json.dumps(vars(a) | {"task": design.TASK}))

    ph_base = design.baseline_physical()
    if a.resume:  # the run's baseline, as it was (only its numbers are used)
        base = json.loads((run / "baseline.json").read_text())
    else:
        base_res = rl.baseline_policy()[1]
        base = {"design": design.BASELINE, **base_res, "feasible": True, "mass_kg": ph_base["mass"],
                "thrust_to_weight": ph_base["thrust_to_weight"], "hover_tilt_deg": ph_base["hover_tilt_deg"],
                "repair_m": 0.0}
    print(f"baseline (planar hexarotor): fitness {base['fitness']:+.2f}  ({base['gates_per_episode']:.1f} gates, "
          f"{base['missed_per_episode']:.2f} missed, {100 * base['crash_rate']:.1f} % crashes per episode)", flush=True)
    (run / "baseline.json").write_text(json.dumps(base, default=float))
    images = run / "images"
    images.mkdir(exist_ok=True)
    render.still(ph_base, images / "baseline.png")
    best, n_eval, batch_no = None, 0, 0
    start = time.time()
    history = []  # (x, fitness) of every feasible design
    reevals = []  # the re-evaluations: list of (stage, rows)
    old_rows = []
    if not a.resume:
        csv = open(run / "progress.csv", "w")
        csv.write("evaluation,phase,batch,hours,fitness,best_fitness,mean_of_trainings,gates,missed,crash_rate,"
                  "feasible," + ",".join(NAMES) + "\n")
    else:  # take over what the interrupted run did, in its folder
        with open(run / "progress.csv") as fh:
            old_rows = list(csvlib.DictReader(fh))
        n_eval, batch_no = len(old_rows), int(old_rows[-1]["batch"]) + 1
        assert n_eval >= a.bo_evals and n_eval % a.batch == 0 and old_rows[a.bo_evals - 1]["phase"] == "BO", \
            "only a run interrupted after its BO (on a whole batch) can be resumed"
        start = time.time() - float(old_rows[-1]["hours"]) * 3600
        best = json.loads((run / "best_design.json").read_text())
        reevals = json.loads((run / "reevaluations.json").read_text())
        history = [(np.array([float(r[n]) for n in NAMES]), float(r["fitness"])) for r in old_rows
                   if r["feasible"] == "1"]
        if n_eval == a.bo_evals and reevals:  # the BO's re-evaluation (done) took a batch number too
            batch_no += 1
        with open(run / "evaluations.jsonl") as fh:  # in step with progress.csv
            lines = fh.readlines()[:n_eval]
        (run / "evaluations.jsonl").write_text("".join(lines))
        csv = open(run / "progress.csv", "a")
        print(f"resuming {run}: {n_eval} evaluations, best {best['fitness']:+.2f}, {float(old_rows[-1]['hours']):.2f} h "
              f"already used", flush=True)
    jsonl = open(run / "evaluations.jsonl", "a" if a.resume else "w")

    def record(xs, phase):
        nonlocal best, n_eval, batch_no
        tic = time.time()
        recs = evaluate_batch(xs, seed=a.seed * 100003 + batch_no)
        fit = np.array([r["fitness"] for r in recs])
        for x, r in zip(xs, recs):
            n_eval += 1
            if r["feasible"]:
                history.append((np.array(x, float), r["fitness"]))
            if r["feasible"] and (best is None or r["fitness"] > best["fitness"]):
                best = dict(r, evaluation=n_eval, phase=phase)
                (run / "best_design.json").write_text(json.dumps(best))
                render.still(design.physical(best["design"]), images / "best.png", highlight=True, level=True)
            csv.write(f"{n_eval},{phase},{batch_no},{(time.time() - start) / 3600:.5f},{r['fitness']:.4f},"
                      f"{best['fitness'] if best else np.nan:.4f},"
                      f"{np.mean(r['fitness_runs']) if r['feasible'] else np.nan:.4f},"
                      f"{r['gates_per_episode']:.3f},{r['missed_per_episode']:.4f},"
                      f"{r['crash_rate']:.4f},{int(r['feasible'])}," + ",".join(f"{v:.6f}" for v in x) + "\n")
            jsonl.write(json.dumps({k: v for k, v in r.items() if k != "policy"} | {"evaluation": n_eval,
                                                                                 "phase": phase}) + "\n")
        csv.flush()
        jsonl.flush()
        n_inf = sum(not r["feasible"] for r in recs)
        print(f"{phase} batch {batch_no:3d}  evaluations {n_eval:4d}  best {best['fitness'] if best else np.nan:+6.2f} "
              f"(batch best {fit.max():+6.2f}, median {np.median(fit):+6.2f}, {n_inf} cannot hover)  "
              f"[{time.time() - tic:5.0f} s, {(time.time() - start) / 3600:.2f} h]", flush=True)
        batch_no += 1
        write_pdf()
        return fit

    def write_pdf():
        if best is not None:
            report.progress_pdf(run / "progress.pdf", run / "progress.csv", base, best, images / "baseline.png",
                                images / "best.png", reevals)

    def reevaluate(stage, n_top, with_baseline):
        """Re-evaluate the n_top best designs so far (and the references) with rl.multistart(): their fitness is now
        the mean final fitness of their multistarts."""
        nonlocal batch_no
        tic = time.time()
        top = sorted(history, key=lambda h: -h[1])[:n_top]
        named = [(f"#{k + 1} of the {stage}", {"x": [float(v) for v in x]}, f) for k, (x, f) in enumerate(top)]
        if with_baseline:  # references, never chosen: the paper's design for this task and its baselines (planar last)
            named.append((report.LEGACY, {"x": design.PAPER_DESIGN}, np.nan))
            for name in sorted(design.BASELINE_NAMES, key=lambda n: n == design.BASELINE["name"]):
                if design.physical({"name": name})["feasible"]:
                    named.append((f"baseline ({name})", {"name": name},
                                  base["fitness"] if name == design.BASELINE["name"] else np.nan))
                else:
                    print(f"   baseline ({name}) cannot hover: not re-evaluated", flush=True)
        res = rl.multistart([design.physical(d) for _, d, _ in named], rl.REEVAL_MULTISTARTS,
                            seed=a.seed * 100003 + batch_no, log=print)
        batch_no += 1
        rows = []
        for (n, d, f), r in zip(named, res):
            best_ms = int(np.argmax(r["final"]))
            rows.append({"name": n, "design": d, "fitness_before": f, "fitness_runs": r["final"],
                         "fitness": float(np.mean(r["final"])), "std": float(np.std(r["final"])),
                         "max": float(np.max(r["final"])), "picked_on": r["picked_on"], "trainings": r["trainings"],
                         "policy": r["policies"][best_ms], "result": r["final_results"][best_ms]})
        print(f"re-evaluation after the {stage} ({rl.REEVAL_MULTISTARTS} halvings each, picked policy on "
              f"{rl.FINAL_EPISODES} episodes; {time.time() - tic:.0f} s):", flush=True)
        for row in rows:
            print(f"   {row['name']:28s} before {row['fitness_before']:+6.2f}   now {row['fitness']:+6.2f} "
                  f"+- {row['std']:.2f}, max {row['max']:+6.2f}  (" + ", ".join(
                      f"{v:+.1f}" for v in row["fitness_runs"]) + ")", flush=True)
        reevals.append((stage, [{k: v for k, v in row.items() if k != "policy"} for row in rows]))
        (run / "reevaluations.json").write_text(json.dumps(reevals))
        write_pdf()
        return rows

    if not a.resume:
        ## Bayesian optimization (Ax: Sobol initialization, then BoTorch with a Gaussian process), batches of a.batch designs
        from ax.api.client import Client
        from ax.api.configs import RangeParameterConfig
        for name in list(logging.root.manager.loggerDict):
            if name.startswith("ax"):
                logging.getLogger(name).setLevel(logging.WARNING)
        client = Client(random_seed=a.seed)
        client.configure_experiment(parameters=[RangeParameterConfig(name=n, parameter_type="float", bounds=(0.0, 1.0))
                                                for n in NAMES], name=f"hexarotor_task_{design.TASK.lower()}")
        client.configure_optimization(objective="fitness")
        client.configure_generation_strategy(method="fast", initialization_budget=a.batch,
                                             initialization_random_seed=a.seed, initialize_with_center=False,
                                             torch_device="cuda")
        while n_eval < a.bo_evals:
            want, trials = min(a.batch, a.bo_evals - n_eval), {}
            while len(trials) < want:  # right after the Sobol batch Ax may hand out fewer trials than asked for
                new = client.get_next_trials(max_trials=want - len(trials))
                assert new, "Ax generated no trials"
                trials.update(new)
            idx = sorted(trials)
            xs = [np.array([trials[i][n] for n in NAMES]) for i in idx]
            fit = record(xs, "BO")
            for i, f in zip(idx, fit):
                client.complete_trial(trial_index=i, raw_data={"fitness": float(f)})

    ## CMA-ES around the best design of the BO (by its re-evaluation)
    if a.resume and reevals:
        rows = reevals[0][1]
    else:
        rows = reevaluate("BO", 5, with_baseline=False)
    start_x = np.array(max(rows, key=lambda r: r["fitness"])["design"]["x"])
    es = cma.CMAEvolutionStrategy(start_x, a.cma_sigma,
                                  {"bounds": [0, 1], "popsize": a.batch, "seed": a.seed, "verbose": -9})
    for b in range(a.bo_evals // a.batch, n_eval // a.batch):  # replay the generations of the interrupted run
        rs = old_rows[b * a.batch:(b + 1) * a.batch]
        xs = es.ask()
        assert np.allclose(xs, [[float(r[n]) for n in NAMES] for r in rs], atol=1e-5), "cannot replay CMA-ES: other seed?"
        es.tell(xs, [-float(r["fitness"]) for r in rs])
    while n_eval < a.bo_evals + a.cma_evals:
        xs = es.ask()
        fit = record(xs, "CMA-ES")
        es.tell(xs, list(-fit))

    ## the final design: the best re-evaluation of the 5 best designs, next to the baseline's
    if a.resume and (run / "final_design.json").exists() and reevals[-1][0] == "optimization":
        print("the final re-evaluation is already done (final_design.json)", flush=True)
    else:
        rows = reevaluate("optimization", 5, with_baseline=True)
        final = max((r for r in rows if r["name"].startswith("#")), key=lambda r: r["fitness"])
        # its results are those of its best multistart's policy (the one rendered); fitness_mean: its re-evaluation
        (run / "final_design.json").write_text(json.dumps({"design": final["design"], **final["result"],
                                                           "fitness_mean": final["fitness"], "fitness_std": final["std"],
                                                           "fitness_runs": final["fitness_runs"],
                                                           "baseline_fitness_mean": rows[-1]["fitness"],
                                                           "baseline_fitness_runs": rows[-1]["fitness_runs"],
                                                           "policy": final["policy"],
                                                           "references": {r["design"]["name"]: r["policy"] for r in rows
                                                                          if "name" in r["design"]}}))
        print(f"done: final design {final['name']}: fitness {final['fitness']:+.2f} +- {final['std']:.2f} vs baseline "
              f"{rows[-1]['fitness']:+.2f} +- {rows[-1]['std']:.2f} (means of {rl.REEVAL_MULTISTARTS} halvings, "
              f"{rl.FINAL_EPISODES} episodes each)", flush=True)
    print("rendering the final design against the baselines", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "render_evaluation.py"), "--design", str(run / "final_design.json"),
                    "--out", str(run), "--prefix", "final_", "--label", "Final design", "--name", "optimized",
                    "--reevaluations", str(run / "reevaluations.json")], check=True)
    (run / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
