"""Reinforcement learning of the controller of every design: PPO (proximal policy optimization, Schulman et al. 2017)
on COPIES copies of the hexarotor flying task B at once in Newton (sim.py). Every design gets several trainings,
narrowed down by sequential halving (halving(), as the paper): 16 trainings, the best 8 go on after 150 iterations
and the best 4 after 300, to ITERS; the design's fitness is its best survivor. All trainings of a batch of designs run
at the same time, each with its own networks.

Policy: observation (sim.OBS, divided by fixed scales OBS_SCALE) -> MLP (3 x 64, tanh, as in the paper) -> mean
rotor commands in [-1, 1]; Gaussian exploration with a learned standard deviation per rotor. Before PPO, the policy
imitates an LQR of its own design (warmstart.py), so it starts out hovering and flying towards the gates instead of
falling. The critic is a separate MLP of the same size, trained on normalized returns.

Reward (sim.py): the score as it happens, with help at the start that is removed during the training (reward_weights):
progress towards the gate (2 per metre, faded out between 50 and 90 % of the training) and a smaller miss penalty
(10 %, grown to full strength at 40 %). The crash penalty is always the full one.

Training many designs at once: the networks of all designs are stacked (batched matrix products); the loss is the sum
of the designs' losses, gradients are clipped per design and Adam runs per design with its own learning rate, which
adapts to the KL divergence between the old and new policy of that design (as rl_games' "adaptive" schedule), so
the designs do not influence each other. Episodes that end by time-out are bootstrapped with the critic. Each design
keeps the policy of its best training score (mean of BEST_WINDOW iterations), in case the training degrades later.

Settings (from a convergence study, 16 designs x 2 seeds): 4096 drones per design, 32 steps per iteration, 8 epochs
of 4 minibatches, KL target 0.03. ITERS = 900 iterations (designs separate about twice as clearly as with 450); about
12 % of trainings collapse, which the halving drops early (measured on 480 trainings).

Fitness (evaluate): the policy's mean action (no exploration), EVAL_COPIES episodes per design, the same episodes
(start states and gates) for every design: mean episode score (gates - 10 x missed - 10 x crash).
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import design
import sim
import warmstart

HIDDEN, LAYERS = 64, 3
COPIES = 4096  # simulated drones per design during training
HORIZON = 32  # control steps per PPO iteration and drone
ITERS = 900  # PPO iterations of a training (450 reached 96 % of the median fitness after 600; at 900 designs separate
#              about twice as clearly)
EPOCHS, MINIBATCHES = 8, 4
GAMMA, LAMBDA = 0.99, 0.95
CLIP, MAX_GRAD, LR0, KL_TARGET = 0.2, 1.0, 3e-4, 0.03
LR_MIN, LR_MAX = 1e-5, 3e-3
LR_DECAY = True  # the largest learning rate falls linearly to 5 % of LR_MAX over the training (stable long trainings)
LOG_STD0 = np.log(0.3)
BOUNDS_COEF = 0.01  # keeps the mean action inside [-1, 1]
VALUE_COEF = 2.0
BEST_WINDOW = 10  # iterations averaged for keeping the best policy
WARM_START = True  # imitate the design's LQR before PPO (warmstart.py)
## observations are divided by these fixed scales: gate offset [m], rotation, velocity [m/s], body rates [rad/s], thrust
OBS_SCALE = np.array([1.0] * 3 + [1.0] * 6 + [2.0] * 3 + [5.0] * 3 + [1.0] * 6, np.float32)
## reward schedule (fractions of the training): the miss penalty grows from 10 % to full strength until PENALTY_FULL;
## the progress reward fades from PROGRESS0 per metre to 0 between PROGRESS_FADE[0] and [1]
PENALTY_FULL, PROGRESS0, PROGRESS_FADE = 0.4, 2.0, (0.5, 0.9)
EVAL_COPIES = 4096  # evaluation episodes per design
## sequential halving (as the paper; halving()): (trainings still going, from PPO iteration). 16 -> 8 -> 4 had the
## highest and least spread fitness of the schedules simulated on 480 trainings of the paper's design
HALVING = ((16, 0), (8, 150), (4, 300))
## trainings at once on the GPU (each simulates COPIES drones): on an RTX 5090 (32 GB) the throughput is the same from 40
## on and 180 fill its memory. Lower it on GPUs with less memory: TRAIN_CHUNK=20 python evolve.py ...
TRAIN_CHUNK = int(os.environ.get("TRAIN_CHUNK", 80))
REEVAL_MULTISTARTS = 8  # re-evaluation of a design: independent halvings, each one's picked policy (on EVAL_COPIES
#                         episodes) then flies FINAL_EPISODES other episodes
EVAL_SEED = 12345
FINAL_EPISODES = 10 * EVAL_COPIES  # the final evaluation of a design's best policy (the paper: 40000 episodes)
FINAL_SEED = 777  # other episodes than EVAL_SEED's, on which the best training was picked
DEVICE = "cuda:0"
HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"  # cached controllers of the baselines
torch.backends.cuda.matmul.allow_tf32 = True  # tensor cores for the batched matrix products (1.5x faster)
ACTOR = [f"w{k}" for k in range(LAYERS + 1)] + [f"b{k}" for k in range(LAYERS + 1)] + ["log_std"]


def hover_action(ph):
    return np.clip(2 * (ph["hover_thrust"] - design.T_MIN) / (design.T_MAX - design.T_MIN) - 1, -1, 1)


def init_net(gen, D, n_in, n_out, out_gain, out_bias):
    """D orthogonally initialized MLPs: w{k} (D, in, out), b{k} (D, out)."""
    p, sizes = {}, [n_in] + [HIDDEN] * LAYERS + [n_out]
    for k in range(LAYERS + 1):
        w = torch.empty(D, sizes[k], sizes[k + 1])
        for d in range(D):
            torch.nn.init.orthogonal_(w[d], gain=out_gain if k == LAYERS else np.sqrt(2), generator=gen)
        p[f"w{k}"] = w
        p[f"b{k}"] = torch.zeros(D, sizes[k + 1])
    p[f"b{LAYERS}"] = torch.as_tensor(out_bias, dtype=torch.float32)
    return p


def mlp(p, x):
    """Batched MLP: x (D, B, in) -> (D, B, out)."""
    for k in range(LAYERS):
        x = torch.tanh(torch.baddbmm(p[f"b{k}"][:, None], x, p[f"w{k}"]))
    return torch.baddbmm(p[f"b{LAYERS}"][:, None], x, p[f"w{LAYERS}"])


class RunningStats:
    """Running mean and variance per design over the last dimension's features: (D, F)."""

    def __init__(self, D, F, device):
        self.mean = torch.zeros(D, F, device=device)
        self.var = torch.ones(D, F, device=device)
        self.count = torch.full((D, 1), 1e-4, device=device)

    def update(self, x):  # x (D, B, F)
        m, v, n = x.mean(1), x.var(1, unbiased=False), x.shape[1]
        delta, tot = m - self.mean, self.count + n
        self.mean = self.mean + delta * n / tot
        self.var = (self.var * self.count + v * n + delta ** 2 * self.count * n / tot) / tot
        self.count = tot

    def norm(self, x):
        return torch.clamp((x - self.mean[:, None]) / torch.sqrt(self.var[:, None] + 1e-8), -5, 5)


class FixedObsNorm:
    """Observations divided by OBS_SCALE (the same interface as RunningStats; fixed, so that the warm-started policy
    keeps meaning the same thing when PPO starts)."""

    def __init__(self, D, device):
        self.mean = torch.zeros(D, sim.OBS, device=device)
        self.var = torch.tensor(OBS_SCALE ** 2, device=device).expand(D, -1).clone()

    def update(self, x):
        pass

    def norm(self, x):
        return torch.clamp((x - self.mean[:, None]) / torch.sqrt(self.var[:, None] + 1e-8), -5, 5)


class BatchedAdam:
    """Adam with a learning rate per design (the first dimension of every parameter)."""

    def __init__(self, params, lr):
        self.params, self.lr, self.t = params, lr, 0
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]

    @torch.no_grad()
    def step(self):
        self.t += 1
        c1, c2 = 1 - 0.9 ** self.t, 1 - 0.999 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            m.mul_(0.9).add_(p.grad, alpha=0.1)
            v.mul_(0.999).addcmul_(p.grad, p.grad, value=0.001)
            lr = self.lr.view(-1, *[1] * (p.dim() - 1))
            p.sub_(lr * (m / c1) / (torch.sqrt(v / c2) + 1e-5))


def clip_grad_per_design(params, max_norm):
    D = params[0].shape[0]
    norm = torch.sqrt(sum((p.grad.reshape(D, -1) ** 2).sum(1) for p in params))
    scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    for p in params:
        p.grad.mul_(scale.view(-1, *[1] * (p.dim() - 1)))


_swarms = {}


def padded(items):
    """Batches are padded (repeating the last design) to a multiple of 4, so that only a few Newton models are built."""
    return list(items) + [items[-1]] * (-len(items) % 4)


def swarm(D, copies, purpose):
    """Newton models are built once per size and purpose ("train", "eval") and reused (building one takes a few
    seconds)."""
    key = (D, copies, purpose)
    if key not in _swarms:
        _swarms[key] = sim.Swarm(D, copies, DEVICE)
    return _swarms[key]


def policy_to_json(agent, d):
    return {"actor": {k: agent["actor"][k][d].tolist() for k in ACTOR},
            "obs_mean": agent["obs_mean"][d].tolist(), "obs_var": agent["obs_var"][d].tolist()}


def stack_policies(policies, device):
    t = lambda key, k=None: torch.tensor(np.stack([np.array(p[key][k] if k else p[key], np.float32) for p in policies]),
                                         device=device)
    return {"actor": {k: t("actor", k) for k in ACTOR}, "obs_mean": t("obs_mean"), "obs_var": t("obs_var")}


def act_mean(pol, obs):
    o = torch.clamp((obs - pol["obs_mean"][:, None]) / torch.sqrt(pol["obs_var"][:, None] + 1e-8), -5, 5)
    return torch.clamp(mlp(pol["actor"], o), -1, 1)


def init_state(phys, seed, log=None):
    """New trainings, one per design (design.physical() dicts): networks (the actor warm started), Adam's moments,
    learning rates, the return normalization and the kept policies. The first dimension of every tensor is the
    training, so that trainings can be taken out (take()) and trained further (run()) in any grouping."""
    dev = torch.device(DEVICE)
    n = len(phys)
    gen = torch.Generator().manual_seed(seed)
    out_bias = np.stack([hover_action(p) for p in phys])
    actor = {k: v.to(dev) for k, v in init_net(gen, n, sim.OBS, sim.ACT, 0.01, out_bias).items()}
    actor["log_std"] = torch.full((n, sim.ACT), float(LOG_STD0), device=dev)
    critic = {k: v.to(dev) for k, v in init_net(gen, n, sim.OBS, 1, 1.0, np.zeros((n, 1))).items()}
    if WARM_START:
        tic = time.time()
        for p in actor.values():
            p.requires_grad_(True)
        err = warmstart.fit(actor, phys, seed, mlp, FixedObsNorm(n, dev).norm, dev)
        actor = {k: v.detach() for k, v in actor.items()}
        if log:
            log(f"      warm start (imitating each design's LQR): {time.time() - tic:.0f} s, "
                f"rms error {np.sqrt(err).mean():.3f}")
    params = list(actor.values()) + list(critic.values())
    return {"actor": actor, "critic": critic, "m": [torch.zeros_like(p) for p in params],
            "v": [torch.zeros_like(p) for p in params], "t": 0, "lr": torch.full((n,), LR0, device=dev),
            "ret_mean": torch.zeros(n, 1, device=dev), "ret_var": torch.ones(n, 1, device=dev),
            "ret_count": torch.full((n, 1), 1e-4, device=dev),
            "best_actor": {k: v.clone() for k, v in actor.items()},
            "best_score": torch.full((n,), -np.inf, device=dev)}


def _combine(states, f):
    """A state made of f(the same tensor of every state)."""
    out = {}
    for k, v in states[0].items():
        if isinstance(v, dict):
            out[k] = {kk: f([s[k][kk] for s in states]) for kk in v}
        elif isinstance(v, list):
            out[k] = [f([s[k][j] for s in states]) for j in range(len(v))]
        else:
            out[k] = f([s[k] for s in states]) if torch.is_tensor(v) else v
    return out


def take(state, idx):
    """The trainings idx of a state (a copy)."""
    i = torch.as_tensor(np.asarray(idx), device=DEVICE)
    return _combine([state], lambda xs: xs[0][i].clone())


def concat(states):
    return _combine(states, torch.cat)


def put(state, idx, sub):
    """Write the trainings of sub (take(state, idx), trained further) back into state."""
    i = torch.as_tensor(np.asarray(idx), device=DEVICE)
    _combine([state, sub], lambda xs: xs[0].__setitem__(i, xs[1]))
    state["t"] = sub["t"]


def policies(state, kept=True):
    """The policies (JSON-able dicts) of a state's trainings: the kept ones (best training score) or the current ones."""
    pols = snapshot(state["best_actor"] if kept else state["actor"], FixedObsNorm(len(state["lr"]), DEVICE))
    return [policy_to_json(pols, d) for d in range(len(state["lr"]))]


def train(phys, iters, seed, log=None, eval_every=0, copies=COPIES, horizon=HORIZON):
    """Train one policy per design (design.physical() dicts, all feasible) for `iters` PPO iterations, all at once.
    Returns the policies (JSON-able dicts, one per design) and a history (run())."""
    state = init_state(phys, seed, log)
    hist = run(state, phys, 0, iters, iters, seed, log, eval_every, copies, horizon)
    return policies(state), hist


def run(state, phys, it0, it1, iters, seed, log=None, eval_every=0, copies=COPIES, horizon=HORIZON):
    """Train the trainings of a state (one per design of phys, all feasible) from PPO iteration it0 to it1 of a
    training of `iters` iterations (the learning rate and the reward follow that schedule), all at once, on new
    episodes. Updates the state and returns a history: per iteration the mean training episode score per training,
    and every eval_every iterations (if > 0) the evaluation score of the current policies."""
    dev = torch.device(DEVICE)
    n_designs, phys = len(phys), padded(phys)
    D = len(phys)
    pad = torch.as_tensor(list(range(n_designs)) + [n_designs - 1] * (D - n_designs), device=dev)
    P = lambda x: x[pad].clone()
    actor = {k: P(v) for k, v in state["actor"].items()}
    critic = {k: P(v) for k, v in state["critic"].items()}
    params = list(actor.values()) + list(critic.values())
    for p in params:
        p.requires_grad_(True)
    lr = P(state["lr"])
    opt = BatchedAdam(params, lr)
    opt.m, opt.v, opt.t = [P(x) for x in state["m"]], [P(x) for x in state["v"]], state["t"]
    obs_rms, ret_rms = FixedObsNorm(D, dev), RunningStats(D, 1, dev)
    ret_rms.mean, ret_rms.var, ret_rms.count = P(state["ret_mean"]), P(state["ret_var"]), P(state["ret_count"])
    ## the policy of each training with the best training score (mean of the last BEST_WINDOW iterations) is kept
    best_actor = {k: P(v) for k, v in state["best_actor"].items()}
    best_score = P(state["best_score"])
    env = swarm(D, copies, "train")
    env.set_designs(phys)
    env.reset(seed, common=False)
    T, M = horizon, copies
    O = torch.zeros(T, D, M, sim.OBS, device=dev)
    A = torch.zeros(T, D, M, sim.ACT, device=dev)
    LP, V, R, DONE = (torch.zeros(T, D, M, device=dev) for _ in range(4))
    torch.manual_seed(seed)
    hist = {"train_score": [], "eval_iter": [], "eval_score": []}
    prev = env.episode_stats()
    prev_log, it_log = prev, it0
    t_sim = t_learn = 0.0
    value = lambda o: mlp(critic, obs_rms.norm(o))[..., 0] * torch.sqrt(ret_rms.var + 1e-8) + ret_rms.mean
    for it in range(it0, it1):
        tic = time.time()
        env.set_reward_weights(*reward_weights(it / iters))
        with torch.no_grad():
            for t in range(T):
                o = env.t_obs.clone()
                obs_rms.update(o)
                mean = mlp(actor, obs_rms.norm(o))
                std = torch.exp(actor["log_std"])[:, None]
                a = mean + std * torch.randn_like(mean)
                env.t_cmd.copy_(a)
                O[t], A[t] = o, a
                LP[t] = (-0.5 * ((a - mean) / std) ** 2 - torch.log(std) - 0.9189385).sum(-1)
                V[t] = value(o)
                env.step(training=True)
                r = env.t_reward + GAMMA * env.t_timeout * value(env.t_obs_end)
                R[t], DONE[t] = r, env.t_done
            v_last = value(env.t_obs)
        torch.cuda.synchronize()
        t_sim += time.time() - tic
        tic = time.time()
        with torch.no_grad():
            adv, gae = torch.zeros_like(R), torch.zeros(D, M, device=dev)
            for k in reversed(range(T)):
                v_next = v_last if k == T - 1 else V[k + 1]
                nd = 1.0 - DONE[k]
                gae = R[k] + GAMMA * nd * v_next - V[k] + GAMMA * LAMBDA * nd * gae
                adv[k] = gae
            ret = adv + V
            flat = lambda x: x.transpose(0, 1).reshape(D, T * M, *x.shape[3:])
            O_, A_, LP_, ADV, RET = flat(O), flat(A), flat(LP), flat(adv), flat(ret)
            ret_rms.update(RET[..., None])
            RETN = (RET - ret_rms.mean) / torch.sqrt(ret_rms.var + 1e-8)
            ADV = (ADV - ADV.mean(1, keepdim=True)) / (ADV.std(1, keepdim=True) + 1e-8)
            ON = obs_rms.norm(O_)
        size = T * M // MINIBATCHES
        for _ in range(EPOCHS):
            perm = torch.randperm(T * M, device=dev)
            kls = []
            for mb in range(MINIBATCHES):
                i = perm[mb * size:(mb + 1) * size]
                o, a, lp0, adv_, ret_ = ON[:, i], A_[:, i], LP_[:, i], ADV[:, i], RETN[:, i]
                mean = mlp(actor, o)
                log_std = actor["log_std"][:, None]
                lp = (-0.5 * ((a - mean) / torch.exp(log_std)) ** 2 - log_std - 0.9189385).sum(-1)
                ratio = torch.exp(lp - lp0)
                pg = -torch.min(ratio * adv_, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv_).mean(1)
                vf = ((mlp(critic, o)[..., 0] - ret_) ** 2).mean(1)
                bounds = (torch.relu(mean - 1.0) ** 2 + torch.relu(-1.0 - mean) ** 2).sum(-1).mean(1)
                for p in params:
                    p.grad = None
                (pg + VALUE_COEF * vf + BOUNDS_COEF * bounds).sum().backward()
                clip_grad_per_design(list(actor.values()), MAX_GRAD)
                clip_grad_per_design(list(critic.values()), MAX_GRAD)
                opt.step()
                with torch.no_grad():
                    kls.append((lp0 - lp).mean(1))
            with torch.no_grad():  # adaptive learning rate per design (on the approximate KL of the epoch)
                kl = torch.stack(kls).mean(0)
                lr_max = LR_MAX * (max(0.05, 1 - it / iters) if LR_DECAY else 1.0)
                lr.copy_(torch.where(kl > 2 * KL_TARGET, lr / 1.5, torch.where(kl < KL_TARGET / 2, lr * 1.5, lr))
                         .clamp(LR_MIN, lr_max))
        torch.cuda.synchronize()
        t_learn += time.time() - tic
        st = env.episode_stats()
        n_ep = st["episodes"] - prev["episodes"]
        score = np.where(n_ep > 0, (st["score_sum"] - prev["score_sum"]) / np.maximum(n_ep, 1), np.nan)
        prev = st
        hist["train_score"].append(score[:n_designs].tolist())
        recent = np.array(hist["train_score"][-BEST_WINDOW:], float)
        if len(recent) == BEST_WINDOW:
            ma = torch.tensor(np.concatenate([np.nanmean(recent, 0), np.full(D - n_designs, -np.inf)]), device=dev,
                              dtype=torch.float32)
            better = torch.nan_to_num(ma, nan=-np.inf) > best_score
            best_score = torch.where(better, ma, best_score)
            for k, v in actor.items():
                best_actor[k][better] = v.detach()[better]
        if eval_every and ((it + 1) % eval_every == 0 or it == iters - 1):
            pols = snapshot(actor, obs_rms)
            res = evaluate(phys[:n_designs], [policy_to_json(pols, d) for d in range(n_designs)])
            hist["eval_iter"].append(it + 1)
            hist["eval_score"].append([r["fitness"] for r in res])
        if log and ((it + 1) % 25 == 0 or it == it1 - 1):
            flown = (it + 1 - it_log) * T * M * D / sim.EPISODE_STEPS  # 5 s flights since the last log line
            per_ep = lambda k: np.sum(st[k] - prev_log[k]) / flown
            msg = f"      PPO {it + 1}/{iters}: training score {np.nanmean(score):+6.2f} (best design " \
                  f"{np.nanmax(score) if np.isfinite(score).any() else np.nan:+6.2f}), lr {lr.mean().item():.1e}, " \
                  f"per 5 s: {per_ep('passed'):.1f} gates, {per_ep('missed'):.2f} missed, " \
                  f"{per_ep('crashed'):.2f} crashes, speed {torch.linalg.norm(O[..., 9:12], dim=-1).mean().item():.2f} m/s"
            prev_log, it_log = st, it + 1
            if hist["eval_iter"] and hist["eval_iter"][-1] == it + 1:
                msg += f", evaluation {np.mean(hist['eval_score'][-1]):+6.2f}"
            log(msg)
    if log and it1 > it0:
        steps = (it1 - it0) * T * M * D
        log(f"      PPO: {D} designs x {M} drones, {steps / 1e6:.0f} M steps, simulation {t_sim:.0f} s, "
            f"learning {t_learn:.0f} s ({steps / (t_sim + t_learn) / 1e6:.1f} M steps/s)")
    n = n_designs
    state.update(actor={k: v.detach()[:n] for k, v in actor.items()},
                 critic={k: v.detach()[:n] for k, v in critic.items()}, m=[x[:n] for x in opt.m],
                 v=[x[:n] for x in opt.v], t=opt.t, lr=lr[:n], ret_mean=ret_rms.mean[:n], ret_var=ret_rms.var[:n],
                 ret_count=ret_rms.count[:n], best_actor={k: v[:n] for k, v in best_actor.items()},
                 best_score=best_score[:n])
    hist["best_training_score"] = best_score[:n].tolist()
    return hist


def reward_weights(frac):
    """(w_progress, w_miss) at a fraction of the training (sim.py)."""
    w_pen = 0.1 + 0.9 * min(1.0, frac / PENALTY_FULL)
    w_prog = PROGRESS0 * float(np.clip((PROGRESS_FADE[1] - frac) / (PROGRESS_FADE[1] - PROGRESS_FADE[0]), 0, 1))
    return w_prog, w_pen


def snapshot(actor, obs_rms):
    return {"actor": {k: v.detach().cpu().numpy() for k, v in actor.items()},
            "obs_mean": obs_rms.mean.cpu().numpy(), "obs_var": obs_rms.var.cpu().numpy()}


@torch.no_grad()
def evaluate(phys, policies, copies=EVAL_COPIES, seed=EVAL_SEED, record=0):
    """Fly every design with its policy (mean action) on the same `copies` episodes; one result dict per design. With
    record > 0, also returns the poses of the first `record` copies of every design at every control step."""
    n_designs, phys, policies = len(phys), padded(phys), padded(policies)
    D = len(phys)
    env = swarm(D, copies, "eval")
    env.set_designs(phys)
    env.reset(seed, common=True)
    pol = stack_policies(policies, torch.device(DEVICE))
    frames = []
    for _ in range(sim.EPISODE_STEPS):
        if record:
            frames.append((*env.pose(), env.gate.numpy(), env.active.numpy()))
        env.t_cmd.copy_(act_mean(pol, env.t_obs))
        env.step(training=False)
    st = env.episode_stats()
    assert np.all(st["episodes"] == copies), st["episodes"]
    res = [{"fitness": float(st["score_sum"][d] / copies), "gates_per_episode": float(st["passed"][d] / copies),
            "missed_per_episode": float(st["missed"][d] / copies), "crash_rate": float(st["crashed"][d] / copies),
            "episodes": int(copies)} for d in range(n_designs)]
    if record:
        idx = (np.arange(n_designs)[:, None] * copies + np.arange(record)[None]).reshape(-1)
        rec = {k: np.stack([f[j][idx] for f in frames]).reshape(len(frames), n_designs, record, -1).squeeze(-1)
               if k == "active" else np.stack([f[j][idx] for f in frames]).reshape(len(frames), n_designs, record, -1)
               for j, k in enumerate(("pos", "quat", "gate", "active"))}
        rec["crashed"] = env.crashed.numpy()[idx].reshape(n_designs, record) > 0
        return res, rec
    return res


def evaluate_final(phys, policies):
    """evaluate() on FINAL_EPISODES episodes (in rounds of EVAL_COPIES, each with its own episodes), not the ones the
    best training was picked on: the score of a picked policy on its own episodes is inflated by their luck."""
    rounds = [evaluate(phys, policies, copies=EVAL_COPIES, seed=FINAL_SEED + k)
              for k in range(-(-FINAL_EPISODES // EVAL_COPIES))]
    return [{**{key: float(np.mean([r[d][key] for r in rounds])) for key in rounds[0][d] if key != "episodes"},
             "episodes": sum(r[d]["episodes"] for r in rounds)} for d in range(len(phys))]


def halving(phys, seed, log=None):
    """Sequential halving for every design (design.physical() dicts, all feasible), as the paper (its section
    III-E.1): HALVING[0][0] trainings per design on the schedule of ITERS iterations; from the iteration of each later
    step of HALVING on, only the design's best ones go on (by evaluate() of their current policies). At the end the
    survivors are evaluated with their kept policies and the best is the design's result. The trainings run in chunks
    of TRAIN_CHUNK. Returns per design that result (evaluate()) with "policy", "fitness_runs" (the fitness of the
    survivors) and "checkpoints" (per later step of HALVING, the scores of the trainings it chose from)."""
    n0 = HALVING[0][0]
    every = [p for p in phys for _ in range(n0)]
    owner = np.repeat(np.arange(len(phys)), n0)
    chunks = lambda ids: [ids[c:c + TRAIN_CHUNK] for c in range(0, len(ids), TRAIN_CHUNK)]
    score_of = lambda ids, kept: [r for c in chunks(ids) for r in evaluate([every[i] for i in c],
                                                                             policies(take(state, c), kept))]
    alive = np.arange(len(every))
    state = concat([init_state([every[i] for i in c], seed * 100003 + int(c[0]), log) for c in chunks(alive)])
    checkpoints = [[] for _ in phys]
    for s, (_, start) in enumerate(HALVING):
        end = HALVING[s + 1][1] if s + 1 < len(HALVING) else ITERS
        for c in chunks(alive):
            if log:
                log(f"[rl] halving: iterations {start}-{end} of {ITERS} for trainings {c[0] + 1}.. "
                    f"({len(c)} of {len(alive)} still going, {len(phys)} designs)")
            sub = take(state, c)
            run(sub, [every[i] for i in c], start, end, ITERS, seed=seed * 100003 + 10007 * (s + 1) + int(c[0]),
                log=log)
            put(state, c, sub)
        if s + 1 < len(HALVING):
            score = np.array([r["fitness"] for r in score_of(alive, kept=False)])
            nxt = []
            for g in range(len(phys)):
                mine = owner[alive] == g
                checkpoints[g].append(score[mine].tolist())
                nxt += alive[mine][np.argsort(-score[mine], kind="stable")[:HALVING[s + 1][0]]].tolist()
            alive = np.array(nxt)
    res, pols = score_of(alive, kept=True), policies(take(state, alive))
    out = []
    for g in range(len(phys)):
        mine = np.flatnonzero(owner[alive] == g)
        j = mine[int(np.argmax([res[i]["fitness"] for i in mine]))]
        out.append({**res[j], "fitness_runs": [res[i]["fitness"] for i in mine], "checkpoints": checkpoints[g],
                    "policy": pols[j]})
    return out


def multistart(phys, n, seed, log=None):
    """n independent sequential halvings (halving()) of every design, all at once; the policy each one picked (on the
    EVAL_COPIES episodes) then flies evaluate_final(). Returns per design: final (the final fitness of every
    multistart), picked_on (the fitness its policy was picked with), trainings (the fitness of the survivors of every
    multistart), final_results (evaluate_final()'s results) and policies (the n picked policies)."""
    D = len(phys)
    res = halving([p for _ in range(n) for p in phys], seed, log)
    final = []
    for i in range(0, n * D, TRAIN_CHUNK):
        final += evaluate_final([phys[j % D] for j in range(i, min(n * D, i + TRAIN_CHUNK))],
                                [r["policy"] for r in res[i:i + TRAIN_CHUNK]])
    return [{"final": [final[s * D + d]["fitness"] for s in range(n)],
             "picked_on": [res[s * D + d]["fitness"] for s in range(n)],
             "trainings": [res[s * D + d]["fitness_runs"] for s in range(n)],
             "final_results": [final[s * D + d] for s in range(n)],
             "policies": [res[s * D + d]["policy"] for s in range(n)]} for d in range(D)]

## ------------------------------------------------------------------------------------------------- the baseline
def baseline_path(name):
    """A baseline's policy is cached under the hash of the code it depends on: editing it triggers a retraining."""
    digest = hashlib.sha1(b"".join((HERE / f).read_bytes() for f in ("rl.py", "sim.py", "design.py"))).hexdigest()
    return BUILD / f"baseline_policy_task{design.TASK}_{name}_{digest[:10]}.json"


def baseline_policies(names, log=print):
    """{name: (policy, its evaluation)} of the baselines (design.BASELINE_NAMES): trained the first time (the same
    training as every design: halving(), all the missing ones together), then read from build/."""
    todo = [n for n in names if not baseline_path(n).exists()]
    if todo:
        log(f"[rl] training the controllers of the baselines {', '.join(todo)} (sequential halving {HALVING}, "
            f"{ITERS} PPO iterations)")
        for n, r in zip(todo, halving([design.physical({"name": n}) for n in todo], seed=0, log=log)):
            policy = r.pop("policy")
            path = baseline_path(n)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps({"policy": policy, "result": r}))
            tmp.rename(path)
            log(f"[rl] baseline {n}: fitness {r['fitness']:+.2f} (trainings: "
                + ", ".join(f"{f:+.2f}" for f in r["fitness_runs"]) + f"), saved {path}")
    saved = {n: json.loads(baseline_path(n).read_text()) for n in names}
    return {n: (d["policy"], d["result"]) for n, d in saved.items()}


def baseline_policy(log=print):
    """The policy of the baseline of the reports (design.BASELINE) and its evaluation."""
    return baseline_policies([design.BASELINE["name"]], log)[design.BASELINE["name"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", action="store_true", help="train the controllers of the baselines")
    a = ap.parse_args()
    if a.baseline:
        baseline_policies([n for n in design.BASELINE_NAMES if design.physical({"name": n})["feasible"]])


if __name__ == "__main__":
    main()
