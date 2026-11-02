"""Warm start of the policies: before reinforcement learning, every design's policy learns to imitate a linear
quadratic regulator (LQR) of that design, so it starts out hovering and flying towards its gate instead of falling.

The LQR is computed from the design's own linearized dynamics around its hover point (hover thrusts and attitude,
which is tilted for some designs): position, velocity, attitude error, body rates and the rotor thrusts (first-order
lag), controlled by the six rotor commands. The policy (the same network that PPO then trains) is fitted to the
LQR's commands on states sampled around the task (gate up to ~1 m away, tilted, moving, spinning), all designs at
once. This takes a few seconds and removes the phase in which a fresh policy only crashes (where training can
also get stuck).
"""
import numpy as np
import scipy.linalg
import torch

import design
import sim

## LQR weights: position [m], velocity [m/s], attitude [rad], body rates [rad/s]; commands relative to T_MAX
Q_DIAG = np.array([4.0] * 3 + [0.5] * 3 + [2.0] * 3 + [0.05] * 3 + [0.0] * 6)
R_SCALE = 1.0
TARGET_CLIP = 0.6  # the regulator flies towards a point at most this far [m] (bounded speed)
POOL, BATCH, STEPS, LR = 65536, 4096, 1500, 3e-3  # samples per design, minibatch, Adam steps


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def hover_rotation(ph):
    """Attitude in which the design hovers: its hover force points up."""
    f = ph["alloc"][:3] @ ph["hover_thrust"]
    f /= np.linalg.norm(f)
    a = np.cross(f, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-9:
        return np.eye(3)
    a /= np.linalg.norm(a)
    return scipy.linalg.expm(skew(a) * np.arccos(np.clip(f[2], -1, 1)))


def lqr_gain(ph):
    """Gain K (6, 18) of u = T_hover - K s, s = (position, velocity, attitude error (body), body rates, thrust error)."""
    Rh = hover_rotation(ph)
    Fm, Tm = ph["alloc"][:3], ph["alloc"][3:]
    Fh = Fm @ ph["hover_thrust"]
    A = np.zeros((18, 18))
    B = np.zeros((18, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 6:9] = -Rh @ skew(Fh) / ph["mass"]
    A[3:6, 12:18] = Rh @ Fm / ph["mass"]
    A[6:9, 9:12] = np.eye(3)
    A[9:12, 12:18] = np.linalg.solve(ph["inertia"], Tm)
    A[12:18, 12:18] = -np.eye(6) / design.TAU_MOTOR
    B[12:18] = np.eye(6) / design.TAU_MOTOR
    Rw = R_SCALE * np.eye(6) / design.T_MAX ** 2
    P = scipy.linalg.solve_continuous_are(A, B, np.diag(Q_DIAG) + 1e-9 * np.eye(18), Rw)
    return np.linalg.solve(Rw, B.T @ P), Rh


def _rotvec_to_matrix(w):
    """(n, 3) rotation vectors -> (n, 3, 3)."""
    th = np.linalg.norm(w, axis=1, keepdims=True)
    k = w / np.maximum(th, 1e-12)
    K = np.zeros((len(w), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _matrix_to_rotvec(R):
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    return v * (th / np.maximum(2 * np.sin(th), 1e-9))[:, None]


def dataset(ph, K, Rh, rng, n):
    """Observations (n, OBS) as sim.write_obs makes them and the LQR's commands (n, ACT) in [-1, 1]."""
    e = rng.uniform([-0.3, -0.8, -0.5], [1.3, 0.8, 0.5], (n, 3))  # gate - position (world)
    v = rng.uniform(-1.5, 1.5, (n, 3))
    yaw = rng.uniform(-np.pi / 4, np.pi / 4, n)
    tilt = rng.normal(0, 0.25, (n, 3))
    R = _rotvec_to_matrix(np.stack([np.zeros(n), np.zeros(n), yaw], 1)) @ Rh @ _rotvec_to_matrix(tilt)
    wb = rng.normal(0, 1.5, (n, 3))
    thrust = np.clip(ph["hover_thrust"] + rng.normal(0, 0.3, (n, 6)), design.T_MIN, design.T_MAX)
    ## the regulator holds the yaw it has (heading does not matter for the task) and flies towards the gate
    Rref = _rotvec_to_matrix(np.stack([np.zeros(n), np.zeros(n), yaw], 1)) @ Rh
    err = e * np.minimum(1.0, TARGET_CLIP / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9))
    s = np.concatenate([-err, v, _matrix_to_rotvec(np.einsum("nji,njk->nik", Rref, R)), wb,
                        thrust - ph["hover_thrust"]], 1)
    ## the regulator works in the frame of its yaw: rotate world position and velocity errors into it
    Ry = _rotvec_to_matrix(np.stack([np.zeros(n), np.zeros(n), yaw], 1))
    s[:, 0:3] = np.einsum("nji,nj->ni", Ry, s[:, 0:3])
    s[:, 3:6] = np.einsum("nji,nj->ni", Ry, s[:, 3:6])
    u = np.clip(ph["hover_thrust"] - s @ K.T, design.T_MIN, design.T_MAX)
    act = 2 * (u - design.T_MIN) / (design.T_MAX - design.T_MIN) - 1
    obs = np.concatenate([e, R[:, 0, :], R[:, 1, :], v, wb, 2 * (thrust - design.T_MIN) / (design.T_MAX - design.T_MIN) - 1],
                         1)
    return obs.astype(np.float32), act.astype(np.float32)


def fit(actor, phys, seed, mlp, obs_norm, device):
    """Fit the actors (batched, D designs) to their designs' LQR. Returns the final mean squared error per design."""
    rng = np.random.default_rng(seed)
    data = [dataset(ph, *lqr_gain(ph), rng, POOL) for ph in phys]
    O = torch.tensor(np.stack([d[0] for d in data]), device=device)
    A = torch.tensor(np.stack([d[1] for d in data]), device=device)
    params = [actor[k] for k in actor if k != "log_std"]
    opt = torch.optim.Adam(params, lr=LR)
    gen = torch.Generator(device=device).manual_seed(seed)
    for _ in range(STEPS):
        i = torch.randint(0, POOL, (BATCH,), device=device, generator=gen)
        loss = ((mlp(actor, obs_norm(O[:, i])) - A[:, i]) ** 2).mean((1, 2))
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    return loss.detach().cpu().numpy()
