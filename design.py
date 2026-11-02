"""Hexarotor design: 15 numbers in [0, 1] -> motor poses, mass, inertia and rotor model (Arza et al., the
"constraints encoding" airframe paper).

A design is a dict {"x": [15 numbers in [0, 1]]}: 5 numbers per motor for motors 0, 1, 2 (r, theta, phi of the motor
position in spherical coordinates around the centre of the electronics cage, and the roll and yaw of the motor), and
motors 3, 4, 5 are their mirror images in the x-z plane, so the airframe is left-right symmetric. The baseline is
{"name": "planar"}: a flat hexarotor, all propellers pointing up.

Everything else follows from the motor poses ([physical]):
  repair     propellers may not blow into each other or into the cage: each propeller's airflow is a cylinder (radius
             of the propeller + 1 mm, length 4 propeller radii, centred on the motor) and the cage a 10 x 8 x 6 cm box.
             Overlapping motors are pushed apart (keeping the symmetry) until nothing overlaps.
  mass       cage + battery (258 g), motors with propellers (12 g each) and carbon arms from the side of the cage to
             each motor (160 g/m), as in the paper.
  hover      a design that cannot hover (no thrusts within the motor limits give a force of at least its weight with
             zero torque, in any attitude) is not trained: it gets the fitness of an immediate crash.
Motors and propellers: Xing2 3-inch (as in the paper's sim2real experiments): thrust = kf * rps^2 between 83 and 400
rps, first-order lag of 47 ms, drag torque = 0.01 * thrust.
"""
import os
import itertools
import warnings

import fcl
import numpy as np

warnings.filterwarnings("ignore", message="Values in x were outside bounds")

G = 9.81
TASK = os.environ.get("TASK", "B")  # the task of the paper to optimize for: A or B (sim.py)
assert TASK in ("A", "B"), f"TASK must be A or B, not {TASK}"
N_X = 15  # numbers per design
N_MOTORS = 6

## Xing2 3 inch motor + propeller (aerial_gym_dev motor 5)
KF = 1.286412e-5  # thrust per rps^2 [N/rps^2]
RPS_MIN, RPS_MAX = 83.0, 400.0
T_MIN, T_MAX = KF * RPS_MIN ** 2, KF * RPS_MAX ** 2  # thrust range per motor [N]
TAU_MOTOR = 0.047  # rotor time constant [s]
CQ = 0.01  # drag torque / thrust [m]
PROP_D = 0.0762  # propeller diameter [m]
M_MOTOR = 0.012  # motor + propeller [kg]

## frame
M_BASE = 0.258  # electronics cage, battery, flight controller [kg]
I_BASE = np.array([[3.326e-4, 2.6e-6, 1.49e-5], [2.6e-6, 4.005e-4, 7e-7], [1.49e-5, 7e-7, 2.607e-4]])  # [kg m^2]
CAGE = np.array([0.10, 0.08, 0.06])  # electronics cage [m]
ARM_KG_PER_M = 0.16
## the airflow of a propeller (repair): a cylinder along the thrust axis
AIR_R, AIR_LEN = PROP_D / 2 + 0.001, 4 * PROP_D / 2

## motor spin directions (+1: the drag torque is along the thrust axis), as in the paper's code: the encoded designs
## (mirror images spin the other way) and the baselines of the paper
DIRECTIONS = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
DIRECTIONS_BASELINES = -DIRECTIONS
R_RANGE = (0.11875, 0.20)  # distance of a motor from the centre of the cage [m]
ANGLE_LIMIT = np.pi / 6  # theta and phi in [30, 150] degrees
## baseline arm length [m]. Task B: roll and pitch inertia of the paper's optimized task B design (406 g). Task A: the
## paper's rule (the average arm length of its optimized task A design), as in the legacy code
PLANAR_ARM = {"A": 0.14411971030614243, "B": 0.129}[TASK]
BASELINE = {"name": "planar"}  # the baseline of the videos and reports
BASELINE_NAMES = ("planar", "franchi", "rajappa", "shen")  # the baselines of the paper
## the paper's optimized design for task B (its 15 numbers; the paper's sim2real experiments)
PAPER_TASK_B = [0.07389526911335745, 0.5737461909178113, 0.18748030828689816, 0.5240017620246588, 0.3641657205395625,
                0.0003355726229867873, 0.5038571114014887, 0.6234686844552624, 0.0026747318769536456,
                0.7235971185690883, 0.00035091783904403625, 0.5130041058440267, 0.9755563380229799,
                0.5071058247064041, 0.45280575608068707]
## the paper's optimized design for task A
PAPER_TASK_A = [0.451101198405499, 0.6856633882638767, 0.26363187550753114, 0.3377464460378437, 0.5992781784525718,
                0.03326610267550566, 0.6531937960610272, 0.4913300472583667, 0.10021143915787441, 0.9831866090934073,
                0.4011815722022327, 0.1832427704794185, 0.6548688259682616, 2.52230693155177e-05, 0.6011774081319896]
PAPER_DESIGN = {"A": PAPER_TASK_A, "B": PAPER_TASK_B}[TASK]  # the paper's optimized design for the task of this run


def mirror(v):
    return v * np.array([1.0, -1.0, 1.0])


def rotation_xyz(roll, pitch, yaw):
    """Rotation matrix of extrinsic x-y-z Euler angles [rad] (URDF rpy): Rz @ Ry @ Rx."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def motors_from_x(x):
    """[0, 1]^15 -> motor positions (6, 3) [m] and Euler angles (6, 3) [deg] (the paper's encoding)."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0).copy()
    assert x.shape == (N_X,), x.shape
    ## motor 0 is in front (phi in [30, 90] deg), motor 2 behind (phi in [90, 150] deg); their roll is within +-45 deg
    x[2], x[12] = x[2] / 2, 0.5 + x[12] / 2
    x[3], x[13] = 0.25 + x[3] / 2, 0.25 + x[13] / 2
    pos, eul = np.zeros((N_MOTORS, 3)), np.zeros((N_MOTORS, 3))
    for i in range(3):
        r = R_RANGE[0] + x[5 * i] * (R_RANGE[1] - R_RANGE[0])
        th = ANGLE_LIMIT + x[5 * i + 1] * (np.pi - 2 * ANGLE_LIMIT)
        ph = ANGLE_LIMIT + x[5 * i + 2] * (np.pi - 2 * ANGLE_LIMIT)
        pos[i] = r * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        eul[i] = [-90 + 180 * x[5 * i + 3], 0.0, 360 * x[5 * i + 4]]
        pos[i + 3] = mirror(pos[i])
        eul[i + 3] = [-eul[i][0], 0.0, 360 - eul[i][2]]
    return pos, eul


def motors_planar(arm):
    """The baseline: a flat hexarotor, motors every 60 degrees (mirror pairs 0-3, 1-4, 2-5), all pointing up."""
    ang = np.deg2rad([-30, -90, -150, 30, 90, 150])
    pos = arm * np.stack([np.cos(ang), np.sin(ang), np.zeros(6)], 1)
    return pos, np.zeros((N_MOTORS, 3))


def motors_baseline(name, arm):
    """The baselines of the paper (its baseline_designs.py): motor positions (6, 3) and Euler angles (6, 3) [deg].
    planar: all propellers up; franchi (Franchi et al. 2018) and rajappa (Rajappa et al. 2015): propellers tilted
    alternately on a flat hexagon; shen (Shen et al. 2022): propellers tilted 45 degrees."""
    if name == "planar":
        return motors_planar(arm)
    ring = np.array([[0.8660254037844386, -0.5, 0], [0, -1, 0], [-0.8660254037844386, -0.5, 0],
                     [-0.8660254037844386, 0.5, 0], [0, 1, 0], [0.8660254037844386, 0.5, 0]])
    if name in ("franchi", "rajappa"):
        a, b = (35.0, 25.0) if name == "franchi" else (13.6, 10.6)
        alphas = [a, -a, a, -a, a, -a] if name == "franchi" else [-a, a, -a, a, -a, a]
        betas = [-b] * 6 if name == "franchi" else [b, -b, b, b, -b, b]
        pos = ring * arm
        eul = np.array([[al, be, np.degrees(np.arctan2(p[1], p[0]))] for p, al, be in zip(pos, alphas, betas)])
        pos[[3, 5]], eul[[3, 5]] = pos[[5, 3]], eul[[5, 3]]
        return pos, eul
    assert name == "shen", name
    P = np.array([[1.0, 0.5, -0.5, -1.0, -0.5, 0.5], [0, np.sqrt(3) / 2, np.sqrt(3) / 2, 0, -np.sqrt(3) / 2,
                                                       -np.sqrt(3) / 2], [0] * 6]) * arm
    Rz90 = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    pos = (Rz90 @ P).T
    st = np.sin(np.radians(45.0))
    nx = np.array([0, -np.sqrt(3) / 2 * st, np.sqrt(3) / 2 * st, 0, -np.sqrt(3) / 2 * st, np.sqrt(3) / 2 * st])
    ny = np.array([-st, 0.5 * st, 0.5 * st, -st, 0.5 * st, 0.5 * st])
    eul = []
    for i in range(6):
        al = np.arcsin(-ny[i])
        be = np.arcsin(np.clip(nx[i] / np.cos(al), -1, 1))
        eul.append(np.degrees([al, be, np.arctan2(pos[i, 1], pos[i, 0]) + np.pi / 2]))
    return pos, np.array(eul)


def axes_from_euler(eul):
    return np.array([rotation_xyz(*np.deg2rad(e)) for e in eul])


## ---------------------------------------------------------------------------------------------------------- repair
def _cyl(pos, R):
    return fcl.CollisionObject(fcl.Cylinder(AIR_R, AIR_LEN), fcl.Transform(R, pos))


_CAGE = fcl.CollisionObject(fcl.Box(*CAGE), fcl.Transform())


def _contact(a, b):
    """(penetration depth, normal from a to b) or None."""
    req = fcl.CollisionRequest(enable_contact=True, num_max_contacts=1)
    res = fcl.CollisionResult()
    fcl.collide(a, b, req, res)
    if not res.is_collision:
        return None
    c = res.contacts[0]
    return float(c.penetration_depth), np.asarray(c.normal, float)


def overlaps(pos, rots):
    """List of (i, j, depth, normal i->j) for overlapping airflow cylinders, j = -1 for the cage."""
    cyl = [_cyl(p, R) for p, R in zip(pos, rots)]
    out = []
    for i in range(N_MOTORS):
        for j in range(i + 1, N_MOTORS):
            c = _contact(cyl[i], cyl[j])
            if c is not None:
                out.append((i, j, *c))
        c = _contact(_CAGE, cyl[i])
        if c is not None:
            out.append((i, -1, c[0], -c[1]))  # normal from the motor towards the cage
    return out


def repair(pos, rots, max_iter=400, margin=2e-4):
    """Push overlapping motors apart (orientations fixed, mirror symmetry kept) until no airflow cylinder overlaps
    another one or the cage. Returns (repaired positions, total translation [m]) or (None, nan) if it fails."""
    p0, p = pos.copy(), pos.copy()
    for _ in range(max_iter):
        ov = overlaps(p, rots)
        if not ov:
            return p, float(np.linalg.norm(p - p0, axis=1).sum())
        delta = np.zeros_like(p)
        for i, j, depth, n in ov:
            step = (depth + margin) * n
            if j < 0:
                delta[i] -= step
            else:
                delta[i] -= step / 2
                delta[j] += step / 2
        for k in range(3):  # keep the mirror symmetry
            d = (delta[k] + mirror(delta[k + 3])) / 2
            if np.linalg.norm(d) < 1e-9:  # a mirror pair pushed apart sideways: the halves cancel, move along y
                d = delta[k] * np.array([0.0, 1.0, 0.0])
            delta[k], delta[k + 3] = d, mirror(d)
        p = p + delta
    return None, float("nan")


## ----------------------------------------------------------------------------------------------- mass and inertia
def mass_properties(pos):
    """Total mass, centre of mass (3,) and inertia about it (3, 3) of the cage, motors and arms."""
    bx, by = 0.9 * CAGE[0] / 2, 0.9 * CAGE[1] / 2
    items = [(M_BASE, np.zeros(3), I_BASE)]
    arms = []
    for p in pos:
        attach = np.array([np.clip(p[0], -bx, bx), by if p[1] > 0 else -by, 0.0])
        v = p - attach
        L = np.linalg.norm(v)
        m = ARM_KG_PER_M * L
        u = v / L
        i_rod = m * L ** 2 / 12 * (np.eye(3) - np.outer(u, u))  # thin rod about its centre
        items.append((m, attach + v / 2, i_rod))
        items.append((M_MOTOR, p, np.zeros((3, 3))))
        arms.append((attach, L))
    mass = sum(m for m, _, _ in items)
    com = sum(m * c for m, c, _ in items) / mass
    inertia = np.zeros((3, 3))
    for m, c, i_loc in items:
        r = c - com
        inertia += i_loc + m * (r @ r * np.eye(3) - np.outer(r, r))
    return mass, com, 0.5 * (inertia + inertia.T), arms


def allocation(pos, axes, com, directions):
    """Body wrench about the centre of mass per unit thrust of each motor: (6, 6), rows force xyz, torque xyz."""
    torque = np.cross(pos - com, axes) + (directions * CQ)[:, None] * axes
    return np.concatenate([axes.T, torque.T], 0)


def hover_analysis(A, mass):
    """Can the design hover, and with which thrusts? The thrusts with zero torque form a convex polytope; the forces
    it makes form a convex set, which contains a force as large as the weight (in some direction) if and only if its
    largest force (found on the vertices) is at least the weight and its smallest force at most the weight. The
    hover thrusts are upright if the design can hover upright (least total thrust), else on the segment between the
    smallest and the largest force. Returns (max force [N], hover thrusts (6,) or None, hover tilt [deg])."""
    F, T = A[:3], A[3:]
    best, best_u = 0.0, None
    for free in itertools.combinations(range(N_MOTORS), 3):
        fixed = [k for k in range(N_MOTORS) if k not in free]
        Tf = T[:, list(free)]
        if abs(np.linalg.det(Tf)) < 1e-12:
            continue
        Tf_inv = np.linalg.inv(Tf)
        for bounds in itertools.product((T_MIN, T_MAX), repeat=3):
            u = np.zeros(N_MOTORS)
            u[fixed] = bounds
            u[list(free)] = -Tf_inv @ (T[:, fixed] @ np.array(bounds))
            if np.all(u >= T_MIN - 1e-9) and np.all(u <= T_MAX + 1e-9):
                f = np.linalg.norm(F @ u)
                if f > best:
                    best, best_u = f, u
    weight = mass * G
    if best < weight:
        return best, None, float("nan")
    from scipy.linalg import null_space
    from scipy.optimize import linprog, minimize
    ## upright: thrusts within limits, zero torque, force = weight along body z; the most even thrusts (least largest
    ## thrust, a linear program: variables u (6) and t, minimize t with u <= t)
    up = np.array([0, 0, weight, 0, 0, 0])
    res = linprog(np.r_[np.zeros(N_MOTORS), 1.0], A_ub=np.c_[np.eye(N_MOTORS), -np.ones(N_MOTORS)],
                  b_ub=np.zeros(N_MOTORS), A_eq=np.c_[A, np.zeros(6)], b_eq=up,
                  bounds=[(T_MIN, T_MAX)] * N_MOTORS + [(None, None)], method="highs")
    if res.status == 0:
        return best, res.x[:N_MOTORS], 0.0
    ## smallest zero-torque force: u = best_u + N z (N spans the thrusts that make no torque), within the limits
    N = null_space(T)
    Gz = np.concatenate([N, -N], 0)
    h = np.concatenate([T_MAX - best_u, best_u - T_MIN])
    res = minimize(lambda z: np.sum((F @ (best_u + N @ z)) ** 2), np.zeros(N.shape[1]),
                   jac=lambda z: 2 * N.T @ F.T @ (F @ (best_u + N @ z)), method="SLSQP",
                   constraints=[{"type": "ineq", "fun": lambda z: h - Gz @ z, "jac": lambda z: -Gz}],
                   options={"ftol": 1e-12, "maxiter": 1000})
    assert res.success, res
    u_lo = np.clip(best_u + N @ res.x, T_MIN, T_MAX)
    if np.linalg.norm(F @ u_lo) > weight:
        return best, None, float("nan")
    lo, hi = 0.0, 1.0
    for _ in range(60):
        s = (lo + hi) / 2
        lo, hi = (s, hi) if np.linalg.norm(F @ (u_lo + s * (best_u - u_lo))) < weight else (lo, s)
    u = u_lo + lo * (best_u - u_lo)
    f = F @ u
    return best, u, float(np.degrees(np.arccos(np.clip(f[2] / np.linalg.norm(f), -1, 1))))


def physical(d):
    """Design dict -> everything the simulator, the policy and the renderer need (numpy, float64)."""
    if "name" in d:  # a baseline: arms of PLANAR_ARM, as published (no repair, their airflow may overlap)
        directions = DIRECTIONS_BASELINES
        pos, eul = motors_baseline(d["name"], PLANAR_ARM)
        rots = axes_from_euler(eul)
        pos_r, moved = pos, 0.0
    else:
        directions = DIRECTIONS
        pos, eul = motors_from_x(d["x"])
        rots = axes_from_euler(eul)
        pos_r, moved = repair(pos, rots)
    axes = rots[:, :, 2]
    ph = {"design": d, "euler_deg": eul, "rotations": rots, "axes": axes, "pos_unrepaired": pos,
          "repair_m": moved, "repair_ok": pos_r is not None, "directions": directions}
    if pos_r is None:
        ph.update(feasible=False, reason="repair failed")
        return ph
    mass, com, inertia, arms = mass_properties(pos_r)
    A = allocation(pos_r, axes, com, directions)
    f_max, u_hover, tilt = hover_analysis(A, mass)
    ph.update(pos=pos_r, mass=mass, com=com, inertia=inertia, arms=arms, alloc=A, max_zero_torque_force=f_max,
              thrust_to_weight=f_max / (mass * G), hover_thrust=u_hover, hover_tilt_deg=tilt,
              feasible=u_hover is not None, reason="" if u_hover is not None else "cannot hover")
    return ph


def random_x(rng):
    return rng.uniform(0, 1, N_X)


def baseline_physical():
    return physical(BASELINE)


def label(d):
    return d["name"] if "name" in d else "evolved"
