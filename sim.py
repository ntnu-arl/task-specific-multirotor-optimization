"""Task B of the paper ("forward flight with sudden avoidance"), or task A ("forward flight with smooth steering", with
the environment variable TASK=A), for many hexarotors at once, simulated with Newton on the GPU.

Task: fly through square gates (0.5 x 0.5 m) for 5 s. The first gate is at the origin and the drone starts 0.5 to 1
m before it (random position, yaw, velocity and body rates). Each time the drone crosses the plane of its gate, the
next gate appears 0.25 m further in x; with probability 0.05 it also jumps 0.5 to 0.7 m to the left or right and up
to 0.1 m up or down, so the drone must be able to dodge sideways within 0.25 m. The drone only sees its next gate.
In task A the next gate is always 0.5 m further in x and up to 0.25 m to the left or right (at the same height).
    score of an episode = gates crossed - 10 x gates missed (crossing the plane outside the square) - 10 if it crashes
A crash (which ends the episode) is flying more than 2.25 m further from the gate than it has been, or flipping
(body z more than 120 degrees from vertical).

Simulation: every hexarotor is a free rigid body of its own (they do not interact), all D designs x M copies in one
Newton model. Per physics step (5 ms) a warp kernel updates the rotor speeds (first-order lag towards the command),
and sets the body wrench (the thrust of each motor along its axis, its moment about the centre of mass and the
propeller drag torque); Newton integrates the bodies. Per control step (10 ms, two physics steps) a second kernel
checks the gates, computes the reward and the score, ends and restarts episodes and writes the observation. A whole
control step is one CUDA graph; the policy reads the observations and writes the commands in place (torch tensors
that share the memory of the warp arrays), so nothing goes through the CPU.

Observation (21): gate position relative to the drone (world frame), first two rows of the rotation matrix, velocity
(world), body rates (body frame), the six rotor thrusts scaled to [-1, 1].
Action (6): rotor commands in [-1, 1] (minimum to maximum thrust).
Reward (training only): the episode score as it happens (+1 per gate, -10 per missed gate, -10 for a crash) with
the miss penalty scaled by w_miss, plus w_progress x the distance flown towards the current gate in the step, minus
a small penalty on body rates above 10 rad/s. rl.py anneals the weights during training (w_miss from 0.1 to 1,
w_progress from 2 per metre to 0): first fly towards the gates, then learn not to miss them, and at the end optimize
the score itself. The crash penalty is always the full one: if crashing were cheaper than missing gates early on,
the policy could learn to crash on purpose and stay stuck there.
"""
import numpy as np
import torch
import warp as wp

import design
import newton

wp.config.quiet = True
CTRL_DT, SUBSTEPS = 0.01, 2
PHYS_DT = CTRL_DT / SUBSTEPS
EPISODE_STEPS = 500  # 5 s
OBS = 21
ACT = design.N_MOTORS
GATE_D, GATE_HALF, GATE_PROB = {"A": 0.5, "B": 0.25}[design.TASK], 0.25, 0.05
GATE_Y_MIN, GATE_Y_MAX, GATE_Z = 0.5, 0.7, 0.1  # task B
GATE_Y_A = 0.25  # task A
CRASH_DIST = GATE_D + 2.0
CRASH_COS_TILT = -0.5
OMEGA_FREE, K_OMEGA = 10.0, 0.01  # body rates above OMEGA_FREE [rad/s] cost K_OMEGA per rad/s and step


@wp.func
def sample_next_gate_a(g: wp.vec3, rng: wp.uint32):
    y = -GATE_Y_A + 2.0 * GATE_Y_A * wp.randf(rng)
    return wp.vec3(g[0] + GATE_D, g[1] + y, g[2]), rng


@wp.func
def sample_next_gate_b(g: wp.vec3, rng: wp.uint32):
    p = wp.randf(rng)
    y_mag = GATE_Y_MIN + wp.randf(rng) * (GATE_Y_MAX - GATE_Y_MIN)
    side = wp.where(wp.randf(rng) < 0.5, -1.0, 1.0)
    z = -GATE_Z + 2.0 * GATE_Z * wp.randf(rng)
    if p < GATE_PROB:
        return wp.vec3(g[0] + GATE_D, g[1] + side * y_mag, g[2] + z), rng
    return wp.vec3(g[0] + GATE_D, g[1], g[2]), rng


sample_next_gate = {"A": sample_next_gate_a, "B": sample_next_gate_b}[design.TASK]


@wp.kernel
def rotor_kernel(body_q: wp.array(dtype=wp.transform), cmd: wp.array2d(dtype=float), rps: wp.array2d(dtype=float),
                 mpos: wp.array2d(dtype=wp.vec3), maxis: wp.array2d(dtype=wp.vec3), mdir: wp.array2d(dtype=float),
                 copies: int, alpha: float, body_f: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    d = i // copies
    q = wp.transform_get_rotation(body_q[i])
    f = wp.vec3(0.0)
    tau = wp.vec3(0.0)
    for k in range(ACT):
        u = 0.5 * (wp.clamp(cmd[i, k], -1.0, 1.0) + 1.0)
        target = wp.sqrt((design.T_MIN + u * (design.T_MAX - design.T_MIN)) / design.KF)
        w = rps[i, k] + alpha * (target - rps[i, k])
        rps[i, k] = w
        t = design.KF * w * w
        a = maxis[d, k]
        f += t * a
        tau += wp.cross(mpos[d, k], t * a) + mdir[d, k] * design.CQ * t * a
    body_f[i] = wp.spatial_vector(wp.quat_rotate(q, f), wp.quat_rotate(q, tau))


@wp.func
def write_obs(i: int, x: wp.vec3, q: wp.quat, v: wp.vec3, w: wp.vec3, gate: wp.vec3, rps: wp.array2d(dtype=float),
              obs: wp.array2d(dtype=float)):
    R = wp.quat_to_matrix(q)
    e = gate - x
    wb = wp.quat_rotate_inv(q, w)
    for k in range(3):
        obs[i, k] = e[k]
        obs[i, 3 + k] = R[0, k]
        obs[i, 6 + k] = R[1, k]
        obs[i, 9 + k] = v[k]
        obs[i, 12 + k] = wb[k]
    for k in range(ACT):
        obs[i, 15 + k] = 2.0 * (design.KF * rps[i, k] * rps[i, k] - design.T_MIN) / (design.T_MAX - design.T_MIN) - 1.0


@wp.kernel
def task_kernel(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
                body_com: wp.array(dtype=wp.vec3), rps: wp.array2d(dtype=float), gate: wp.array(dtype=wp.vec3),
                prev_x: wp.array(dtype=wp.vec3), d_best: wp.array(dtype=float), steps: wp.array(dtype=int),
                rng_state: wp.array(dtype=wp.uint32), training: int, weights: wp.array(dtype=float),
                active: wp.array(dtype=int), score: wp.array(dtype=float), gates_passed: wp.array(dtype=int),
                gates_missed: wp.array(dtype=int), crashed: wp.array(dtype=int),
                ep_sum: wp.array(dtype=float), ep_count: wp.array(dtype=int),
                reward: wp.array(dtype=float), done: wp.array(dtype=float), timeout: wp.array(dtype=float),
                obs_end: wp.array2d(dtype=float), obs: wp.array2d(dtype=float)):
    i = wp.tid()
    reward[i] = 0.0
    done[i] = 0.0
    timeout[i] = 0.0
    if active[i] == 0:  # evaluation: this episode has ended, the drone is frozen
        return
    tf = body_q[i]
    x = wp.transform_get_translation(tf)
    q = wp.transform_get_rotation(tf)
    v = wp.spatial_top(body_qd[i])
    w = wp.spatial_bottom(body_qd[i])
    x0 = prev_x[i]
    g = gate[i]
    progress = wp.length(x0 - g) - wp.length(x - g)
    s = float(0.0)
    miss = float(0.0)
    rng = rng_state[i]
    ## gate: the drone crossed its plane (normal +x) between the last control step and this one
    if (g[0] - x0[0]) * (g[0] - x[0]) <= 0.0 and x[0] != x0[0]:
        c = x0 + (x - x0) * ((g[0] - x0[0]) / (x[0] - x0[0]))
        if wp.max(wp.abs(c[1] - g[1]), wp.abs(c[2] - g[2])) <= GATE_HALF:
            s += 1.0
            gates_passed[i] = gates_passed[i] + 1
        else:
            s -= 10.0
            miss = 10.0
            gates_missed[i] = gates_missed[i] + 1
        g, rng = sample_next_gate(g, rng)
        gate[i] = g
        d_best[i] = wp.length(x - g)
    d = wp.length(x - g)
    d_best[i] = wp.min(d_best[i], d)
    R = wp.quat_to_matrix(q)
    crash = d - d_best[i] > CRASH_DIST or R[2, 2] < CRASH_COS_TILT or not (wp.isfinite(x) and wp.isfinite(v))
    n = steps[i] + 1
    steps[i] = n
    if crash:
        s -= 10.0
        crashed[i] = crashed[i] + 1
    r = s + (1.0 - weights[1]) * miss + weights[0] * progress - K_OMEGA * wp.max(wp.length(w) - OMEGA_FREE, 0.0)
    total = score[i] + s
    score[i] = total
    prev_x[i] = x
    reward[i] = wp.where(wp.isfinite(r), r, -10.0)
    write_obs(i, x, q, v, w, g, rps, obs_end)
    end = crash or n >= EPISODE_STEPS
    if not end:
        write_obs(i, x, q, v, w, g, rps, obs)
        rng_state[i] = rng
        return
    done[i] = 1.0
    timeout[i] = wp.where(crash, 0.0, 1.0)
    ep_sum[i] = ep_sum[i] + total
    ep_count[i] = ep_count[i] + 1
    if training == 0:
        active[i] = 0
        rng_state[i] = rng
        return
    ## new episode (training): random start before the first gate at the origin
    score[i] = 0.0
    steps[i] = 0
    x = wp.vec3(-1.0 + 0.5 * wp.randf(rng), -0.5 + wp.randf(rng), -0.5 + wp.randf(rng))
    yaw = (wp.randf(rng) - 0.5) * wp.pi / 3.0
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    v = wp.vec3(0.4 * wp.randf(rng) - 0.2, 0.4 * wp.randf(rng) - 0.2, 0.4 * wp.randf(rng) - 0.2)
    w = wp.vec3(0.4 * wp.randf(rng) - 0.2, 0.4 * wp.randf(rng) - 0.2, 0.4 * wp.randf(rng) - 0.2)
    for k in range(ACT):
        u = 0.4 + 0.2 * wp.randf(rng)
        rps[i, k] = wp.sqrt((design.T_MIN + u * (design.T_MAX - design.T_MIN)) / design.KF)
    rng_state[i] = rng
    com_w = wp.quat_rotate(q, body_com[i])
    body_q[i] = wp.transform(x, q)
    body_qd[i] = wp.spatial_vector(v + wp.cross(w, com_w), w)
    g = wp.vec3(0.0, 0.0, 0.0)
    gate[i] = g
    prev_x[i] = x
    d_best[i] = wp.length(x - g)
    write_obs(i, x, q, v, w, g, rps, obs)


@wp.kernel
def start_kernel(seed: int, copies: int, common: int, body_q: wp.array(dtype=wp.transform),
                 body_qd: wp.array(dtype=wp.spatial_vector), body_com: wp.array(dtype=wp.vec3),
                 rps: wp.array2d(dtype=float), gate: wp.array(dtype=wp.vec3), prev_x: wp.array(dtype=wp.vec3),
                 d_best: wp.array(dtype=float), steps: wp.array(dtype=int), rng_state: wp.array(dtype=wp.uint32),
                 active: wp.array(dtype=int), score: wp.array(dtype=float), obs: wp.array2d(dtype=float)):
    """First episode of every drone. common=1: copy m of every design gets the same random numbers (same start and
    the same gates for every design, so their scores are compared on the same episodes)."""
    i = wp.tid()
    m = wp.where(common == 1, i % copies, i)
    rng = wp.rand_init(seed, m)
    x = wp.vec3(-1.0 + 0.5 * wp.randf(rng), -0.5 + wp.randf(rng), -0.5 + wp.randf(rng))
    yaw = (wp.randf(rng) - 0.5) * wp.pi / 3.0
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    v = wp.vec3(0.4 * wp.randf(rng) - 0.2, 0.4 * wp.randf(rng) - 0.2, 0.4 * wp.randf(rng) - 0.2)
    w = wp.vec3(0.4 * wp.randf(rng) - 0.2, 0.4 * wp.randf(rng) - 0.2, 0.4 * wp.randf(rng) - 0.2)
    for k in range(ACT):
        u = 0.4 + 0.2 * wp.randf(rng)
        rps[i, k] = wp.sqrt((design.T_MIN + u * (design.T_MAX - design.T_MIN)) / design.KF)
    com_w = wp.quat_rotate(q, body_com[i])
    body_q[i] = wp.transform(x, q)
    body_qd[i] = wp.spatial_vector(v + wp.cross(w, com_w), w)
    g = wp.vec3(0.0, 0.0, 0.0)
    gate[i] = g
    prev_x[i] = x
    d_best[i] = wp.length(x - g)
    steps[i] = 0
    active[i] = 1
    score[i] = 0.0
    rng_state[i] = rng
    write_obs(i, x, q, v, w, g, rps, obs)


def use_stream(device):
    """Torch and warp share one (non-default) CUDA stream: their work is ordered without synchronizing, and a control
    step can be captured as a CUDA graph (not possible on the default stream)."""
    if not str(device).startswith("cuda") or getattr(use_stream, "done", False):
        return
    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)
    wp.set_device(device)
    wp.set_stream(wp.stream_from_torch(stream))
    use_stream.done = True


class Swarm:
    """D designs x M copies of a hexarotor flying the task. Torch views (sharing memory): obs (D, M, OBS), cmd
    (D, M, ACT), reward/done/timeout (D, M), obs_end (D, M, OBS) (the observation before a restart, for bootstrapping
    the value of a time-out)."""

    def __init__(self, n_designs, copies, device="cuda:0"):
        self.D, self.M, self.N = n_designs, copies, n_designs * copies
        use_stream(device)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-design.G)
        for _ in range(self.N):
            builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3) * 1e-3), lock_inertia=True)
        self.model = builder.finalize(device=device)
        self.device = self.model.device
        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        self.s0, self.s1 = self.model.state(), self.model.state()
        dev, N = self.device, self.N
        self.mpos = wp.zeros((n_designs, ACT), dtype=wp.vec3, device=dev)
        self.maxis = wp.zeros((n_designs, ACT), dtype=wp.vec3, device=dev)
        self.mdir = wp.zeros((n_designs, ACT), dtype=float, device=dev)
        z = lambda dtype, *shape: wp.zeros(shape if shape else N, dtype=dtype, device=dev)
        self.cmd, self.rps = z(float, N, ACT), z(float, N, ACT)
        self.gate, self.prev_x, self.d_best = z(wp.vec3), z(wp.vec3), z(float)
        self.steps, self.active, self.rng_state = z(int), z(int), z(wp.uint32)
        self.score, self.passed, self.missed, self.crashed = z(float), z(int), z(int), z(int)
        self.ep_sum, self.ep_count = z(float), z(int)
        self.reward, self.done, self.timeout = z(float), z(float), z(float)
        self.obs, self.obs_end = z(float, N, OBS), z(float, N, OBS)
        view = lambda a, *shape: wp.to_torch(a).view(n_designs, copies, *shape)
        self.t_obs, self.t_obs_end, self.t_cmd = view(self.obs, OBS), view(self.obs_end, OBS), view(self.cmd, ACT)
        self.t_reward, self.t_done, self.t_timeout = view(self.reward), view(self.done), view(self.timeout)
        self.weights = wp.array([0.0, 1.0], dtype=float, device=dev)  # w_progress, w_miss
        self.graphs = {}
        self.alpha = PHYS_DT / (PHYS_DT + design.TAU_MOTOR)

    def set_designs(self, phys):
        """phys: one design.physical() dict per design (feasible ones); fewer than D leaves the last slots as they are
        (their copies fly the last design again)."""
        assert 0 < len(phys) <= self.D
        phys = list(phys) + [phys[-1]] * (self.D - len(phys))
        mass = np.repeat([p["mass"] for p in phys], self.M).astype(np.float32)
        inertia = np.repeat(np.stack([p["inertia"] for p in phys]), self.M, 0).astype(np.float32)
        com = np.repeat(np.stack([p["com"] for p in phys]), self.M, 0).astype(np.float32)
        self.model.body_mass.assign(mass)
        self.model.body_inv_mass.assign(1.0 / mass)
        self.model.body_inertia.assign(inertia)
        self.model.body_inv_inertia.assign(np.linalg.inv(inertia.astype(np.float64)).astype(np.float32))
        self.model.body_com.assign(com)
        self.mpos.assign(np.stack([p["pos"] - p["com"] for p in phys]).astype(np.float32))
        self.maxis.assign(np.stack([p["axes"] for p in phys]).astype(np.float32))
        self.mdir.assign(np.stack([p["directions"] for p in phys]).astype(np.float32))

    def set_reward_weights(self, w_progress, w_miss):
        self.weights.assign(np.array([w_progress, w_miss], np.float32))

    def reset(self, seed, common):
        """Start the first episode of every drone (common: the same episodes for every design)."""
        wp.launch(start_kernel, self.N, inputs=[seed, self.M, int(common), self.s0.body_q, self.s0.body_qd,
                                                self.model.body_com, self.rps, self.gate, self.prev_x, self.d_best,
                                                self.steps, self.rng_state, self.active, self.score, self.obs],
                  device=self.device)
        for a in (self.passed, self.missed, self.crashed, self.ep_sum, self.ep_count):
            a.zero_()

    def _step(self, training):
        for _ in range(SUBSTEPS):
            wp.launch(rotor_kernel, self.N, inputs=[self.s0.body_q, self.cmd, self.rps, self.mpos, self.maxis,
                                                    self.mdir, self.M, self.alpha, self.s0.body_f],
                      device=self.device)
            self.solver.integrate_bodies(self.model, self.s0, self.s1, PHYS_DT, 0.0)
            self.s0, self.s1 = self.s1, self.s0
        wp.launch(task_kernel, self.N, inputs=[self.s0.body_q, self.s0.body_qd, self.model.body_com, self.rps,
                                               self.gate, self.prev_x, self.d_best, self.steps, self.rng_state,
                                               int(training), self.weights, self.active, self.score, self.passed, self.missed,
                                               self.crashed, self.ep_sum, self.ep_count, self.reward, self.done,
                                               self.timeout, self.obs_end, self.obs], device=self.device)

    def step(self, training):
        """One control step with the commands in cmd (an even number of physics steps: the state buffers end where
        they began, so the CUDA graph can be replayed)."""
        if not self.device.is_cuda:
            self._step(training)
            return
        if training not in self.graphs:
            with wp.ScopedCapture(device=self.device) as cap:
                self._step(training)
            self.graphs[training] = cap.graph
        wp.capture_launch(self.graphs[training])

    def pose(self):
        """Body position (N, 3) and wxyz quaternion (N, 4) (numpy), for the videos."""
        q = self.s0.body_q.numpy()
        return q[:, :3].copy(), np.concatenate([q[:, 6:7], q[:, 3:6]], 1)

    def episode_stats(self):
        """Per design: summed episode scores, episode count, gates passed, missed, crashes (numpy, (D,))."""
        r = lambda a: a.numpy().reshape(self.D, self.M).sum(1)
        return {k: r(a) for k, a in (("score_sum", self.ep_sum), ("episodes", self.ep_count), ("passed", self.passed),
                                      ("missed", self.missed), ("crashed", self.crashed))}

    def thrust(self):
        return design.KF * self.rps.numpy() ** 2
