"""Offline video of the evaluated design and the baselines with Newton's GPU ray tracer (headless, no display needed).

All drones fly the same EPISODES episodes of task B (same start, same gates), each with its own learned policy, one
on top of the other in the same place (they are replayed, not simulated: they fly through each other and through
the floor). The episodes shown are ones with a sudden sideways gate. Slow motion: SLOWMO x slower than real time. A
drone that crashes falls through the floor; a drone that misses a gate flies on.
Look: a pitch black world over a grey and black checkered floor, transparent red glass gates; every airframe has a
white cage (the evaluated design: green), orange arms and pastel motors (blue or yellow by spin direction). Each drone
carries a label with its name; the numbers are in the PDF reports (report.py).
The ray tracer only finds the closest surface, so the glass is composited: the gates are traced in a model of their
own (depth and normals) and blended over the rest of the scene with a Fresnel rim and specular highlights.
still() renders a close-up picture of a single design in its hover attitude for the reports.
"""
import dataclasses
import subprocess
from pathlib import Path

import cv2
import numpy as np
import warp as wp
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

import design
import newton
import report
import sim
import warmstart
from newton.sensors import SensorTiledCamera

W, H, FPS = 1280, 720, 30  # output size, frame rate
SS = 2  # supersampling of the video (antialiasing)
SLOWMO = 2.0  # video seconds per flight second
SUBFRAMES = 2  # renders averaged per frame (antialiasing and a little motion blur)
FLOOR_Z = -0.75  # the gates are around z = 0, the drones start within 0.5 m of it
TILE = 0.5  # checkered floor [m]
EPISODES = 3
HFOV = np.deg2rad(55)
BLACK = 0xFF000000  # packed ABGR
CAGE_COLOR, CAGE_HIGHLIGHT = (0.95, 0.95, 0.95), (0.15, 0.95, 0.2)  # RGB in [0, 1]; the evaluated design is green
ARM_COLOR = (1.0, 0.423, 0.039)
MOTOR_COLORS = {1.0: (0.5, 0.5, 1.0), -1.0: (1.0, 1.0, 0.5)}  # pastel blue / yellow by spin direction
FLOOR_COLORS = ((0.36, 0.36, 0.38), (0.03, 0.03, 0.03))
GATE_TINT = np.array([0.95, 0.08, 0.1], np.float32)  # RGB
GATE_R = 0.018  # radius of the glass tubes of a gate [m]
LIGHT = np.array([1.0, -1.0, 1.0], np.float32) / np.sqrt(3)  # towards the ray tracer's default light
LABEL_COLORS = {True: (110, 245, 90), False: (235, 235, 235)}  # RGB, evaluated design / baselines
LABEL_FONT = ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family=report.SERIF)), 22)
LABEL_UP = 38  # height of the labels above their drones [px]
HIDDEN = (-1e4, 0.0, 0.0)  # where the gates the drones have left behind go (out of the camera's range)


def look_at(eye, target):
    """Camera-to-world rotation for the ray tracer (camera x right, y up, looking along -z)."""
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    return np.stack([r, np.cross(r, f), -f], 1)


def rot_to_quat_xyzw(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-3:
        return np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w])
    k = int(np.argmax(np.diag(R)))
    i, j = (k + 1) % 3, (k + 2) % 3
    s = np.sqrt(max(1e-12, 1 + R[k, k] - R[i, i] - R[j, j])) * 2
    q = np.zeros(4)
    q[k], q[i], q[j], q[3] = s / 4, (R[i, k] + R[k, i]) / s, (R[j, k] + R[k, j]) / s, (R[j, i] - R[i, j]) / s
    return q


class Camera:
    """Pinhole camera on a Newton model and, optionally, on a model of glass objects composited over it
    (glass()). render() returns a BGR image."""

    def __init__(self, model, glass_model=None, hfov=HFOV, ss=1):
        self.ss, self.hfov = ss, hfov
        vfov = 2 * np.arctan(np.tan(hfov / 2) * H / W)
        self.tracers = []
        for m in (model, glass_model):
            if m is None:
                continue
            cam = SensorTiledCamera(model=m)
            cam.default_render_config.enable_shadows = True
            cam.utils.create_default_light(enable_shadows=True)
            self.rays = cam.utils.compute_camera_rays_pinhole(W * ss, H * ss, camera_fovs=float(vfov))
            self.tracers.append((m, cam, cam.utils.create_color_image_output(W * ss, H * ss, 1),
                                 cam.utils.create_depth_image_output(W * ss, H * ss, 1),
                                 cam.utils.create_normal_image_output(W * ss, H * ss, 1)))
        self.dirs = self.rays.numpy()[0, :, :, 1]  # camera-space ray directions (H ss, W ss, 3)
        self.clear = dataclasses.replace(SensorTiledCamera.GRAY_CLEAR_DATA, clear_color=BLACK)

    def render(self, states, eye, R):
        """states: one per model. Returns the BGR image (H, W, 3) as float32 in [0, 255]."""
        eye = np.asarray(eye, float)
        tf = wp.array([[wp.transformf(wp.vec3f(*eye), wp.quatf(*rot_to_quat_xyzw(R)))]], dtype=wp.transformf)
        out = []
        for (model, cam, color, depth, normal), state in zip(self.tracers, states):
            model.bvh_refit_shapes(state)
            cam.update(state, tf, self.rays, color_image=color, depth_image=depth, normal_image=normal,
                       clear_data=self.clear)
            img = color.numpy()[0, 0].view(np.uint8).reshape(H * self.ss, W * self.ss, 4)[..., :3]
            out.append((img.astype(np.float32) / 255, depth.numpy()[0, 0], normal.numpy()[0, 0]))
        img = out[0][0] if len(out) == 1 else glass(out[0], out[1], (self.dirs @ R.T).astype(np.float32))
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) if self.ss > 1 else img
        return np.ascontiguousarray(255 * np.clip(img[..., ::-1], 0, 1))

    def project(self, p, eye, R):
        """Pixel (u, v) of a world point in the output image, or None if it is behind the camera."""
        c = R.T @ (np.asarray(p, float) - eye)
        if c[2] > -1e-3:
            return None
        f = W / 2 / np.tan(self.hfov / 2)
        return W / 2 + f * c[0] / -c[2], H / 2 - f * c[1] / -c[2]


def glass(scene, gates, dirs):
    """The gates (colour, hit distance, normal of their own model) over the scene as red glass: what is behind is
    seen tinted red, more opaque and more reflective at grazing angles (Fresnel), with highlights of the light and a
    faint reflection of the sky. dirs: world ray directions (H, W, 3)."""
    color, depth, _ = scene
    _, g_depth, n = gates
    m = (g_depth > 0) & ((depth <= 0) | (g_depth < depth))
    if not m.any():
        return color
    fill = -dirs[dirs.shape[0] // 2, dirs.shape[1] // 2] + np.array([0, 0, 0.25], np.float32)  # from behind the camera
    fill /= np.linalg.norm(fill)
    c, n, d = color[m], n[m], dirs[m]
    cos = np.abs(np.sum(n * d, -1, keepdims=True))
    n = np.where(np.sum(n * d, -1, keepdims=True) > 0, -n, n)  # facing the camera
    fresnel = 0.04 + 0.96 * (1 - cos) ** 5
    alpha = 0.32 + 0.5 * (1 - cos) ** 2
    refl = d - 2 * np.sum(d * n, -1, keepdims=True) * n
    spec = (0.8 * np.clip(refl @ LIGHT, 0, 1) ** 40 + 0.25 * np.clip(refl @ fill, 0, 1) ** 12)[:, None] \
        + fresnel * 0.3 * np.clip(refl[:, 2:3], 0, 1) ** 2
    body = GATE_TINT * (0.7 + 0.5 * np.clip(n @ LIGHT, 0, 1)[:, None] + 0.5 * (1 - cos) ** 3)  # glowing rim
    behind = c * (1 - 0.6 * (1 - GATE_TINT))  # seen through red glass
    out = color.copy()
    out[m] = behind * (1 - alpha) + body * alpha + spec
    return out


def _xf(p, R):
    return wp.transform(wp.vec3(*map(float, p)), wp.quat(*map(float, rot_to_quat_xyzw(R))))


def _no_collisions(builder):
    cfg = builder.ShapeConfig()
    cfg.has_shape_collision = cfg.has_particle_collision = False
    return cfg


def add_drone(builder, ph, highlight=False):
    """The airframe as designed: cage, carbon arms from the side of the cage to the motors, motors along their thrust
    axes and propeller discs (coloured by spin direction)."""
    body = builder.add_body(mass=1.0)
    cfg = _no_collisions(builder)
    builder.add_shape_box(body, hx=design.CAGE[0] / 2, hy=design.CAGE[1] / 2, hz=design.CAGE[2] / 2, cfg=cfg,
                          color=CAGE_HIGHLIGHT if highlight else CAGE_COLOR)
    for p, R, (attach, L), s in zip(ph["pos"], ph["rotations"], ph["arms"], ph["directions"]):
        u = (p - attach) / L
        a = np.cross([0, 0, 1.0], u)
        Ra = np.eye(3) if np.linalg.norm(a) < 1e-9 else _axis_angle(a / np.linalg.norm(a), np.arccos(np.clip(u[2], -1, 1)))
        builder.add_shape_cylinder(body, xform=_xf((attach + p) / 2, Ra), radius=0.006, half_height=L / 2, cfg=cfg,
                                   color=ARM_COLOR)
        motor = MOTOR_COLORS[float(np.sign(s))]
        builder.add_shape_cylinder(body, xform=_xf(p, R), radius=0.012, half_height=0.012, cfg=cfg, color=motor)
        builder.add_shape_cylinder(body, xform=_xf(p + 0.016 * R[:, 2], R), radius=design.PROP_D / 2,
                                   half_height=0.0015, cfg=cfg, color=motor)
    return body


def _axis_angle(axis, angle):
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def add_gate(builder):
    """A square gate (inner size 0.5 x 0.5 m) of glass tubes in the y-z plane, as a body of its own (moved every
    frame: gates the drones have left behind are hidden)."""
    body = builder.add_body(mass=1.0)
    cfg = _no_collisions(builder)
    h = sim.GATE_HALF + GATE_R
    along_y = _axis_angle(np.array([1.0, 0, 0]), np.pi / 2)  # the capsule's axis (z) along y
    for dy, dz, R in ((0, h, along_y), (0, -h, along_y), (h, 0, np.eye(3)), (-h, 0, np.eye(3))):
        builder.add_shape_capsule(body, xform=_xf((0.0, dy, dz), R), radius=GATE_R, half_height=h, cfg=cfg,
                                  color=tuple(GATE_TINT))
    return body


def add_floor(builder, x0, x1, y0, y1, z=FLOOR_Z):
    """Checkered floor: a black slab with grey tiles on it."""
    cfg = _no_collisions(builder)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(cx, cy, z - 0.02), wp.quat_identity()), hx=(x1 - x0) / 2,
                          hy=(y1 - y0) / 2, hz=0.02, cfg=cfg, color=FLOOR_COLORS[1])
    for i in range(int(np.floor(x0 / TILE)), int(np.ceil(x1 / TILE))):
        for j in range(int(np.floor(y0 / TILE)), int(np.ceil(y1 / TILE))):
            if (i + j) % 2 == 0:
                builder.add_shape_box(-1, xform=wp.transform(wp.vec3((i + 0.5) * TILE, (j + 0.5) * TILE, z + 0.001),
                                                             wp.quat_identity()),
                                      hx=TILE / 2, hy=TILE / 2, hz=0.001, cfg=cfg, color=FLOOR_COLORS[0])


def gate_sequence(gates):
    """Distinct consecutive targets of one drone over an episode: (n, 3)."""
    keep = np.concatenate([[True], np.any(np.abs(np.diff(gates, axis=0)) > 1e-6, axis=1)])
    return gates[keep]


def pick_episodes(rec, n):
    """Episodes (indices of the recorded copies) with a sudden sideways gate early enough to be seen, best first."""
    out = []
    for c in range(rec["pos"].shape[2]):
        g = gate_sequence(rec["gate"][:, 0, c])
        jumps = np.nonzero(np.abs(np.diff(g[:, 1])) > 0.3)[0]
        if len(jumps) and jumps[0] < 12:
            out.append(c)
    rest = [c for c in range(rec["pos"].shape[2]) if c not in out]
    return (out + rest)[:n]


def falling(pos, quat, active):
    """A drone whose episode ended by a crash falls from where it was (ballistic, from rest), through the floor."""
    pos, quat = pos.copy(), quat.copy()
    ended = np.nonzero(active == 0)[0]
    if len(ended):
        k0 = ended[0]
        t = (np.arange(len(pos)) - k0) * sim.CTRL_DT
        drop = np.where(t > 0, 0.5 * design.G * t ** 2, 0.0)
        pos[k0:, 2] = pos[k0, 2] - drop[k0:]
        quat[k0:] = quat[k0]
    return pos, quat


def draw_labels(img, points, labels, highlight):
    """Names (in LABEL_FONT) LABEL_UP pixels above the drones (pixel positions, None: not shown), with a thin line down
    to their drone. img: BGR, drawn in place. The labels stay at the same height over their drone even if they
    overlap (they do not jump around)."""
    pil = Image.fromarray(img[:, :, ::-1])
    draw = ImageDraw.Draw(pil)
    for k, p in enumerate(points):
        if p is None:
            continue
        u, v = p
        x0, y0, x1, y1 = draw.textbbox((0, 0), labels[k], font=LABEL_FONT, anchor="ms")
        x = min(max(u, 4 - x0), W - 4 - x1)
        y = max(v - LABEL_UP, 4 - y0)
        draw.line([(u, v - 8), (u, y + y1 + 2)], fill=(150, 150, 150), width=1)
        draw.text((x, y), labels[k], font=LABEL_FONT, anchor="ms", fill=LABEL_COLORS[k == highlight],
                  stroke_width=2, stroke_fill=(0, 0, 0))
    img[:] = np.asarray(pil)[:, :, ::-1]


def render(phys, labels, rec, crashed, out, highlight=0):
    """Comparison video. phys: the designs (physical()); labels: their names; rec: rl.evaluate(..., record=K)
    recording of all of them; crashed: (designs, K) booleans (the episode ended by a crash); highlight: the index of
    the evaluated design (green cage, green label), None for none."""
    out = Path(out)
    episodes = pick_episodes(rec, EPISODES)
    tmp = out.with_suffix(".tmp.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(tmp)],
                          stdin=subprocess.PIPE)
    n = len(phys)
    for c in episodes:
        builder, gate_builder = (newton.ModelBuilder(up_axis=newton.Axis.Z) for _ in range(2))
        tracks = []
        for i in range(n):
            p, q = rec["pos"][:, i, c], rec["quat"][:, i, c]
            if crashed[i][c]:
                p, q = falling(p, q, rec["active"][:, i, c])
            tracks.append((p, q))
        seq = max((gate_sequence(rec["gate"][:, i, c]) for i in range(n)), key=len)
        x_end = max(p[p[:, 2] > FLOOR_Z, 0].max() for p, _ in tracks) + 1.0
        add_floor(builder, -3, x_end + 5, -4, 4)
        bodies = [add_drone(builder, ph, i == highlight) for i, ph in enumerate(phys)]
        gates = [(add_gate(gate_builder), g) for g in seq if g[0] <= x_end + 0.5]
        model, gate_model = builder.finalize(), gate_builder.finalize()
        state, gate_state = model.state(), gate_model.state()
        cam = Camera(model, gate_model, ss=SS)
        body_q, gate_q = state.body_q.numpy(), gate_state.body_q.numpy()
        n_steps = len(tracks[0][0])
        n_frames = int(sim.EPISODE_STEPS * sim.CTRL_DT * SLOWMO * FPS)
        cam_x = cam_y = behind = None
        for f in range(n_frames):
            acc = np.zeros((H, W, 3), np.float32)
            for s in range(SUBFRAMES):
                tk = min((f + s / SUBFRAMES) / (SLOWMO * FPS) / sim.CTRL_DT, n_steps - 1.001)
                k, a = int(tk), tk - int(tk)
                now = []
                for b, (p, q) in zip(bodies, tracks):
                    pp = (1 - a) * p[k] + a * p[k + 1]
                    qq = q[k] if a < 0.5 else q[k + 1]
                    body_q[b] = [*pp, qq[1], qq[2], qq[3], qq[0]]
                    now.append(pp)
                flying = [p for p in now if p[2] > FLOOR_Z] or now  # the camera follows the drones still flying
                lead, lead_y = np.mean([p[0] for p in flying]), np.mean([p[1] for p in flying])
                cam_x = lead if cam_x is None else cam_x + 0.15 * (lead - cam_x)  # smooth chase
                cam_y = lead_y if cam_y is None else cam_y + 0.05 * (lead_y - cam_y)
                last = min(p[0] for p in flying) - 0.3  # the gates behind all the flying drones are hidden
                behind = last if behind is None else max(behind, last)
                for b, g in gates:
                    gate_q[b] = [*(g if g[0] > behind else HIDDEN), 0, 0, 0, 1]
                state.body_q.assign(body_q)
                gate_state.body_q.assign(gate_q)
                eye = np.array([cam_x - 1.6, cam_y - 1.4, 0.65])
                R = look_at(eye, np.array([cam_x + 0.7, cam_y + 0.05, -0.05]))
                acc += cam.render([state, gate_state], eye, R)
            img = np.ascontiguousarray((acc / SUBFRAMES).astype(np.uint8))
            draw_labels(img, [cam.project(p + [0, 0, 0.05], eye, R) if p[2] > FLOOR_Z else None for p in now],
                        labels, highlight)
            ff.stdin.write(img.tobytes())
        for _ in range(FPS // 2):  # half a second of black between episodes
            ff.stdin.write(np.zeros((H, W, 3), np.uint8).tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    tmp.rename(out)


def still(ph, out, highlight=False, level=False):
    """Close-up picture of one design in its hover attitude (level: with the cage level), coloured as in the video."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    add_floor(builder, -3, 3, -3, 3, z=-0.45)
    body = add_drone(builder, ph, highlight)
    model = builder.finalize()
    state = model.state()
    body_q = state.body_q.numpy()
    body_q[body] = [0, 0, 0, *rot_to_quat_xyzw(np.eye(3) if level else warmstart.hover_rotation(ph))]
    state.body_q.assign(body_q)
    eye = np.array([0.55, -0.75, 0.42])
    img = Camera(model, hfov=np.deg2rad(40), ss=2).render([state], eye, look_at(eye, [0, 0, -0.02]))
    cv2.imwrite(str(out), img.astype(np.uint8))
