"""Null-geodesic integrator for the Schwarzschild metric.

Reference Python implementation that mirrors the GLSL shader used in
`demos/08_schwarzschild/`. Used to:

* Sanity-check the shader's photon ring radius and deflection angle
* Export photon trajectories as JSON for Manim/3D scenes

Geometric units: G = c = 1, mass M = 1. Event horizon at r_s = 2M = 2.
The well-known "photon sphere" sits at r = 3M = 3 and the critical
impact parameter for capture is b_crit = 3*sqrt(3) * M ~= 5.196.

The equation of motion for a null geodesic, written in 3D pseudo-
Cartesian coordinates, is::

    d^2 x / d lambda^2  =  -3 * M * |h|^2 * x / r^5

where h = x x dx/d lambda is the conserved angular-momentum vector and
lambda is an affine parameter. This drops out of the standard u(phi)
treatment (u = 1/r, u'' + u = 3 M u^2) re-projected to 3D. (Many
shadertoy implementations use a "-1.5" prefactor; that is the same
equation in units where r_s = 1, i.e. M = 1/2.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

M = 1.0
R_S = 2.0 * M
PHOTON_SPHERE = 3.0 * M
B_CRIT = 3.0 * np.sqrt(3.0) * M


def geodesic_rhs(_lam: float, state: np.ndarray) -> np.ndarray:
    pos = state[:3]
    vel = state[3:]
    r2 = float(pos @ pos)
    r = np.sqrt(r2)
    h = np.cross(pos, vel)
    h2 = float(h @ h)
    acc = -3.0 * M * h2 * pos / (r2 * r2 * r)
    return np.concatenate([vel, acc])


@dataclass
class Trajectory:
    lam: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    captured: bool


def _horizon_event(_lam: float, state: np.ndarray) -> float:
    return float(np.linalg.norm(state[:3]) - R_S * 1.001)


_horizon_event.terminal = True
_horizon_event.direction = -1.0


def _escape_event(_lam: float, state: np.ndarray, far: float) -> float:
    return float(np.linalg.norm(state[:3]) - far)


def shoot_photon(
    start: np.ndarray,
    direction: np.ndarray,
    *,
    far: float = 60.0,
    max_lambda: float = 200.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Trajectory:
    """Integrate a single photon trajectory from `start` in unit `direction`.

    Returns the dense trajectory and a `captured` flag (True if it crosses
    the event horizon before escaping to `far`).
    """
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    state0 = np.concatenate([np.asarray(start, dtype=float), direction])

    def escape(lam, state):
        return _escape_event(lam, state, far)

    escape.terminal = True
    escape.direction = 1.0

    sol = solve_ivp(
        geodesic_rhs,
        (0.0, max_lambda),
        state0,
        events=[_horizon_event, escape],
        rtol=rtol,
        atol=atol,
        max_step=0.5,
        dense_output=True,
    )
    captured = len(sol.t_events[0]) > 0
    return Trajectory(lam=sol.t, pos=sol.y[:3].T, vel=sol.y[3:].T, captured=captured)


def deflection_angle(impact_parameter: float, r_start: float = 50.0) -> float:
    """Compute total deflection angle for a photon with given impact parameter.

    Photons launched from x = -r_start moving in +x direction, offset in y by b.
    Deflection is the angle between final direction and the +x axis.
    """
    start = np.array([-r_start, impact_parameter, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    traj = shoot_photon(start, direction, far=r_start * 2.0)
    if traj.captured:
        return float("nan")
    final_dir = traj.vel[-1] / np.linalg.norm(traj.vel[-1])
    return float(np.arctan2(final_dir[1], final_dir[0]))


def tetrad_ray(
    camera_pos: np.ndarray,
    view_dir: np.ndarray,
    *,
    mass: float = M,
    cam_radial_v: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Reference reproduction of the shader's gold-standard tetrad ray init.

    Mirrors the committed WGSL in ``demos/08_schwarzschild/main.webgpu.js``
    (the ``fs()`` entry point, "gold-standard tetrad ray construction" block,
    ~lines 372-407). This is the newest, hardest-to-eyeball, silent-failure-prone
    layer of the showpiece — a sign error or a botched spherical->Cartesian
    Jacobian here does not crash, it renders a physically-wrong frame and
    corrupts the disk's Doppler beaming asymmetry. The geodesic RHS above is the
    *stable* shared core; this function is the part that actually needs a guard.

    Coordinate convention matches the shader: pseudo-Cartesian with the **y axis**
    as the spherical polar axis, ``x = r sinθ cosφ, y = r cosθ, z = r sinθ sinφ``.

    Parameters
    ----------
    camera_pos : (3,) Cartesian camera position.
    view_dir   : (3,) Cartesian view ray direction (need not be normalized);
                 in the shader this is ``nView``, the pixel ray after the camera
                 matrix.
    mass       : black-hole mass M (geometric units).
    cam_radial_v : camera radial velocity in the local static frame
                 (``u.camRadialV``); negative = infalling.

    Returns
    -------
    (direction, omega_cam):
        ``direction`` — the initial coordinate momentum ``p`` (Cartesian,
        unnormalized) handed to the geodesic integrator.
        ``omega_cam`` — ``1 / E_local``, the per-ray Doppler/redshift weight the
        shader carries downstream to disk coloring.

    Analytic limits (asserted in ``test_schwarzschild.py``):
        * r -> ∞, static camera:  ``direction`` ∝ ``view_dir`` and ``omega_cam`` -> 1.
        * static camera:          ``omega_cam`` = 1/sqrt(1 - 2M/r)  (grav. blueshift).
        * flat limit, radial in-faller looking outward (n1=+1, v_rad=-v):
                                  ``omega_cam`` = sqrt((1+v)/(1-v))  (SR Doppler).
    """
    cam = np.asarray(camera_pos, dtype=float)
    n_view = np.asarray(view_dir, dtype=float)
    n_view = n_view / np.linalg.norm(n_view)

    r0 = float(np.linalg.norm(cam))
    ct0 = cam[1] / max(r0, 1e-3)
    st0 = np.sqrt(max(1.0 - ct0 * ct0, 0.0))
    r_xz = np.sqrt(max(cam[0] * cam[0] + cam[2] * cam[2], 1e-6))
    cp0 = cam[0] / r_xz
    sp0 = cam[2] / r_xz

    r_hat = np.array([st0 * cp0, ct0, st0 * sp0])
    th_hat = np.array([ct0 * cp0, -st0, ct0 * sp0])
    ph_hat = np.array([-sp0, 0.0, cp0])

    n1 = float(n_view @ r_hat)
    n2 = float(n_view @ th_hat)
    n3 = float(n_view @ ph_hat)

    f_cam = 1.0 - 2.0 * mass / max(r0, mass * 2.05)
    f_cam_sqrt = np.sqrt(max(f_cam, 1e-4))

    v_rad = float(np.clip(cam_radial_v, -0.998, 0.998))
    gam = 1.0 / np.sqrt(max(1.0 - v_rad * v_rad, 1e-6))

    e_local = f_cam_sqrt * gam * (1.0 + n1 * v_rad)
    inv_e = 1.0 / max(abs(e_local), 1e-6)

    k_r = (gam * f_cam_sqrt * (v_rad + n1)) * inv_e
    k_th = (n2 / max(r0, 1e-3)) * inv_e
    k_ph = (n3 / max(r0 * max(st0, 0.05), 1e-3)) * inv_e

    px = st0 * cp0 * k_r + r0 * ct0 * cp0 * k_th - r0 * st0 * sp0 * k_ph
    py = ct0 * k_r - r0 * st0 * k_th
    pz = st0 * sp0 * k_r + r0 * ct0 * sp0 * k_th + r0 * st0 * cp0 * k_ph

    direction = np.array([px, py, pz])
    omega_cam = float(inv_e)
    return direction, omega_cam


def export_demo_trajectories(out_path: Path, n_rays: int = 18) -> None:
    """Sample a fan of photon trajectories at varying impact parameters
    and dump as JSON for use in Manim or 3D scenes."""
    bs = np.linspace(0.5 * B_CRIT, 2.5 * B_CRIT, n_rays)
    rays = []
    for b in bs:
        traj = shoot_photon(
            np.array([-30.0, b, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            far=60.0,
        )
        rays.append(
            {
                "impact_parameter": float(b),
                "captured": bool(traj.captured),
                "points": traj.pos.tolist(),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"M": 1.0, "rays": rays}, indent=2))


if __name__ == "__main__":
    print(f"r_s={R_S}  photon sphere={PHOTON_SPHERE}  b_crit={B_CRIT:.4f}")
    for b in [B_CRIT * 0.9, B_CRIT * 1.01, B_CRIT * 1.5, B_CRIT * 3.0]:
        ang_deg = np.degrees(deflection_angle(b))
        print(f"b={b:6.3f}  deflection={ang_deg:8.3f} deg")
    out = Path(__file__).parent / "_export" / "rays.json"
    export_demo_trajectories(out)
    print(f"wrote {out}")
