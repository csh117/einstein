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

    @property
    def r(self) -> np.ndarray:
        return np.linalg.norm(self.pos, axis=1)


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
