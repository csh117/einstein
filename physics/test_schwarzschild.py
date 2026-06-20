"""Validation tests for the reference Schwarzschild physics.

Two jobs, matching the audit's Batch 1 (docs/audit/20260619_cc-only_secondpass.md):

1. **Integrator invariants** (the stable shared core): b_crit, capture
   threshold, angular-momentum conservation, weak-field deflection. Cheap
   regression guards that any future edit to ``geodesic_rhs`` would trip.

2. **Tetrad / Doppler analytic limits** (the *high-value* part): assert that
   ``tetrad_ray`` — the Python mirror of the committed WGSL gold-standard ray
   init in ``demos/08_schwarzschild/main.webgpu.js`` — satisfies its physical
   limits. This is the layer the audit flagged as silent-wrong-frame risk:
   newest code, hand-transcribed once per shader language, no reference anywhere.
   Anchored to the **committed WGSL** (main.webgpu.js), deliberately NOT the
   in-flight GLSL rewrite (main.js), per the second-pass plan.

Runs two ways:
    pytest physics/test_schwarzschild.py        # if pytest is installed
    python  physics/test_schwarzschild.py       # plain-asserts fallback, no pytest

Deps: numpy + scipy (already in physics/requirements.txt); pytest is optional
(physics/requirements-dev.txt).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schwarzschild import (  # noqa: E402
    B_CRIT,
    M,
    PHOTON_SPHERE,
    R_S,
    deflection_angle,
    shoot_photon,
    tetrad_ray,
)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_X_HAT = np.array([1.0, 0.0, 0.0])


def _is_captured(b: float, r_start: float = 60.0) -> bool:
    traj = shoot_photon(
        np.array([-r_start, b, 0.0]),
        _X_HAT,
        far=160.0,
        max_lambda=800.0,
    )
    return traj.captured


def _deflection(b: float, r_start: float) -> float:
    """Total deflection magnitude for impact parameter ``b``, measured from a
    finite start at x=-r_start out to far=2*r_start. Returns |angle| in radians.
    Uses shoot_photon directly so we can size max_lambda to the path length
    (the module's deflection_angle() hard-codes the default and only works for
    small r_start)."""
    far = 2.0 * r_start
    traj = shoot_photon(
        np.array([-r_start, b, 0.0]),
        _X_HAT,
        far=far,
        max_lambda=4.0 * r_start + 400.0,
    )
    assert not traj.captured, f"photon at b={b} unexpectedly captured"
    final_dir = traj.vel[-1] / np.linalg.norm(traj.vel[-1])
    return abs(float(np.arctan2(final_dir[1], final_dir[0])))


# --------------------------------------------------------------------------- #
# 1. integrator invariants  (the stable shared core)
# --------------------------------------------------------------------------- #
def test_b_crit_constant():
    """b_crit = 3*sqrt(3)*M is the textbook photon-capture impact parameter."""
    assert np.isclose(B_CRIT, 3.0 * np.sqrt(3.0) * M)
    assert np.isclose(R_S, 2.0 * M)
    assert np.isclose(PHOTON_SPHERE, 3.0 * M)


def test_capture_escape_pair():
    """A photon well inside b_crit is captured; one well outside escapes."""
    assert _is_captured(0.85 * B_CRIT)
    assert not _is_captured(1.15 * B_CRIT)


def test_capture_threshold_matches_b_crit():
    """Bisect the actual numerical capture boundary and confirm it lands on
    3*sqrt(3)*M. Parameter-free 'golden' value — catches a wrong prefactor or
    a wrong M-scaling in geodesic_rhs (either would move the threshold)."""
    lo, hi = 0.5 * B_CRIT, 1.5 * B_CRIT  # lo captured, hi escapes
    assert _is_captured(lo)
    assert not _is_captured(hi)
    for _ in range(34):
        mid = 0.5 * (lo + hi)
        if _is_captured(mid):
            lo = mid
        else:
            hi = mid
    b_thresh = 0.5 * (lo + hi)
    assert abs(b_thresh - B_CRIT) < 0.03 * B_CRIT, (b_thresh, B_CRIT)


def test_angular_momentum_conserved():
    """h = x × v is an exact constant of geodesic_rhs (acc ∝ x ⇒ x×acc = 0).
    Its magnitude equals the impact parameter b. Verify it stays put along an
    escaping trajectory — drift would signal a broken RHS or stepper."""
    b = 1.5 * B_CRIT
    traj = shoot_photon(
        np.array([-60.0, b, 0.0]), _X_HAT, far=160.0, max_lambda=800.0
    )
    h_mag = np.linalg.norm(np.cross(traj.pos, traj.vel), axis=1)
    assert np.allclose(h_mag, b, rtol=1e-4, atol=1e-4)


def _gr_deflection_series(b: float) -> float:
    """Schwarzschild light-bending angle as a series in u = M/b:
        α = 4u + (15π/4)u² + (128/3)u³ + O(u⁴)
    Leading term is Einstein's 4M/b ( = 2 R_S / b ), twice the Newtonian value.
    At b≈40M the second/third terms already contribute ~8%, so testing against
    the leading term alone is wrong — we test the series."""
    u = M / b
    return 4.0 * u + (15.0 * np.pi / 4.0) * u * u + (128.0 / 3.0) * u**3


def test_weak_field_deflection_matches_einstein():
    """Deflection matches the GR series (whose leading term is Einstein's 4M/b),
    confirming both the correct prefactor and the first relativistic corrections.
    A Newtonian prefactor (2M/b) or a missing factor of 2 is a ~50-100% error;
    tolerance here is 1%, comfortably above the residual O(u⁴) + integrator error
    (~0.06% at b=40M, smaller for larger b)."""
    r_start = 1500.0
    for b in (40.0, 60.0, 90.0):
        defl = _deflection(b, r_start)
        expected = _gr_deflection_series(b)
        assert abs(defl - expected) < 0.01 * expected, (b, defl, expected)


# --------------------------------------------------------------------------- #
# 2. tetrad / Doppler analytic limits  (the silent-wrong-frame risk layer)
# --------------------------------------------------------------------------- #
def test_tetrad_direction_reduces_to_view_far_static():
    """At large r with a static camera the tetrad ray must reduce to 'shoot the
    pixel ray straight': dir ∝ nView and omega_cam -> 1. This is the strongest
    single check on the spherical basis + spherical->Cartesian Jacobian — a sign
    or transpose error breaks it."""
    r0 = 1.0e6
    cam = np.array([0.3, 0.7, -0.5])
    cam = cam / np.linalg.norm(cam) * r0
    for nv in (
        np.array([0.0, 0.0, -1.0]),
        np.array([0.2, -0.3, 0.9]),
        np.array([1.0, 0.0, 0.0]),
        np.array([-0.6, 0.5, 0.62]),
    ):
        direction, omega = tetrad_ray(cam, nv, mass=M, cam_radial_v=0.0)
        d_hat = direction / np.linalg.norm(direction)
        nv_hat = nv / np.linalg.norm(nv)
        assert np.allclose(d_hat, nv_hat, atol=1e-4), (nv, d_hat)
        assert abs(omega - 1.0) < 1e-4


def test_omega_cam_static_is_gravitational_blueshift():
    """Static camera (v_rad=0): omega_cam = 1/sqrt(1 - 2M/r), the gravitational
    blueshift of a static observer relative to infinity. Independent of view
    direction. Catches the f_cam factor."""
    nv = np.array([0.1, 0.2, -0.97])
    for r0 in (8.0, 20.0, 50.0):
        cam = np.array([0.6, 0.0, 0.8]) * r0  # |(0.6,0,0.8)| = 1
        _, omega = tetrad_ray(cam, nv, mass=M, cam_radial_v=0.0)
        expected = 1.0 / np.sqrt(1.0 - 2.0 * M / r0)
        assert abs(omega - expected) < 1e-6 * expected, (r0, omega, expected)


def test_omega_cam_sr_doppler_radial_infaller():
    """Flat-space limit (large r), camera looking radially OUTWARD (n1=+1) while
    infalling at speed v (v_rad=-v): omega_cam = sqrt((1+v)/(1-v)), the special-
    relativistic longitudinal *blueshift*. This is the factor that drives the
    iconic blue-approaching / red-receding disk asymmetry — the thing most worth
    not getting silently wrong."""
    r0 = 1.0e6
    cam = np.array([0.36, 0.48, 0.8])
    cam = cam / np.linalg.norm(cam) * r0
    nv = cam / np.linalg.norm(cam)  # look radially outward => n1 = +1
    for v in (0.1, 0.3, 0.6, 0.9):
        _, omega = tetrad_ray(cam, nv, mass=M, cam_radial_v=-v)
        expected = np.sqrt((1.0 + v) / (1.0 - v))
        assert abs(omega - expected) < 1e-3 * expected, (v, omega, expected)


def test_omega_cam_redshift_when_receding():
    """Symmetry check: same geometry, camera moving radially OUTWARD (v_rad=+v)
    while looking outward => redshift (omega_cam < 1)."""
    r0 = 1.0e6
    cam = np.array([0.36, 0.48, 0.8])
    cam = cam / np.linalg.norm(cam) * r0
    nv = cam / np.linalg.norm(cam)
    _, omega = tetrad_ray(cam, nv, mass=M, cam_radial_v=+0.5)
    expected = np.sqrt((1.0 - 0.5) / (1.0 + 0.5))
    assert omega < 1.0
    assert abs(omega - expected) < 1e-3 * expected


def test_omega_cam_factorizes_grav_times_doppler():
    """At finite r, radial in-faller looking outward, omega_cam must factor as
    (gravitational blueshift) x (SR Doppler) = 1/sqrt(f) * sqrt((1+v)/(1-v)).
    Couples the two effects — catches a sign error that happens to cancel in
    either limit alone."""
    for r0 in (12.0, 30.0):
        cam = np.array([0.48, 0.6, 0.64]) * r0  # unit-norm direction * r0
        nv = cam / np.linalg.norm(cam)  # n1 = +1
        f = 1.0 - 2.0 * M / r0
        for v in (0.2, 0.5):
            _, omega = tetrad_ray(cam, nv, mass=M, cam_radial_v=-v)
            expected = (1.0 / np.sqrt(f)) * np.sqrt((1.0 + v) / (1.0 - v))
            assert abs(omega - expected) < 1e-6 * expected, (r0, v, omega, expected)


# --------------------------------------------------------------------------- #
# pytest-free runner
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
