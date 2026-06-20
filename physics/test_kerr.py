"""Validation tests for the Kerr null-geodesic reference (kerr.py).

Targets are independently sourced (Bardeen 1972 / Teo 2003 photon-orbit formulae,
the Kerr inverse metric, the ZAMO tetrad), NOT derived from kerr.py — so these
check the implementation, not its own algebra. The a=0 cases additionally
cross-check against the separately-tested Schwarzschild module.

    pytest physics/test_kerr.py        # or:  python physics/test_kerr.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kerr import (  # noqa: E402
    carter_Q,
    critical_xi,
    equatorial_inbound,
    horizon_radius,
    integrate,
    integrate_ks,
    inv_metric,
    ks_camera_photon_yaxis,
    ks_equatorial_captured,
    ks_equatorial_captured_yaxis,
    ks_f_l,
    ks_f_l_yaxis,
    ks_two_H,
    ks_two_H_yaxis,
    photon_orbit_radius,
    two_H,
    zamo_photon_init,
)
from schwarzschild import B_CRIT  # noqa: E402

M = 1.0
SPINS = (0.0, 0.5, 0.9, 0.998)


# --------------------------------------------------------------------------- #
# closed forms (independently tabulated)
# --------------------------------------------------------------------------- #
def test_horizon_radius():
    expected = {0.0: 2.0, 0.5: 1.8660254, 0.9: 1.4358899, 0.998: 1.0632139, 1.0: 1.0}
    for a, r in expected.items():
        assert abs(horizon_radius(M, a) - r) < 1e-6, (a, horizon_radius(M, a), r)


def test_photon_orbit_radii():
    # (a, r_ph prograde, r_ph retrograde) from the reference sheet
    table = [
        (0.0, 3.0, 3.0), (0.5, 2.3472964, 3.5320889),
        (0.9, 1.5578546, 3.9102679), (1.0, 1.0, 4.0),
    ]
    for a, rp, rr in table:
        assert abs(photon_orbit_radius(M, a, True) - rp) < 1e-6, (a, "pro")
        assert abs(photon_orbit_radius(M, a, False) - rr) < 1e-6, (a, "retro")


def test_critical_xi_a0_is_b_crit():
    # a=0: |ξ| = 3√3 M, identical to the Schwarzschild capture impact parameter.
    assert abs(critical_xi(M, 0.0, True) - B_CRIT) < 1e-9
    assert abs(critical_xi(M, 0.0, False) + B_CRIT) < 1e-9
    # a=M extremal: +2 (prograde) / −7 (retrograde)
    assert abs(critical_xi(M, 1.0, True) - 2.0) < 1e-9
    assert abs(critical_xi(M, 1.0, False) + 7.0) < 1e-9


def test_inverse_metric_schwarzschild_limit():
    """At a=0 the Kerr inverse metric must equal the Schwarzschild one."""
    for r in (4.0, 10.0, 30.0):
        for th in (0.5, np.pi / 2, 2.3):
            g_tt, g_tph, g_rr, g_thth, g_phph = inv_metric(r, th, M, 0.0)
            f = 1.0 - 2.0 * M / r
            assert abs(g_tt - (-1.0 / f)) < 1e-9
            assert abs(g_tph) < 1e-12
            assert abs(g_rr - f) < 1e-9
            assert abs(g_thth - 1.0 / (r * r)) < 1e-9
            assert abs(g_phph - 1.0 / (r * r * np.sin(th) ** 2)) < 1e-9


# --------------------------------------------------------------------------- #
# integrator: conserved quantities
# --------------------------------------------------------------------------- #
def test_carter_Q_and_null_conserved_offequatorial():
    """Off-equatorial photon: Carter Q, and the null condition 2H≈0, must hold
    along the whole trajectory — the core check that the (numerical) metric
    derivatives in the RHS are right."""
    a = 0.9
    # Launch from a ZAMO at mid-latitude, aimed inward + out of the equator.
    r0, th0 = 30.0, 1.1
    nhat = (-0.85, 0.35, 0.4)  # mostly inward, some θ and φ
    E, L, pr0, pth0, Q0 = zamo_photon_init(r0, th0, nhat, M, a)
    ray = integrate(r0, th0, 0.0, pr0, pth0, E, L, M=M, a=a, far=60.0,
                    max_lambda=400.0, rtol=1e-10, atol=1e-12)
    Q = np.array([carter_Q(th, pth, E, L, a) for th, pth in zip(ray.theta, ray.pth)])
    assert np.allclose(Q, Q0, rtol=1e-5, atol=1e-5), (Q.min(), Q.max(), Q0)
    H2 = np.array([two_H(r, th, pr, pth, E, L, M, a)
                   for r, th, pr, pth in zip(ray.r, ray.theta, ray.pr, ray.pth)])
    assert np.max(np.abs(H2)) < 1e-6 * E * E, np.max(np.abs(H2))


# --------------------------------------------------------------------------- #
# integrator: equatorial circular photon orbit
# --------------------------------------------------------------------------- #
def test_equatorial_circular_photon_orbit():
    """A photon placed exactly on the (unstable) circular orbit with ξ = ξ_c
    must stay at r_ph — a direct joint check of the orbit radius, ξ_c, and the
    integrator. (Unstable, so only integrate a short affine length.)"""
    for a in (0.0, 0.5, 0.9):
        for prograde in (True, False):
            r_ph = photon_orbit_radius(M, a, prograde)
            xi = critical_xi(M, a, prograde)
            ray = integrate(r_ph, np.pi / 2, 0.0, 0.0, 0.0, 1.0, xi,
                            M=M, a=a, far=10.0 * r_ph, max_lambda=6.0,
                            rtol=1e-11, atol=1e-13)
            assert np.max(np.abs(ray.r - r_ph)) < 1e-3, (a, prograde, np.max(np.abs(ray.r - r_ph)))


# --------------------------------------------------------------------------- #
# integrator: capture thresholds & frame-dragging asymmetry
# --------------------------------------------------------------------------- #
def _is_captured(L, a, r0=200.0):
    ray = equatorial_inbound(r0, L, M=M, a=a, E=1.0, far=1.5 * r0,
                             max_lambda=6000.0, rtol=1e-9, atol=1e-11)
    return ray.captured


def _capture_boundary(a, L_captured, L_escaped):
    lo, hi = L_captured, L_escaped
    assert _is_captured(lo, a) and not _is_captured(hi, a), (a, lo, hi)
    for _ in range(38):
        mid = 0.5 * (lo + hi)
        if _is_captured(mid, a):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_capture_threshold_a0_matches_schwarzschild():
    """At a=0 the equatorial capture boundary in ξ must be ±3√3 M = B_CRIT,
    tying the Kerr integrator to the independently-tested Schwarzschild code."""
    xi_plus = _capture_boundary(0.0, 0.0, +10.0)
    xi_minus = _capture_boundary(0.0, 0.0, -10.0)
    assert abs(xi_plus - B_CRIT) < 3e-3, (xi_plus, B_CRIT)
    assert abs(xi_minus + B_CRIT) < 3e-3, (xi_minus, -B_CRIT)


def test_kerr_shadow_is_frame_drag_asymmetric():
    """The headline physics: spin makes the photon-ring edges asymmetric. The
    prograde/retrograde capture boundaries must match ξ_± = ±3√(M r_ph) − a,
    and the prograde edge must sit closer in than the retrograde one."""
    for a in (0.5, 0.9):
        xi_plus = _capture_boundary(a, 0.0, +12.0)
        xi_minus = _capture_boundary(a, 0.0, -12.0)
        assert abs(xi_plus - critical_xi(M, a, True)) < 3e-3, (a, xi_plus)
        assert abs(xi_minus - critical_xi(M, a, False)) < 3e-3, (a, xi_minus)
        assert abs(xi_plus) < abs(xi_minus)  # prograde edge nearer (dragging)


def test_frame_dragging_sign_and_magnitude():
    """A zero-angular-momentum photon (L=0) is still dragged: dφ/dλ has the sign
    of a and ≈ 2MaE/r³ at large r."""
    r0, th = 100.0, np.pi / 2
    for a in (0.5, -0.5, 0.9):
        # purely radial-outward ZAMO direction => n_φ = 0 => L = 0
        E, L, pr0, pth0, _Q = zamo_photon_init(r0, th, (1.0, 0.0, 0.0), M, a)
        assert abs(L) < 1e-9
        g_tt, g_tph, g_rr, g_thth, g_phph = inv_metric(r0, th, M, a)
        dphi = -g_tph * E + g_phph * L  # = -g_tph * E
        assert np.sign(dphi) == np.sign(a), (a, dphi)
        approx = 2.0 * M * a * E / r0**3
        assert abs(dphi - approx) < 0.05 * abs(approx), (a, dphi, approx)


# --------------------------------------------------------------------------- #
# Kerr-Schild Cartesian form (the shader's algorithm) vs the BL physics
# --------------------------------------------------------------------------- #
_KS_D = 1500.0


def _ks_capture_boundary(a, b_captured, b_escaped):
    lo, hi = b_captured, b_escaped
    assert (ks_equatorial_captured(lo, a=a, M=M, D=_KS_D)
            and not ks_equatorial_captured(hi, a=a, M=M, D=_KS_D)), (a, lo, hi)
    for _ in range(34):
        mid = 0.5 * (lo + hi)
        if ks_equatorial_captured(mid, a=a, M=M, D=_KS_D):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_ks_null_condition_conserved():
    """The Kerr-Schild integrator must keep g^μν p_μ p_ν ≈ 0 along a ray
    (validates the numerical metric-gradient RHS in Cartesian form)."""
    res = integrate_ks(np.array([-40.0, 6.0, 3.0]), np.array([1.0, 0.0, 0.0]),
                       M=M, a=0.9, far=80.0, max_lambda=400.0, rtol=1e-10, atol=1e-12)
    H = np.array([ks_two_H(p, mom, -res["E"], M, 0.9)
                  for p, mom in zip(res["pos"], res["p"])])
    assert np.max(np.abs(H)) < 5e-4, np.max(np.abs(H))


def test_ks_a0_capture_matches_schwarzschild():
    """Kerr-Schild Cartesian at a=0 must give the same |b_crit| = 3√3 M."""
    b_hi = _ks_capture_boundary(0.0, 0.0, +10.0)   # = −ξ₋ = +3√3
    b_lo = _ks_capture_boundary(0.0, 0.0, -10.0)   # = −ξ₊ = −3√3
    assert abs(b_hi - B_CRIT) < 1.2e-2, (b_hi, B_CRIT)
    assert abs(b_lo + B_CRIT) < 1.2e-2, (b_lo, -B_CRIT)


def test_ks_kerr_shadow_asymmetry_matches_bl():
    """Headline: the Cartesian form reproduces the BL frame-drag asymmetry.
    ξ = −b, so the b-boundaries map to ξ_∓; assert they match critical_xi and
    the prograde edge sits nearer."""
    for a in (0.5, 0.9):
        b_hi = _ks_capture_boundary(a, 0.0, +14.0)   # −b_hi = ξ₋ (retrograde)
        b_lo = _ks_capture_boundary(a, 0.0, -14.0)   # −b_lo = ξ₊ (prograde)
        assert abs(-b_hi - critical_xi(M, a, False)) < 1.5e-2, (a, -b_hi)
        assert abs(-b_lo - critical_xi(M, a, True)) < 1.5e-2, (a, -b_lo)
        assert abs(b_lo) < abs(b_hi)  # prograde edge nearer (dragging)


# --------------------------------------------------------------------------- #
# spin-axis = +y formulas (the literal shader template)
# --------------------------------------------------------------------------- #
def test_ks_yaxis_is_rotation_of_zaxis():
    """The y-axis Kerr-Schild f and l must be the exact proper rotation R_x(90°)
    of the validated z-axis version: world(spin=y) --R--> canonical(spin=z),
    (x,y,z)->(x,−z,y); covector maps back by R^T. Instant, exact certification of
    the formulas that go into WGSL."""
    def R(v):     return np.array([v[0], -v[2], v[1]])    # world -> canonical
    def Rinv(v):  return np.array([v[0], v[2], -v[1]])    # R^T: canonical -> world
    for a in (0.3, 0.9):
        for pos in [(3.0, 1.0, 2.0), (-5.0, 2.0, -1.0), (4.0, -3.0, 2.0),
                    (1.5, 0.2, 4.0), (-2.0, -4.0, -3.0)]:
            pos = np.array(pos)
            f_y, l_y = ks_f_l_yaxis(pos, M, a)
            f_z, l_z = ks_f_l(R(pos), M, a)
            assert abs(f_y - f_z) < 1e-9, (a, pos, f_y, f_z)
            assert np.allclose(l_y, Rinv(l_z), atol=1e-9), (a, pos, l_y, Rinv(l_z))


def test_ks_yaxis_integrator_chirality():
    """The y-axis integrator must show the right asymmetry: at a=0.9 the prograde
    edge is ξ₊≈2.84, the retrograde ξ₋≈−6.83. So a b=+1 photon (prograde, inside)
    plunges, b=+5 (just past prograde edge) escapes, b=−5 (still inside retrograde)
    plunges, b=−8 (past retrograde edge) escapes."""
    a = 0.9
    assert ks_equatorial_captured_yaxis(+1.0, a=a, M=M, D=800.0)
    assert not ks_equatorial_captured_yaxis(+5.0, a=a, M=M, D=800.0)
    assert ks_equatorial_captured_yaxis(-5.0, a=a, M=M, D=800.0)
    assert not ks_equatorial_captured_yaxis(-8.0, a=a, M=M, D=800.0)


def test_ks_camera_init_is_null_and_aimed():
    """The shader's camera-ray init must produce a photon that is (a) exactly
    null in Kerr-Schild and (b) whose coordinate velocity dx/dλ points along the
    requested pixel direction. Those two properties ARE its correctness — instant
    certification of the one piece that isn't a transcription of validated code."""
    a = 0.9
    cam = np.array([5.0, 2.0, -18.0])  # off-axis, r≈19
    for nView in (np.array([0.0, 0.0, 1.0]), np.array([0.2, -0.3, 0.93]),
                  np.array([-0.5, 0.1, 0.86]), np.array([0.7, 0.2, 0.68])):
        nView = nView / np.linalg.norm(nView)
        p, E = ks_camera_photon_yaxis(cam, nView, M, a)
        # (a) null in KS
        assert abs(ks_two_H_yaxis(cam, p, -E, M, a)) < 1e-9, (nView, E)
        # (b) coordinate velocity parallel to the view direction
        f, l = ks_f_l_yaxis(cam, M, a)
        dx = p - f * l * (E + float(l @ p))
        dxhat = dx / np.linalg.norm(dx)
        assert np.allclose(dxhat, nView, atol=1e-9), (nView, dxhat)
        assert E > 0


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            fails += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
