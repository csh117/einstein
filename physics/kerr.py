"""Kerr null-geodesic integrator — reference for the WebGPU ray-marcher.

Real (not approximate) Kerr photon dynamics, so the showpiece's spinning black
hole is physically correct rather than Schwarzschild-with-a-disk-tweak. This is
the validated reference; `demos/08_schwarzschild/main.webgpu.js` transcribes the
same algorithm into WGSL. `test_kerr.py` checks this module against independently
tabulated values (Bardeen 1972 / Teo 2003 / YNOGK), and against the already-tested
Schwarzschild code in the a=0 limit.

Formulation
-----------
Boyer-Lindquist coordinates (t, r, θ, φ), geometric units G = c = M-scalable,
signature (−+++). Spin axis = polar axis (θ measured from it). With

    Σ = r² + a²cos²θ,   Δ = r² − 2Mr + a²,   A = (r²+a²)² − a²Δsin²θ

the inverse metric is

    g^tt = −A/(ΣΔ),  g^tφ = −2Mar/(ΣΔ),  g^rr = Δ/Σ,  g^θθ = 1/Σ,
    g^φφ = (Δ − a²sin²θ)/(ΣΔsin²θ).

Photons follow the null Hamiltonian H = ½ g^μν p_μ p_ν = 0 with

    dx^μ/dλ = g^μν p_ν,   dp_μ/dλ = −½ (∂_μ g^αβ) p_α p_β.

Because g is independent of t and φ, E ≡ −p_t and L ≡ +p_φ are conserved, so we
integrate only (r, θ, φ, p_r, p_θ). The momentum derivatives use **numerical**
(central-difference) metric derivatives — robust, transcription-error-free, and
the exact scheme mirrored in the shader so the two stay in lock-step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


def horizon_radius(M: float, a: float) -> float:
    """Outer event horizon r_+ = M + sqrt(M² − a²)."""
    return M + np.sqrt(max(M * M - a * a, 0.0))


def ergosphere_radius(M: float, a: float, theta: float) -> float:
    """Outer ergosurface r_E(θ) = M + sqrt(M² − a²cos²θ)."""
    ct = np.cos(theta)
    return M + np.sqrt(max(M * M - a * a * ct * ct, 0.0))


def photon_orbit_radius(M: float, a: float, prograde: bool = True) -> float:
    """Equatorial circular photon-orbit radius.

    r_ph = 2M{1 + cos[(2/3) arccos(∓a/M)]}, upper sign prograde. a=0 -> 3M;
    a=M -> 1M (prograde) / 4M (retrograde)."""
    sign = -1.0 if prograde else 1.0
    return 2.0 * M * (1.0 + np.cos((2.0 / 3.0) * np.arccos(sign * a / M)))


def critical_xi(M: float, a: float, prograde: bool = True) -> float:
    """Conserved ξ = L/E of the equatorial circular photon orbit (photon-ring
    edge): ξ = ±3√(M r_ph) − a (+ prograde). a=0 -> ±3√3 M; a=M -> +2 / −7."""
    r_ph = photon_orbit_radius(M, a, prograde)
    sign = 1.0 if prograde else -1.0
    return sign * 3.0 * np.sqrt(M * r_ph) - a


def _SDA(r: float, theta: float, M: float, a: float) -> tuple[float, float, float]:
    ct = np.cos(theta)
    st = np.sin(theta)
    Sigma = r * r + a * a * ct * ct
    Delta = r * r - 2.0 * M * r + a * a
    A = (r * r + a * a) ** 2 - a * a * Delta * st * st
    return Sigma, Delta, A


def inv_metric(r: float, theta: float, M: float, a: float):
    """Kerr inverse-metric components (g^tt, g^tφ, g^rr, g^θθ, g^φφ)."""
    st = np.sin(theta)
    Sigma, Delta, A = _SDA(r, theta, M, a)
    st2 = max(st * st, 1e-12)  # guard the polar axis
    g_tt = -A / (Sigma * Delta)
    g_tph = -2.0 * M * a * r / (Sigma * Delta)
    g_rr = Delta / Sigma
    g_thth = 1.0 / Sigma
    g_phph = (Delta - a * a * st2) / (Sigma * Delta * st2)
    return g_tt, g_tph, g_rr, g_thth, g_phph


def two_H(r, theta, pr, pth, E, L, M, a) -> float:
    """g^αβ p_α p_β with p_t=−E, p_φ=L (== 2H; should stay ≈ 0 on a null ray)."""
    g_tt, g_tph, g_rr, g_thth, g_phph = inv_metric(r, theta, M, a)
    return (g_tt * E * E - 2.0 * g_tph * E * L + g_phph * L * L
            + g_rr * pr * pr + g_thth * pth * pth)


def carter_Q(theta, pth, E, L, a) -> float:
    """Carter constant Q = p_θ² + cos²θ (L²/sin²θ − a²E²)."""
    ct = np.cos(theta)
    st2 = max(np.sin(theta) ** 2, 1e-12)
    return pth * pth + ct * ct * (L * L / st2 - a * a * E * E)


def zamo_photon_init(r0, theta0, nhat, M, a):
    """Initial (E, L, p_r, p_θ, Q) for a photon launched from a ZAMO at
    (r0, θ0) in local orthonormal unit 3-direction nhat = (n_r, n_θ, n_φ).

    Local energy is normalized to 1: p_(t̂) = −1, p_(î) = n_î. Dual tetrad gives
    p_r = √g_rr n_r, p_θ = √g_θθ n_θ, L = √g_φφ n_φ, E = α + ωL (α lapse,
    ω = 2Mar/A the frame-drag rate). E = α + ωL is the dragging coupling."""
    nhat = np.asarray(nhat, dtype=float)
    nhat = nhat / np.linalg.norm(nhat)
    st = np.sin(theta0)
    Sigma, Delta, A = _SDA(r0, theta0, M, a)
    g_rr_cov = Sigma / Delta
    g_thth_cov = Sigma
    g_phph_cov = (A / Sigma) * st * st
    alpha = np.sqrt(Sigma * Delta / A)
    omega = 2.0 * M * a * r0 / A
    pr = np.sqrt(g_rr_cov) * nhat[0]
    pth = np.sqrt(g_thth_cov) * nhat[1]
    L = np.sqrt(g_phph_cov) * nhat[2]
    E = alpha + omega * L
    Q = carter_Q(theta0, pth, E, L, a)
    return float(E), float(L), float(pr), float(pth), float(Q)


@dataclass
class KerrRay:
    lam: np.ndarray
    r: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    pr: np.ndarray
    pth: np.ndarray
    E: float
    L: float
    captured: bool
    escaped: bool


def _rhs(lam, s, M, a, E, L):
    r, th, _phi, pr, pth = s
    g_tt, g_tph, g_rr, g_thth, g_phph = inv_metric(r, th, M, a)
    dr = g_rr * pr
    dth = g_thth * pth
    dphi = -g_tph * E + g_phph * L  # g^φt p_t + g^φφ p_φ
    # dp_i/dλ = −½ ∂_i (g^αβ p_α p_β) at fixed momenta — central differences.
    hr = 1e-6 * max(abs(r), 1.0)
    hth = 1e-6
    dH_dr = (two_H(r + hr, th, pr, pth, E, L, M, a)
             - two_H(r - hr, th, pr, pth, E, L, M, a)) / (2.0 * hr)
    dH_dth = (two_H(r, th + hth, pr, pth, E, L, M, a)
              - two_H(r, th - hth, pr, pth, E, L, M, a)) / (2.0 * hth)
    return [dr, dth, dphi, -0.5 * dH_dr, -0.5 * dH_dth]


def integrate(
    r0, theta0, phi0, pr0, pth0, E, L, *, M=1.0, a=0.0,
    far=None, max_lambda=4000.0, rtol=1e-9, atol=1e-11,
) -> KerrRay:
    """Integrate one photon. Terminates on capture (r → r_+·1.01) or escape
    (r > far). `far` defaults to 1.2·r0."""
    if far is None:
        far = 1.2 * r0
    r_horizon = horizon_radius(M, a) * 1.01

    def hit_horizon(lam, s, *_):
        return s[0] - r_horizon
    hit_horizon.terminal = True
    hit_horizon.direction = -1.0

    def escape(lam, s, *_):
        return s[0] - far
    escape.terminal = True
    escape.direction = 1.0

    sol = solve_ivp(
        _rhs, (0.0, max_lambda), [r0, theta0, phi0, pr0, pth0],
        args=(M, a, E, L), events=[hit_horizon, escape],
        rtol=rtol, atol=atol, max_step=1.0, dense_output=False,
    )
    captured = len(sol.t_events[0]) > 0
    escaped = len(sol.t_events[1]) > 0
    return KerrRay(
        lam=sol.t, r=sol.y[0], theta=sol.y[1], phi=sol.y[2],
        pr=sol.y[3], pth=sol.y[4], E=E, L=L,
        captured=captured, escaped=escaped,
    )


def equatorial_inbound(r0, L, *, M=1.0, a=0.0, E=1.0, far=None, **kw) -> KerrRay:
    """Launch an equatorial (θ=π/2, p_θ=0) photon inbound from r0 with energy E
    and axial angular momentum L. p_r fixed by the null condition. Prograde
    L>0, retrograde L<0."""
    g_tt, g_tph, g_rr, g_thth, g_phph = inv_metric(r0, np.pi / 2, M, a)
    rad = -(g_tt * E * E - 2.0 * g_tph * E * L + g_phph * L * L) / g_rr
    if rad < 0:
        raise ValueError(f"no inbound ray at r0={r0} with L={L} (turning point outside)")
    pr0 = -np.sqrt(rad)  # inward
    return integrate(r0, np.pi / 2, 0.0, pr0, 0.0, E, L, M=M, a=a, far=far, **kw)


# =========================================================================== #
# Kerr-Schild Cartesian formulation  (the form the WGSL ray-marcher uses)
# =========================================================================== #
# Boyer-Lindquist has a coordinate singularity at the horizon and isn't
# Cartesian — bad for a Cartesian ray-marcher that pushes right up to r_+. The
# Kerr-Schild form is horizon-penetrating and Cartesian: spin axis = z,
#   g_μν = η_μν + f l_μ l_ν,   g^μν = η^μν − f l^μ l^ν   (l null),
# with the "radius" r(x,y,z) the positive root of r⁴ − (R²−a²)r² − a²z² = 0.
# Photons integrate (x, p) with dx^i/dλ = g^iμ p_μ, dp_i/dλ = −½ ∂_i g^μν p_μ p_ν
# and p_t = −E conserved. This module validates it against the BL version via the
# coordinate-invariant shadow edges (ξ_±); the shader transcribes it verbatim.


def ks_radius(pos, a: float) -> float:
    """Kerr-Schild radius r: positive root of r⁴ − (R²−a²)r² − a²z² = 0."""
    x, y, z = pos
    R2 = x * x + y * y + z * z
    a2 = a * a
    r2 = 0.5 * ((R2 - a2) + np.sqrt((R2 - a2) ** 2 + 4.0 * a2 * z * z))
    return np.sqrt(max(r2, 1e-12))


def ks_f_l(pos, M: float, a: float):
    """Kerr-Schild scalar f and covariant null 3-vector l_i (spin axis = z)."""
    x, y, z = pos
    r = ks_radius(pos, a)
    a2, r2 = a * a, 0.0
    r2 = r * r
    f = 2.0 * M * r2 * r / (r2 * r2 + a2 * z * z)
    l = np.array([(r * x + a * y) / (r2 + a2),
                  (r * y - a * x) / (r2 + a2),
                  z / r])
    return f, l


def ks_two_H(pos, p, pt: float, M: float, a: float) -> float:
    """g^μν p_μ p_ν = (−p_t² + p·p) − f (l^μ p_μ)²,  l^μ p_μ = −p_t + l·p."""
    f, l = ks_f_l(pos, M, a)
    eta = -pt * pt + float(p @ p)
    lp = -pt + float(l @ p)
    return eta - f * lp * lp


def _rhs_ks(lam, s, M, a, pt):
    pos = np.array(s[:3])
    p = np.array(s[3:])
    f, l = ks_f_l(pos, M, a)
    lp = -pt + float(l @ p)
    dx = p - f * l * lp  # dx^i/dλ = g^iμ p_μ
    grad = np.empty(3)
    for i in range(3):
        h = 1e-6 * max(abs(pos[i]), 1.0)
        pp = pos.copy(); pp[i] += h
        pm = pos.copy(); pm[i] -= h
        grad[i] = (ks_two_H(pp, p, pt, M, a) - ks_two_H(pm, p, pt, M, a)) / (2.0 * h)
    return [*dx, *(-0.5 * grad)]


def integrate_ks(pos0, p0, *, M=1.0, a=0.0, E=None, far=None,
                 max_lambda=8000.0, rtol=1e-9, atol=1e-11):
    """Integrate a Kerr-Schild Cartesian photon from pos0 with spatial momentum
    p0. E defaults to |p0| (flat-space null normalization). Captured when the KS
    radius drops below r_+ (horizon-penetrating, so this is clean)."""
    pos0 = np.asarray(pos0, float)
    p0 = np.asarray(p0, float)
    if E is None:
        E = float(np.linalg.norm(p0))
    pt = -E
    if far is None:
        far = 1.5 * float(np.linalg.norm(pos0))
    r_capture = horizon_radius(M, a) * 1.001

    def hit_horizon(lam, s, *_):
        return ks_radius(s[:3], a) - r_capture
    hit_horizon.terminal = True
    hit_horizon.direction = -1.0

    def escape(lam, s, *_):
        return float(np.linalg.norm(s[:3])) - far
    escape.terminal = True
    escape.direction = 1.0

    sol = solve_ivp(_rhs_ks, (0.0, max_lambda), [*pos0, *p0],
                    args=(M, a, pt), events=[hit_horizon, escape],
                    rtol=rtol, atol=atol, max_step=1.0)
    return {
        "pos": sol.y[:3].T, "p": sol.y[3:].T, "lam": sol.t,
        "captured": len(sol.t_events[0]) > 0,
        "escaped": len(sol.t_events[1]) > 0, "E": E,
    }


def ks_equatorial_captured(b, *, M=1.0, a=0.0, D=1000.0) -> bool:
    """Equatorial (z=0) photon from (−D, b, 0) heading +x. Impact parameter |b|;
    conserved L_z = −b, so ξ = L_z/E = −b. Returns True if it plunges."""
    pos0 = np.array([-D, b, 0.0])
    p0 = np.array([1.0, 0.0, 0.0])  # E = 1
    res = integrate_ks(pos0, p0, M=M, a=a, far=1.5 * D)
    return res["captured"]


# --------------------------------------------------------------------------- #
# Spin axis = +y  (the shader's convention: disk normal = y, equatorial plane y=0)
# --------------------------------------------------------------------------- #
# Same Kerr-Schild physics as above, rotated so the spin axis is +y instead of
# +z. These are the EXACT formulas transcribed into the WGSL ray-marcher.
# test_kerr.py proves they are the proper rotation R_x(90°) of the validated
# z-axis version: world(spin=y) --R--> canonical(spin=z), (x,y,z) -> (x,−z,y).


def ks_radius_yaxis(pos, a: float) -> float:
    x, y, z = pos
    R2 = x * x + y * y + z * z
    a2 = a * a
    r2 = 0.5 * ((R2 - a2) + np.sqrt((R2 - a2) ** 2 + 4.0 * a2 * y * y))
    return np.sqrt(max(r2, 1e-12))


def ks_f_l_yaxis(pos, M: float, a: float):
    """Kerr-Schild f and covariant null l_i with spin axis = +y."""
    x, y, z = pos
    r = ks_radius_yaxis(pos, a)
    a2, r2 = a * a, r * r
    f = 2.0 * M * r2 * r / (r2 * r2 + a2 * y * y)
    l = np.array([(r * x - a * z) / (r2 + a2),
                  y / r,
                  (r * z + a * x) / (r2 + a2)])
    return f, l


def ks_two_H_yaxis(pos, p, pt: float, M: float, a: float) -> float:
    f, l = ks_f_l_yaxis(pos, M, a)
    return -pt * pt + float(p @ p) - f * (-pt + float(l @ p)) ** 2


def _rhs_ks_yaxis(lam, s, M, a, pt):
    pos = np.array(s[:3])
    p = np.array(s[3:])
    f, l = ks_f_l_yaxis(pos, M, a)
    lp = -pt + float(l @ p)
    dx = p - f * l * lp
    grad = np.empty(3)
    for i in range(3):
        h = 1e-6 * max(abs(pos[i]), 1.0)
        pp = pos.copy(); pp[i] += h
        pm = pos.copy(); pm[i] -= h
        grad[i] = (ks_two_H_yaxis(pp, p, pt, M, a)
                   - ks_two_H_yaxis(pm, p, pt, M, a)) / (2.0 * h)
    return [*dx, *(-0.5 * grad)]


def ks_equatorial_captured_yaxis(b, *, M=1.0, a=0.0, D=1000.0) -> bool:
    """Equatorial (y=0) photon from (−D, 0, b) heading +x. L_y = b ⇒ ξ = b.
    Captured region ξ∈(ξ₋,ξ₊); prograde (co-rotating with +y spin) is b>0."""
    pos0 = np.array([-D, 0.0, b])
    p0 = np.array([1.0, 0.0, 0.0])
    pt = -1.0
    r_cap = horizon_radius(M, a) * 1.001

    def hor(lam, s, *_):
        return ks_radius_yaxis(s[:3], a) - r_cap
    hor.terminal = True
    hor.direction = -1.0

    def esc(lam, s, *_):
        return float(np.linalg.norm(s[:3])) - 1.5 * D
    esc.terminal = True
    esc.direction = 1.0

    sol = solve_ivp(_rhs_ks_yaxis, (0.0, 12000.0), [*pos0, *p0],
                    args=(M, a, pt), events=[hor, esc],
                    rtol=1e-9, atol=1e-11, max_step=1.0)
    return len(sol.t_events[0]) > 0


def ks_camera_photon_yaxis(cam, view_dir, M: float, a: float):
    """Initial (p, E) for a photon launched from `cam` whose coordinate velocity
    points along `view_dir` (world unit) and that is exactly null in Kerr-Schild
    (spin=y). This is the shader's camera-ray init.

    Method: a camera at rest in KS coords sees the photon with 4-velocity
    v = (v^t, view_dir). Null ⇒ g_μν v^μ v^ν = 0, i.e. with l_t = 1,
        (f−1)(v^t)² + 2f(l·n̂)v^t + [1 + f(l·n̂)²] = 0.
    Take the future-directed root, then p_i = n̂_i + f l_i (v^t + l·n̂),
    E = v^t − f(v^t + l·n̂). (Reduces to p=n̂, E=1 at large r where f→0.)"""
    n = np.asarray(view_dir, float)
    n = n / np.linalg.norm(n)
    f, l = ks_f_l_yaxis(cam, M, a)
    ln = float(l @ n)
    A = f - 1.0
    B = 2.0 * f * ln
    C = 1.0 + f * ln * ln
    disc = max(B * B - 4.0 * A * C, 0.0)
    if abs(A) < 1e-9:           # f≈1 (only near horizon; camera never is)
        vt = -C / B
    else:
        vt = (-B - np.sqrt(disc)) / (2.0 * A)   # future-directed (vt→+1 as f→0)
    lv = vt + ln
    p = n + f * l * lv
    E = vt - f * lv
    return p, float(E)


if __name__ == "__main__":
    for a in (0.0, 0.5, 0.9, 0.998, 1.0):
        rp = photon_orbit_radius(1.0, a, True)
        rr = photon_orbit_radius(1.0, a, False)
        print(f"a={a:5.3f}  r+={horizon_radius(1.0, a):.4f}  "
              f"r_ph(pro)={rp:.4f}  r_ph(retro)={rr:.4f}  "
              f"xi(pro)={critical_xi(1.0, a, True):+.4f}  "
              f"xi(retro)={critical_xi(1.0, a, False):+.4f}")
