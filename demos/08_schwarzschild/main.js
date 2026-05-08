// Schwarzschild + leading-order Kerr ray-marcher with thick disk, blackbody
// emission, Doppler-aberrated sky, optional HDR background.
//
// Sliders correspond to real physical quantities:
//
//   Black hole:  M (mass), a/M (dimensionless spin)
//   Disk:        Mdot, r_in, r_out, optical depth tau, scale height H/R, rotation
//   Viewing:     inclination, camera boost beta (relativistic motion)
//
// Derived in-shader:
//   ISCO(a)               Bardeen-Press-Teukolsky 1972 closed-form
//   T(r)                  Shakura-Sunyaev: T ~ (r_in/r)^3/4 (1 - sqrt(r_in/r))^1/4
//   T_peak                T_peak = T_base * Mdot^1/4 / M^3/4
//   Blackbody RGB(T_K)    Tanner-Helland approximation
//   Stefan-Boltzmann      Emission ~ T^4
//   Keplerian rotation    omega(r) = sqrt(M / r^3)
//   Doppler beaming       beta(r) = sqrt(M/r), flux ~ D^4
//   Gravitational redshift  factor sqrt(1 - 2M/r)
//   Lense-Thirring drag   a_LT = (4 a M / r^3) * (J_hat x v)
//   Aberration / Doppler  full vector relativistic transform on sky direction
//
// Integration: classical RK4, adaptive step size dt(r).
// Disk: Gaussian-density slab, multi-pass front-to-back integration.
// Pipeline: shader -> HDR linear -> bloom -> ACES tonemap -> sRGB.

import * as THREE from "three";
import { OrbitControls }    from "three/addons/controls/OrbitControls.js";
import { EffectComposer }   from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass }       from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass }  from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass }       from "three/addons/postprocessing/OutputPass.js";

// -------- renderer --------
const canvas = document.getElementById("gl");
const renderer = new THREE.WebGLRenderer({
    canvas, antialias: true, powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping        = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.outputColorSpace    = THREE.SRGBColorSpace;

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 200);
const CAM_DISTANCE_DEFAULT = 18.0;
camera.position.set(0, 4.0, -17);
camera.lookAt(0, 0, 0);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0, 0);
controls.minDistance = 4;
controls.maxDistance = 80;
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.6;

// -------- helpers --------
const $ = (id) => document.getElementById(id);

function iscoKerr(aOverM, prograde = true) {
    const a  = aOverM;
    const a2 = a * a;
    const Z1 = 1 + Math.cbrt(1 - a2)
                 * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
    const Z2 = Math.sqrt(3 * a2 + Z1 * Z1);
    const sgn = prograde ? -1 : 1;
    return 3 + Z2 + sgn * Math.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2));
}

// Peak disk temperature in Kelvin given M, Mdot.
// Calibrated so default (M=1, Mdot=1) ~ 8000 K (yellow-white).
function peakTempK(M, Mdot) {
    return 8000 * Math.pow(Math.max(Mdot, 1e-3), 0.25)
                / Math.pow(Math.max(M, 1e-3),    0.75);
}

// -------- uniforms --------
const uniforms = {
    uResolution:  { value: new THREE.Vector2() },
    uCameraPos:   { value: new THREE.Vector3() },
    uCameraMat:   { value: new THREE.Matrix4() },
    uCameraBeta:  { value: new THREE.Vector3() },  // legacy; always (0,0,0) in gold-standard mode
    uCamRadialV:  { value: 0.0 },                  // signed; <0 = inward free-fall, =0 = static
    uTanFov:      { value: Math.tan((camera.fov * 0.5 * Math.PI) / 180) },
    uMass:        { value: 1.0 },
    uSpin:        { value: 0.0 },        // a/M
    uMaxSteps:    { value: 600 },
    uTime:        { value: 0 },
    uShowDisk:    { value: 1.0 },
    uMdot:        { value: 1.0 },
    uDiskInner:   { value: 6.0 },
    uDiskOuter:   { value: 14.0 },
    uDiskH:       { value: 0.08 },       // scale height H/R
    uSpinDir:     { value: 1.0 },
    uTau:         { value: 1.5 },
    uTpeakK:      { value: peakTempK(1.0, 1.0) },
    uHasSky:      { value: 0 },
    uSkyTex:      { value: null },
    uSkyRot:      { value: 0.0 },

    // Hotspot (orbiting Gaussian blob on Keplerian circular orbit)
    uHot:         { value: 0.0 },        // 0/1 enable
    uHotR:        { value: 6.0 },        // orbital radius (M)
    uHotW:        { value: 0.6 },        // gaussian sigma (M)
    uHotBrt:      { value: 1.0 },        // brightness multiplier
    uHotPhi:      { value: 0.0 },        // current azimuthal phase (JS-driven)

    // Bipolar jet
    uJet:         { value: 0.0 },        // 0/1 enable
    uJetBeta:     { value: 0.85 },       // bulk velocity / c
    uJetAngle:    { value: 0.15 },       // half-opening angle (rad)
    uJetLen:      { value: 25.0 },       // tip extent (M)
    uJetBrt:      { value: 1.0 },        // brightness

    // Diagnostic overlays
    uShowPRing:   { value: 0.0 },        // 0/1 photon-ring critical curve
    uBhScreen:    { value: new THREE.Vector2(0, 0) }, // BH center in NDC (JS-projected)
    uBhDist:      { value: 18.0 },       // camera-BH distance (for b_crit projection)

    // Spaghettification cube: up to 64 free-fall test particles (xyz + r_p
    // in .w for the gravitational-redshift factor). Inactive particles
    // flagged with w < 0. JS updates each frame.
    uParticles:   { value: Array.from({length: 64},
                              () => new THREE.Vector4(0, 0, 0, -1)) },
    // Signed radial velocity per particle (negative = inward), used for
    // the kinematic Doppler factor on the emitted light.
    uParticleVel: { value: new Float32Array(64) },
    uNumParticles:{ value: 0 },                    // count of active particles
    uPartBoxMin:  { value: new THREE.Vector3(0, 0, 0) },  // axis-aligned bbox in world coords
    uPartBoxMax:  { value: new THREE.Vector3(0, 0, 0) },  // for shader early-out
};

// -------- shaders --------
const vertexShader = /* glsl */ `
out vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position, 1.0); }
`;

const fragmentShader = /* glsl */ `
precision highp float;
out vec4 fragColor;
in  vec2 vUv;

uniform vec2  uResolution;
uniform vec3  uCameraPos;
uniform mat4  uCameraMat;
uniform vec3  uCameraBeta;     // legacy; gold-standard mode always sets (0,0,0)
uniform float uCamRadialV;     // signed; <0 = inward free-fall, =0 = static
uniform float uTanFov;
uniform float uMass;
uniform float uSpin;
uniform int   uMaxSteps;
uniform float uTime;
uniform float uShowDisk;
uniform float uMdot;
uniform float uDiskInner;
uniform float uDiskOuter;
uniform float uDiskH;
uniform float uSpinDir;
uniform float uTau;
uniform float uTpeakK;
uniform int   uHasSky;
uniform sampler2D uSkyTex;
uniform float uSkyRot;

uniform float uHot;
uniform float uHotR;
uniform float uHotW;
uniform float uHotBrt;
uniform float uHotPhi;

uniform float uJet;
uniform float uJetBeta;
uniform float uJetAngle;
uniform float uJetLen;
uniform float uJetBrt;

uniform float uShowPRing;
uniform vec2  uBhScreen;
uniform float uBhDist;

// Spaghettification particles. .xyz = world Cartesian position, .w = r_p
// (particle's BL radius for gravitational-redshift coloring). w<0 means
// inactive (don't render). Bounding box uniforms enable a per-step
// early-out so the inner loop only runs when the ray is near the cube.
// uParticleVel[i] = signed radial proper-velocity (negative = inward)
// for the kinematic Doppler factor in the particle's rest frame.
uniform vec4  uParticles[64];
uniform float uParticleVel[64];
uniform int   uNumParticles;
uniform vec3  uPartBoxMin;
uniform vec3  uPartBoxMax;

const float FAR = 90.0;
const int   MAX_DISK_HITS = 4;

// ============ hash ============
uint pcg(uint v) {
    v = v * 747796405u + 2891336453u;
    uint w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}
vec2 h22(uvec2 v) {
    uint a = pcg(v.x ^ pcg(v.y));
    uint b = pcg(a + 17u);
    return vec2(float(a), float(b)) / 4294967295.0;
}

// ============ relativistic aberration & Doppler ============
//   Camera moves with velocity beta (in BH rest frame).
//   nObs : direction of pixel ray (camera frame, "where camera looks").
//   Returns nRest : the corresponding direction in BH frame.
vec3 aberrate(vec3 nObs, vec3 beta) {
    float bb = dot(beta, beta);
    if (bb < 1e-10) return nObs;
    float b = sqrt(bb);
    vec3  bh = beta / b;
    float cosObs = dot(nObs, bh);
    float gamma  = 1.0 / sqrt(1.0 - bb);
    float cosRest = (cosObs + b) / (1.0 + b * cosObs);
    float sinRestFactor = 1.0 / (gamma * (1.0 + b * cosObs));
    vec3  nPerp = nObs - cosObs * bh;
    return normalize(cosRest * bh + sinRestFactor * nPerp);
}

// Doppler factor for light arriving at observer from direction nObs (camera
// frame, view direction looking outward). For light at rest-frame frequency
// nu_rest, observer sees nu_obs = D * nu_rest, intensity ~ D^4.
//   D = 1 / [gamma * (1 - beta . nObs)]
float dopplerFactor(vec3 nObs, vec3 beta) {
    float bb = dot(beta, beta);
    if (bb < 1e-10) return 1.0;
    float gamma = 1.0 / sqrt(1.0 - bb);
    return 1.0 / (gamma * (1.0 - dot(beta, nObs)));
}

// ============ Tanner-Helland blackbody (T in Kelvin -> linear sRGB) ============
vec3 blackbodyRGB(float T) {
    float t = clamp(T * 0.01, 10.0, 400.0);  // T/100 in [1000K, 40000K]
    float r, g, b;
    if (t <= 66.0) {
        r = 1.0;
        g = clamp(0.39008157876902 * log(t) - 0.6318414437886, 0.0, 1.0);
        b = (t <= 19.0) ? 0.0
            : clamp(0.5432067891101 * log(t - 10.0) - 1.196254089462, 0.0, 1.0);
    } else {
        r = clamp(1.29293618606275 * pow(t - 60.0, -0.1332047592), 0.0, 1.0);
        g = clamp(1.12989086089529 * pow(t - 60.0, -0.0755148492), 0.0, 1.0);
        b = 1.0;
    }
    // Approximate sRGB -> linear
    return pow(vec3(r, g, b), vec3(2.2));
}

// ============ procedural starfield (BH-frame direction in) ============
vec3 sampleSkyProcedural(vec3 dir) {
    float theta = atan(dir.z, dir.x);
    float phi   = asin(clamp(dir.y, -1.0, 1.0));

    vec3 col = vec3(0.004, 0.006, 0.012);

    float band_axis = dot(dir, normalize(vec3(0.35, 0.20, 0.92)));
    float band      = exp(-band_axis * band_axis * 22.0);
    col += vec3(0.18, 0.13, 0.09) * band * 0.55;
    float knot      = exp(-band_axis * band_axis * 60.0);
    col += vec3(0.30, 0.22, 0.14) * knot * 0.30;

    for (int oct = 0; oct < 3; oct++) {
        float scale = 60.0 * pow(2.0, float(oct));
        vec2  uv    = vec2(theta + 3.14159265, phi + 1.57079633) * scale;
        vec2  cell  = floor(uv);
        uvec2 ucell = uvec2(int(cell.x) + 1024, int(cell.y) + 1024)
                    + uvec2(oct * 7919);
        vec2  rh    = h22(ucell);
        float thr   = 0.9965 - 0.001 * float(oct);
        if (rh.x > thr) {
            vec2  frac   = fract(uv) - 0.5;
            float d      = length(frac);
            float bright = pow(rh.y, 5.0);
            float star   = bright * exp(-d * d * 60.0);
            // colour temperature 3000-12000K randomly
            float T_K   = mix(3000.0, 12000.0, rh.x);
            vec3  tint  = blackbodyRGB(T_K) * 1.6;
            col += tint * star;
        }
    }
    return col;
}

// HDR equirectangular lookup with optional yaw rotation
vec3 sampleSkyTexture(vec3 dir) {
    float theta = atan(dir.z, dir.x) + uSkyRot;
    float phi   = asin(clamp(dir.y, -1.0, 1.0));
    vec2 uv = vec2(0.5 + theta / (2.0 * 3.14159265),
                   0.5 + phi   / 3.14159265);
    vec3 c = texture(uSkyTex, uv).rgb;
    // sky textures are usually sRGB-encoded
    return pow(c, vec3(2.2));
}

vec3 sampleSky(vec3 dirRest, float omegaCam) {
    vec3 col = (uHasSky == 1) ? sampleSkyTexture(dirRest)
                              : sampleSkyProcedural(dirRest);
    // Gold-standard Doppler. Sky source is at r→∞ in the asymptotic-flat
    // frame, so its 4-velocity has only a t-component (= 1) and the photon
    // emission frequency equals the conserved energy E (= 1 in our gauge).
    // The camera measures ω_cam, so D = ω_cam / ω_emit = ω_cam.
    // Intensity boost = D^4. This single formula folds gravitational
    // redshift and kinematic Doppler together.
    return col * pow(omegaCam, 4.0);
}

// ============ disk physics ============
float diskTempNorm(float r) {
    // Shakura-Sunyaev thin-disk T(r)/T_max profile (peak ~ 1.0)
    float x = uDiskInner / r;
    if (x >= 1.0 - 1e-4) return 0.0;
    float t = pow(x, 0.75) * pow(max(1.0 - sqrt(x), 0.0), 0.25);
    return t * 2.05;
}

vec3 diskEmission(vec3 hit) {
    float r   = length(hit.xz);
    float phi = atan(hit.z, hit.x);

    // Keplerian swirl
    float omega = uSpinDir * sqrt(uMass) / max(pow(r, 1.5), 0.5);
    float a = phi + omega * uTime;

    float n  = 0.5 + 0.5 * sin(a *  4.0 + r * 1.4 + 1.6 * sin(a * 2.0 - r * 0.4));
    float n2 = 0.5 + 0.5 * sin(a * 11.0 - r * 4.0 + uTime * 0.3);
    float n3 = 0.5 + 0.5 * sin(a * 23.0 + r * 9.0);
    float tex = pow(clamp(n * (0.6 + 0.4 * n2) * (0.7 + 0.3 * n3), 0.0, 1.0), 0.55);

    float tNorm = diskTempNorm(r);
    float T_K   = tNorm * uTpeakK;
    vec3  bbRGB = blackbodyRGB(T_K);

    // Stefan-Boltzmann scaling, normalized to a reference T_ref = 6000K
    float SB = pow(T_K / 6000.0, 4.0);

    // surface emission
    vec3 col = bbRGB * SB * (0.30 + 1.7 * tex) * uMdot;

    // soft inner / outer edges
    float fade = smoothstep(uDiskInner, uDiskInner * 1.10, r) *
                 smoothstep(uDiskOuter, uDiskOuter * 0.94, r);
    col *= fade;
    return col;
}

// ============ Cartesian Schwarzschild + leading-order Lense-Thirring drag ============
//
// State: x (3D position, BH-centered Cartesian), p (3D momentum). For null
// geodesics with E=1 gauge, |p| → 1 at infinity. World convention: y is the
// BH spin axis; disk lies in the xz plane.
//
//   dx/dλ = p
//   dp/dλ = -3 M h² x / r⁵           Schwarzschild gravitational pull
//          + (4 a M / r³) (ŷ × p)    leading-order Lense-Thirring drag
//
// where h² = |x × p|² (conserved in pure Schwarzschild). The Schwarzschild
// term reproduces the orbit equation d²u/dφ² + u = 3Mu² (u = 1/r) — verified
// by giving |a| = 1/(3M) at the photon sphere r = 3M. The LT term is the
// leading frame-drag perturbation; correct to O(a/M) for any astrophysical
// spin (a/M ≲ 0.9). At near-extremal spin, strong-field Kerr corrections
// (not modeled) become noticeable.
//
// Cartesian state has no polar singularity, unlike Boyer-Lindquist.

void geoDeriv(vec3 x, vec3 p, float M, float a,
              out vec3 dx, out vec3 dp)
{
    float r2 = dot(x, x);
    float r  = sqrt(r2);
    float r5 = r2 * r2 * r;
    vec3  cxp = cross(x, p);
    float h2  = dot(cxp, cxp);

    dx = p;
    dp = -3.0 * M * h2 * x / r5;

    // Lense-Thirring frame drag (BH spin along +y).
    if (abs(a) > 1e-6) {
        dp += (4.0 * a * M / (r2 * r)) * cross(vec3(0.0, 1.0, 0.0), p);
    }
}

void rk4StepCart(inout vec3 x, inout vec3 p,
                 float M, float a, float dl)
{
    vec3 k1x, k1p; geoDeriv(x,              p,              M, a, k1x, k1p);
    vec3 k2x, k2p; geoDeriv(x + 0.5*dl*k1x, p + 0.5*dl*k1p, M, a, k2x, k2p);
    vec3 k3x, k3p; geoDeriv(x + 0.5*dl*k2x, p + 0.5*dl*k2p, M, a, k3x, k3p);
    vec3 k4x, k4p; geoDeriv(x + dl*k3x,     p + dl*k3p,     M, a, k4x, k4p);

    x += dl * (k1x + 2.0*k2x + 2.0*k3x + k4x) / 6.0;
    p += dl * (k1p + 2.0*k2p + 2.0*k3p + k4p) / 6.0;
}

// ============ ray ============

void main() {
    vec2  frag   = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0;
    float aspect = uResolution.x / uResolution.y;
    vec3  dirCam = normalize(vec3(frag.x * aspect * uTanFov, frag.y * uTanFov, -1.0));
    vec3  nView  = normalize((uCameraMat * vec4(dirCam, 0.0)).xyz);

    // ============================================================
    //   GOLD-STANDARD TETRAD-BASED RAY CONSTRUCTION (Schwarzschild)
    // ============================================================
    //
    // Build the photon's 4-momentum in the BL coord basis from the camera's
    // local Lorentz frame (its tetrad). This replaces the flat-space SR
    // aberration with a curved-spacetime construction. Two observer types
    // are supported: static (uCamRadialV=0) and radial free-fall
    // (uCamRadialV<0, valid for Schwarzschild only). Tangential boost is
    // not modelled here.
    //
    // Conventions:
    //   - Camera position is uCameraPos in world Cartesian, BH at origin.
    //   - BL spherical (r0, θ0, φ0) extracted from Cartesian.
    //   - Spatial tetrad orthonormal {e_1, e_2, e_3} = {r̂, θ̂, φ̂}.
    //   - Gauge: conserved E = −k_t = 1, so |k|→1 at infinity (matches the
    //     existing Cartesian integrator's parameterisation).
    //
    // For a freely-falling observer with proper-velocity component vRad
    // along +r̂ (so vRad < 0 means inward), the tetrad is built by Lorentz
    // boosting the static tetrad:
    //   e_0_FF^μ = γ(e_0_st^μ + vRad · e_1_st^μ)
    //   e_1_FF^μ = γ(e_1_st^μ + vRad · e_0_st^μ)
    //   e_2_FF^μ = e_2_st^μ,  e_3_FF^μ = e_3_st^μ
    // Static-observer tetrad in Schwarzschild:
    //   e_0_st = (1/√f, 0, 0, 0),  e_1_st = (0, √f, 0, 0)
    //   e_2_st = (0, 0, 1/r, 0),   e_3_st = (0, 0, 0, 1/(r sinθ))
    //
    // Photon 4-momentum in tetrad (with ω_cam = 1 gauge):
    //   k^α_tet = (1, n^1, n^2, n^3)
    // Convert to coord basis k^μ = e_α^μ k^α_tet, then rescale by 1/E_local
    // to enforce the integrator's E=1 gauge.

    // BL position from Cartesian
    float r0  = length(uCameraPos);
    float ct0 = uCameraPos.y / max(r0, 1e-3);
    float st0 = sqrt(max(1.0 - ct0 * ct0, 0.0));
    float r_xz = sqrt(max(uCameraPos.x * uCameraPos.x +
                          uCameraPos.z * uCameraPos.z, 1e-6));
    float cp0 = uCameraPos.x / r_xz;
    float sp0 = uCameraPos.z / r_xz;

    // Spatial tetrad basis vectors in world Cartesian (orthonormal)
    vec3 r_hat  = vec3(st0 * cp0, ct0,  st0 * sp0);
    vec3 th_hat = vec3(ct0 * cp0, -st0, ct0 * sp0);
    vec3 ph_hat = vec3(-sp0,        0.0, cp0);

    // Project pixel direction onto the spatial tetrad
    float n1 = dot(nView, r_hat);
    float n2 = dot(nView, th_hat);
    float n3 = dot(nView, ph_hat);

    // Schwarzschild f at the camera (clamped slightly above the horizon
    // to avoid the static-observer singularity).
    float fCam     = 1.0 - 2.0 * uMass / max(r0, uMass * 2.05);
    float fCamSqrt = sqrt(max(fCam, 1e-4));

    // Camera radial velocity (signed; sign convention: +r̂ direction, so
    // <0 means inward free-fall). Clamped strictly inside the light cone.
    float vRad = clamp(uCamRadialV, -0.998, 0.998);
    float gam  = 1.0 / sqrt(max(1.0 - vRad * vRad, 1e-6));

    // Photon 4-momentum in coord basis, ω_cam = 1 gauge:
    //   k^t = γ(1 + n1 vRad)/√f
    //   k^r = γ √f (vRad + n1)
    //   k^θ = n2/r
    //   k^φ = n3/(r sinθ)
    // Conserved E_local = −k_t = f · k^t = √f · γ · (1 + n1 vRad).
    float E_local = fCamSqrt * gam * (1.0 + n1 * vRad);
    float invE    = 1.0 / max(abs(E_local), 1e-6);

    // Rescale to the integrator's E = 1 gauge.
    float kT  = (gam * (1.0 + n1 * vRad) / fCamSqrt) * invE;   // = 1/fCam
    float kR  = (gam * fCamSqrt * (vRad + n1))       * invE;
    float kTh = (n2 / max(r0, 1e-3))                 * invE;
    float kPh = (n3 / max(r0 * max(st0, 0.05), 1e-3)) * invE;

    // Convert spatial parts (k^r, k^θ, k^φ) to Cartesian (k^x, k^y, k^z)
    // using the standard spherical→Cartesian Jacobian.
    float pX = st0 * cp0 * kR + r0 * ct0 * cp0 * kTh - r0 * st0 * sp0 * kPh;
    float pY = ct0 * kR        - r0 * st0 * kTh;
    float pZ = st0 * sp0 * kR + r0 * ct0 * sp0 * kTh + r0 * st0 * cp0 * kPh;
    vec3 dir = vec3(pX, pY, pZ);

    // Camera-frame frequency at the pixel (in E=1 gauge): ω_cam = 1/E_local.
    // This carries through to every Doppler computation downstream.
    float omegaCam = invE;

    // Schwarzschild only in gold-standard mode (Kerr spin gated).
    float a = 0.0;

    // Cartesian state: position x (BH-centred), momentum p (= k^μ spatial
    // parts in Cartesian). The integrator uses E=1 affine parameterisation.
    vec3 x = uCameraPos;
    vec3 p = dir;

    float Rplus = 2.0 * uMass;

    bool  captured = false;
    bool  escaped  = false;
    vec3  accumCol = vec3(0.0);
    float accumA   = 0.0;
    int   diskHits = 0;

    for (int i = 0; i < 2000; i++) {
        if (i >= uMaxSteps) break;
        if (accumA > 0.99)  break;

        float r = length(x);
        if (r < Rplus * 1.001) { captured = true; break; }
        if (r > FAR)           { escaped  = true; break; }

        // Adaptive step: smaller near horizon, larger far away.
        float dl = clamp(r * 0.10, 0.04, 0.7);

        vec3 prev_x = x;
        vec3 prev_p = p;

        rk4StepCart(x, p, uMass, a, dl);

        // Equatorial-plane crossing: disk and hotspot both live in y=0.
        // diskHits counter gates MAX_DISK_HITS budget so lensed re-crossings
        // don't infinitely stack.
        bool wantEquatorial = (uShowDisk > 0.5 || uHot > 0.5);
        if (wantEquatorial && diskHits < MAX_DISK_HITS &&
            prev_x.y * x.y < 0.0)
        {
            float fr   = prev_x.y / (prev_x.y - x.y);
            fr         = clamp(fr, 0.0, 1.0);
            vec3  hit  = mix(prev_x, x, fr);
            vec3  pHit = mix(prev_p, p, fr);
            float rd   = length(hit.xz);

            // Doppler/beaming: photon vs disk-fluid Keplerian flow.
            // L_z (about BH spin axis ŷ) = (hit × pHit)·ŷ.
            float Lz = hit.z * pHit.x - hit.x * pHit.z;

            // Disk element on circular Keplerian orbit (Kerr formula retained
            // as a leading-order proxy; exact in Schwarzschild):
            //   Ω = 1/(r^{3/2}/√M ± a)   (prograde +, retrograde −)
            //   u^t = 1/√(1 − 3M/r + 2a√(M/r³))
            float Omega  = uSpinDir / (pow(rd, 1.5)/sqrt(uMass) + uSpinDir * a);
            float gtt0   = 1.0 - 2.0*uMass/rd;
            float gtp0   = -2.0*uMass*a/rd;
            float gpp0   = (rd*rd + a*a + 2.0*uMass*a*a/rd);
            float denomU = max(gtt0 - 2.0*Omega*(-gtp0) - Omega*Omega*gpp0, 1e-3);
            float u_t    = 1.0 / sqrt(denomU);
            float u_p    = Omega * u_t;

            // Photon's frequency in disk-fluid frame: ω_em = −p_μ u^μ
            // = E·u_t − L_z·u^φ, with E = 1 (integrator gauge). Camera-frame
            // frequency is omegaCam (computed from the tetrad ray init).
            // D = ω_cam / ω_em folds gravitational + kinematic together.
            float omegaEm = u_t - Lz * u_p;
            float D       = omegaCam / max(omegaEm, 1e-3);
            float boost   = pow(D, 4.0);

            // Track whether anything actually emitted this crossing — only
            // burn a MAX_DISK_HITS slot when something rendered. Otherwise
            // lensed rays passing through the inner shadow gap waste hits
            // and photon-ring secondary images get truncated.
            bool emitted = false;

            // ----- Disk surface emission -----
            if (uShowDisk > 0.5 && rd > uDiskInner && rd < uDiskOuter) {
                // Disk thickness slab path length: H / |p̂_y|.
                float vyAbs  = abs(normalize(pHit).y);
                float H      = uDiskH * rd;
                float pathH  = H / max(vyAbs, 0.04);
                float tauEff = uTau * pathH;
                float alpha  = 1.0 - exp(-tauEff);

                vec3 emission = diskEmission(hit) * boost;
                accumCol += emission * alpha * (1.0 - accumA);
                accumA   += alpha * (1.0 - accumA);
                emitted = true;
            }

            // ----- Orbiting hotspot (Gaussian on Keplerian orbit) -----
            // Hotspot azimuthal phase uHotPhi is JS-driven (Keplerian Ω at uHotR).
            if (uHot > 0.5) {
                float phH    = atan(hit.z, hit.x);
                float dr     = rd - uHotR;
                float dphi   = phH - uHotPhi;
                dphi = mod(dphi + 3.14159265, 6.28318530) - 3.14159265;
                float arc    = uHotR * dphi;
                float dist2  = dr*dr + arc*arc;
                float w2     = max(uHotW * uHotW, 1e-3);
                float gauss  = exp(-dist2 / w2);
                if (gauss > 0.001) {
                    // Hot blue-white: a recently magnetised plasma blob.
                    vec3 hotCol = blackbodyRGB(18000.0) *
                                  (8.0 * uHotBrt) * gauss * boost;
                    accumCol += hotCol * (1.0 - accumA);
                    // Don't fully consume accumA so the disk behind still shows.
                    accumA   += clamp(0.6 * gauss, 0.0, 1.0) * (1.0 - accumA);
                    emitted = true;
                }
            }

            if (emitted) diskHits++;
        }

        // ----- Bipolar jet (volumetric) -----
        // Conical plasma along ±y. Density gaussian off-axis, Doppler-boosted
        // by the bulk velocity along sign(y)·ŷ. Sampled along the ray path.
        if (uJet > 0.5) {
            float pLen = length(x);
            float yAbs = abs(x.y);
            if (pLen < uJetLen && yAbs > 0.4 * uMass) {
                float cosTh   = yAbs / max(pLen, 1e-3);
                float coneCos = cos(uJetAngle);
                if (cosTh > coneCos) {
                    float dlPhys = length(x - prev_x);
                    float rPerp  = sqrt(max(pLen*pLen - yAbs*yAbs, 0.0));
                    // off-axis half-width grows with height (cone)
                    float sig    = max(uJetAngle * yAbs * 0.7, 0.25);
                    float density = exp(-rPerp*rPerp / (sig*sig));
                    // density tapers toward the tip
                    density *= smoothstep(uJetLen, 0.35 * uJetLen, pLen);
                    // and ramps up away from the very base (avoid disk overlap)
                    density *= smoothstep(0.4 * uMass, 1.5 * uMass, yAbs);

                    // Photon direction in Cartesian (forward along λ).
                    vec3 photonDir = normalize(x - prev_x);
                    vec3 nObs       = -photonDir;
                    vec3 betaJet    = vec3(0.0, sign(x.y) * uJetBeta, 0.0);
                    float bb        = uJetBeta * uJetBeta;
                    float gJ        = 1.0 / sqrt(max(1.0 - bb, 1e-4));
                    float Djet      = 1.0 / max(gJ * (1.0 - dot(betaJet, nObs)), 1e-3);
                    float jBoost    = pow(Djet, 4.0);

                    // Synchrotron-ish blue. Prefactor 0.05 is tuned so the
                    // optically-thick saturation brightness (∝ jBoost ÷ alpha)
                    // sits in the same HDR range as the disk peak, so neither
                    // washes out the other after ACES + bloom. Saturated blue
                    // (R << B) so the colour survives bloom desaturation.
                    vec3 jetCol = vec3(0.25, 0.55, 1.0);
                    float emitScale = density * uJetBrt * dlPhys;
                    vec3 emit = jetCol * jBoost * emitScale * 0.05;
                    float jetA = clamp(emitScale * 0.18, 0.0, 0.9);

                    accumCol += emit * (1.0 - accumA);
                    accumA   += jetA * (1.0 - accumA);
                }
            }
        }

        // ----- Spaghettification particles + wireframe edges (lensed) -----
        // Bounding-box early-out keeps the inner loops cheap: rays that
        // don't pass through the cube's bbox skip both loops entirely.
        // Coloring: blackbody at T_obs = T_emit·√f_p, where f_p = 1 - 2M/r_p
        // is the gravitational redshift factor at the particle. Camera-side
        // factor (omegaCam) folds in the camera tetrad's blueshift/Doppler.
        // Wireframe edges connect the 8 corners of the 4×4×4 grid via 12
        // straight line segments. As the cube falls, the 4 radial-direction
        // edges visibly stretch (spaghettification) while the 8 tangential
        // edges shrink — exactly the GR tidal pattern.
        if (uNumParticles > 0 &&
            all(greaterThanEqual(x, uPartBoxMin)) &&
            all(lessThanEqual(x, uPartBoxMax)))
        {
            float dlPhys = length(x - prev_x);

            // Photon direction at this step (used for kinematic Doppler
            // computation in the particle's rest frame).
            vec3  photonDir = normalize(x - prev_x);

            // ---- Particle dots ----
            for (int i = 0; i < 64; i++) {
                if (i >= uNumParticles) break;
                vec4  pt   = uParticles[i];
                if (pt.w < 0.0) continue;
                vec3  d    = x - pt.xyz;
                float d2   = dot(d, d);
                if (d2 > 0.25) continue;
                // σ² = 0.018 (σ ≈ 0.135 M): sharper individual dots.
                float gauss = exp(-d2 / 0.018);
                if (gauss < 0.002) continue;
                float fP    = 1.0 - 2.0 * uMass / max(pt.w, uMass * 2.05);
                float fPSqrt = sqrt(max(fP, 1e-3));
                // Kinematic Doppler at the emitter:
                //   ω_emit_p = γ_p (1 − β_p·n̂) · (1/√f_p)
                // β_p = v_p · r̂_p (signed; negative = inward fall).
                float v_p     = clamp(uParticleVel[i], -0.99, 0.99);
                vec3  rHatP   = pt.xyz / max(length(pt.xyz), 1e-3);
                float gammaP  = 1.0 / sqrt(max(1.0 - v_p*v_p, 1e-4));
                float dopShift = gammaP * (1.0 - v_p * dot(rHatP, photonDir));
                // Camera-side D = ω_cam / ω_emit_p folds it all together.
                float D       = omegaCam * fPSqrt / max(dopShift, 1e-3);
                float boost   = pow(D, 4.0);
                // Observed temperature uses the same factor (T_obs ∝ D · T_emit).
                float T_obs   = 14000.0 * D;
                T_obs = clamp(T_obs, 600.0, 30000.0);
                vec3  emitCol = blackbodyRGB(T_obs);
                vec3  emission = emitCol * boost * gauss * dlPhys * 0.7;
                float partA    = clamp(gauss * dlPhys * 0.5, 0.0, 0.9);
                accumCol += emission * (1.0 - accumA);
                accumA   += partA * (1.0 - accumA);
            }

            // ---- Wireframe edges (12 segments connecting cube corners) ----
            // Indexing: idx = k*16 + j*4 + i, with k = radial, i,j tangential.
            // Corner indices: 0,3,12,15,48,51,60,63.
            for (int e = 0; e < 12; e++) {
                int ia, ib;
                if      (e == 0) { ia=0;  ib=3;  }      // i-edges (tangential, shrink)
                else if (e == 1) { ia=12; ib=15; }
                else if (e == 2) { ia=48; ib=51; }
                else if (e == 3) { ia=60; ib=63; }
                else if (e == 4) { ia=0;  ib=12; }      // j-edges (tangential, shrink)
                else if (e == 5) { ia=3;  ib=15; }
                else if (e == 6) { ia=48; ib=60; }
                else if (e == 7) { ia=51; ib=63; }
                else if (e == 8) { ia=0;  ib=48; }      // k-edges (radial, stretch)
                else if (e == 9) { ia=3;  ib=51; }
                else if (e ==10) { ia=12; ib=60; }
                else             { ia=15; ib=63; }
                vec4 a = uParticles[ia];
                vec4 b = uParticles[ib];
                if (a.w < 0.0 || b.w < 0.0) continue;
                vec3 ab = b.xyz - a.xyz;
                float ab2 = dot(ab, ab);
                if (ab2 < 1e-4) continue;
                vec3 ax = x - a.xyz;
                float t = clamp(dot(ax, ab) / ab2, 0.0, 1.0);
                vec3 closest = a.xyz + t * ab;
                vec3 dE = x - closest;
                float dE2 = dot(dE, dE);
                if (dE2 > 0.10) continue;          // tube radius ≈ 0.32 M
                // Tube cross-section: σ² = 0.006 (σ ≈ 0.077 M). Crisp lines.
                float lineG = exp(-dE2 / 0.006);
                if (lineG < 0.05) continue;
                // Average corner radius and velocity → Doppler/redshift.
                float rE     = 0.5 * (a.w + b.w);
                float v_pE   = clamp(0.5 * (uParticleVel[ia] + uParticleVel[ib]),
                                     -0.99, 0.99);
                float fE     = 1.0 - 2.0 * uMass / max(rE, uMass * 2.05);
                float fESqrt = sqrt(max(fE, 1e-3));
                vec3  rHatE  = closest / max(length(closest), 1e-3);
                float gammaE = 1.0 / sqrt(max(1.0 - v_pE*v_pE, 1e-4));
                float dopE   = gammaE * (1.0 - v_pE * dot(rHatE, photonDir));
                float D_E    = omegaCam * fESqrt / max(dopE, 1e-3);
                float boostE = pow(D_E, 4.0);
                float T_obsE = clamp(14000.0 * D_E, 600.0, 30000.0);
                vec3  edgeCol = blackbodyRGB(T_obsE);
                vec3  emitE   = edgeCol * boostE * lineG * dlPhys * 0.45;
                float edgeA   = clamp(lineG * dlPhys * 0.4, 0.0, 0.85);
                accumCol += emitE * (1.0 - accumA);
                accumA   += edgeA * (1.0 - accumA);
            }
        }
    }

    vec3 background;
    if (captured) {
        background = vec3(0.0);
    } else {
        // Either properly escaped, or ran out of step budget. In the
        // ran-out case, treat the ray as escaped if we're well outside the
        // strong-field region — keeps lowering uMaxSteps from punching a
        // black panel through the middle.
        bool treatAsEscaped = escaped || (length(x) > Rplus * 1.5);
        if (treatAsEscaped) {
            background = sampleSky(normalize(p), omegaCam);
        } else {
            background = vec3(0.0);
        }
    }

    vec3 col = accumCol + background * (1.0 - accumA);

    // ----- Photon-ring overlay (analytic critical curve) -----
    // Ring sits at b_crit = 3√3 M (Schwarzschild approx; Kerr breaks the
    // symmetry slightly but this is a teaching aid not a fit). Projected to
    // pixel space using camera-BH distance and FOV; the locus is a true
    // circle in pixel space because perpendicular pixel distance scales by
    // the same (H/2)/uTanFov factor for x and y.
    if (uShowPRing > 0.5) {
        vec2 pixHere = gl_FragCoord.xy;
        vec2 pixCent = (uBhScreen * 0.5 + 0.5) * uResolution;
        float pixD   = length(pixHere - pixCent);
        float bcrit  = 3.0 * sqrt(3.0) * uMass;
        float ringPix = (bcrit / max(uBhDist, 1.0)) *
                        (uResolution.y * 0.5) / max(uTanFov, 1e-3);
        // Two strokes: the n=∞ critical curve (bright) and a hint of the
        // n=1 echo just inside (very faint). Explicit squaring instead of
        // pow(neg, 2.0), which is undefined per GLSL spec.
        float w1 = 1.6;
        float d1 = (pixD - ringPix)         / w1;
        float d2 = (pixD - ringPix * 0.985) / 1.0;
        float ring1 = exp(-d1 * d1);
        float ring2 = exp(-d2 * d2) * 0.35;
        col += vec3(0.55, 0.85, 1.0) * (ring1 * 1.1 + ring2 * 0.6);
    }

    fragColor = vec4(col, 1.0);
}
`;

const material = new THREE.ShaderMaterial({
    uniforms,
    vertexShader,
    fragmentShader,
    glslVersion: THREE.GLSL3,
    depthTest: false,
    depthWrite: false,
});
const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
scene.add(quad);

// -------- post-processing --------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.55, 0.65, 1.05,
);
composer.addPass(bloomPass);
composer.addPass(new OutputPass());

// -------- HDR sky loading --------
function tryLoadSkyTexture(url) {
    if (!url) return;
    const loader = new THREE.TextureLoader();
    loader.crossOrigin = "anonymous";
    loader.load(url,
        (tex) => {
            tex.colorSpace = THREE.SRGBColorSpace;
            tex.minFilter = THREE.LinearFilter;
            tex.magFilter = THREE.LinearFilter;
            tex.wrapS     = THREE.RepeatWrapping;
            tex.wrapT     = THREE.ClampToEdgeWrapping;
            uniforms.uSkyTex.value = tex;
            uniforms.uHasSky.value = 1;
            const lab = $("vSkyStat");
            if (lab) lab.textContent = "loaded";
        },
        undefined,
        (err) => {
            const lab = $("vSkyStat");
            if (lab) lab.textContent = "load failed";
            console.warn("sky texture failed", err);
        }
    );
}

// -------- UI bindings --------
function bindRange(id, valueId, uniform, fmt = (v) => v.toFixed(2), onChange = null) {
    const r = $(id), v = $(valueId);
    const apply = () => {
        const x = parseFloat(r.value);
        uniform.value = x;
        v.textContent = fmt(x);
        if (onChange) onChange(x);
    };
    r.addEventListener("input", apply);
    apply();
}
function bindCheck(id, uniform, onTrue = 1.0, onFalse = 0.0, onChange = null) {
    const c = $(id);
    const apply = () => {
        uniform.value = c.checked ? onTrue : onFalse;
        if (onChange) onChange(c.checked);
    };
    c.addEventListener("change", apply);
    apply();
}

function refreshDerived() {
    const M    = uniforms.uMass.value;
    const a    = uniforms.uSpin.value;
    const Mdot = uniforms.uMdot.value;
    $("vRs").textContent     = (2 * M).toFixed(2);
    $("vBcrit").textContent  = (3 * Math.sqrt(3) * M).toFixed(3);
    const ip = iscoKerr(a, true)  * M;
    const ir = iscoKerr(a, false) * M;
    $("vIscoP").textContent  = ip.toFixed(2);
    $("vIscoR").textContent  = ir.toFixed(2);
    const Tp = peakTempK(M, Mdot);
    uniforms.uTpeakK.value = Tp;
    $("vTpeak").textContent = (Tp / 1000).toFixed(1) + " kK";
}

bindRange("rMass", "vMass", uniforms.uMass, (v) => v.toFixed(2), refreshDerived);
bindRange("rSpin", "vSpin", uniforms.uSpin, (v) => v.toFixed(3), refreshDerived);
bindCheck("cDisk", uniforms.uShowDisk);
bindRange("rMdot", "vMdot", uniforms.uMdot,      (v) => v.toFixed(2), refreshDerived);
bindRange("rDi",   "vDi",   uniforms.uDiskInner, (v) => v.toFixed(2));
bindRange("rDo",   "vDo",   uniforms.uDiskOuter, (v) => v.toFixed(1));
bindRange("rTau",  "vTau",  uniforms.uTau,       (v) => v.toFixed(2));
bindRange("rDH",   "vDH",   uniforms.uDiskH,     (v) => v.toFixed(2));
bindCheck("cSpin", uniforms.uSpinDir, 1.0, -1.0);
$("bSnapISCO").addEventListener("click", () => {
    const ip = iscoKerr(uniforms.uSpin.value, true) * uniforms.uMass.value;
    $("rDi").value = ip.toFixed(2);
    $("rDi").dispatchEvent(new Event("input"));
});

// Orbiting hotspot
bindCheck("cHot",  uniforms.uHot);
bindRange("rHotR", "vHotR", uniforms.uHotR, (v) => v.toFixed(1));
bindRange("rHotW", "vHotW", uniforms.uHotW);
bindRange("rHotB", "vHotB", uniforms.uHotBrt);

// Bipolar jet
bindCheck("cJet",   uniforms.uJet);
bindRange("rJetB",  "vJetB",  uniforms.uJetBeta);
bindRange("rJetA",  "vJetA",  uniforms.uJetAngle, (v) => v.toFixed(3));
bindRange("rJetL",  "vJetL",  uniforms.uJetLen,   (v) => v.toFixed(0));
bindRange("rJetBr", "vJetBr", uniforms.uJetBrt);

// Diagnostic overlays
bindCheck("cPRing", uniforms.uShowPRing);

// Inclination
const inclState = { value: 70 };
bindRange("rIncl", "vIncl", inclState, (v) => v.toFixed(0) + "°", setInclination);
function setInclination(deg) {
    const offset = camera.position.clone().sub(controls.target);
    const dist   = offset.length() || CAM_DISTANCE_DEFAULT;
    const phi    = THREE.MathUtils.degToRad(deg);
    const lon = Math.atan2(offset.z, offset.x);
    const x = dist * Math.sin(phi) * Math.cos(lon);
    const z = dist * Math.sin(phi) * Math.sin(lon);
    const y = dist * Math.cos(phi);
    camera.position.set(x, y, z);
    controls.update();
}

// (Camera boost slider removed in gold-standard mode — observer 4-velocity
// is now driven by uCamRadialV from the plunge state, not a manual knob.)

// Auto-orbit / FOV / reset
bindCheck("cAuto", { value: 0 }, 1.0, 0.0, (on) => { controls.autoRotate = on; });
bindRange("rAo", "vAo", { value: 0 }, (v) => v.toFixed(2),
          (v) => { controls.autoRotateSpeed = v; });
bindRange("rFov", "vFov", { value: 0 }, (v) => v.toFixed(0), (v) => {
    camera.fov = v;
    camera.updateProjectionMatrix();
    uniforms.uTanFov.value = Math.tan((v * 0.5 * Math.PI) / 180);
});
$("bReset").addEventListener("click", () => {
    plunge.active = false;
    plunge.phase  = "idle";
    controls.enabled = true;
    camera.position.set(0, 4.0, -17);
    controls.target.set(0, 0, 0);
    // Restore FOV from slider in case plunge widened it.
    const fovSlider = parseFloat($("rFov").value);
    camera.fov = fovSlider;
    camera.updateProjectionMatrix();
    uniforms.uTanFov.value = Math.tan((fovSlider * 0.5 * Math.PI) / 180);
    uniforms.uCameraBeta.value.set(0, 0, 0);
    uniforms.uCamRadialV.value = 0.0;
    renderer.toneMappingExposure = 1.0;
    if (fadeEl) {
        fadeEl.style.transition = "opacity 0.4s linear";
        fadeEl.style.opacity    = "0";
    }
    if (hudEl) {
        hudEl.style.transition = "opacity 0.3s linear";
        hudEl.style.opacity    = "0";
    }
    $("rIncl").value = 70;  $("rIncl").dispatchEvent(new Event("input"));
});

// Camera-plunge state. Free-fall radial geodesic in Schwarzschild, with the
// physical proper-time speed v_r(r) = sqrt(2M/r) (fall from rest at infinity).
// At r=18M that's β ≈ 0.33, accelerating to ≈1 at the horizon. We add a
// modest tangential component (corkscrew) so the disk swings around the
// camera, push FOV wider for "engulfed" feel, boost exposure to mimic
// blueshift intensity gain (γ²), and overlay a live β/γ/r HUD. Black fade on
// horizon crossing, then auto-restore.
// ============ Spaghettification cube ============
// 4×4×4 = 64 free-fall test particles arranged as a cube at spawn. Each
// particle integrates its own radial geodesic from rest:
//   d²r/dτ² = -M/r²   (Schwarzschild proper-time radial geodesic equals
//                       Newton in this gauge — verified from the metric)
// Velocity-Verlet / semi-implicit Euler (acceleration first, then position)
// so the integration kicks off correctly from v=0. Inner particles
// accelerate faster → cube stretches radially as it falls.
const spag = {
    particles:  [],          // {pos: Vec3, dir: Vec3, r: float, r0: float, v_r: float}
    active:     false,
    timeAccel: 10.0,         // proper-time / wall-second
};

// Click-to-drop: cast a ray from the camera through the clicked pixel;
// spawn a 4×4×4 cube of test particles centered at the click point.
function spawnSpaghetti(spawnCenter) {
    spag.particles.length = 0;
    spag.active = true;

    // Orthonormal basis at spawn: radialOut = +r̂, t1/t2 perpendicular.
    const radialOut = spawnCenter.clone().normalize();
    const radialIn  = radialOut.clone().multiplyScalar(-1);
    let helper = new THREE.Vector3(0, 1, 0);
    if (Math.abs(radialOut.dot(helper)) > 0.95) helper.set(1, 0, 0);
    const t1 = new THREE.Vector3().crossVectors(radialOut, helper).normalize();
    const t2 = new THREE.Vector3().crossVectors(radialOut, t1).normalize();

    // 4×4×4 = 64 particles, spacing 0.5 M → cube total side ≈ 1.5 M.
    // Each axis: index k in [0..3], offset = (k - 1.5) * spacing.
    const spacing = 0.5;
    for (let k = 0; k < 4; k++) {
        for (let j = 0; j < 4; j++) {
            for (let i = 0; i < 4; i++) {
                const radOff = (k - 1.5) * spacing;
                const t1Off  = (i - 1.5) * spacing;
                const t2Off  = (j - 1.5) * spacing;
                const pos = spawnCenter.clone()
                    .add(radialIn.clone().multiplyScalar(radOff))
                    .add(t1.clone().multiplyScalar(t1Off))
                    .add(t2.clone().multiplyScalar(t2Off));
                const r = pos.length();
                spag.particles.push({
                    pos: pos,
                    r:   r,
                    r0:  r,                       // for diagnostic / display only
                    v_r: 0,                       // start at rest
                    dir: pos.clone().normalize(), // fixed angular direction
                });
            }
        }
    }
}

// Pure-radial gold-standard plunge. Free-fall from rest at infinity,
// dr/dτ = −sqrt(2M/r). Updates uCamRadialV each frame so the shader's tetrad
// uses the actual physical 4-velocity of the camera. No corkscrew, no FOV
// ramp, no exposure boost, no look-back: the brightness/aberration changes
// you see come entirely from the gold-standard tetrad math.
const plunge = {
    active:       false,
    phase:        "idle",          // "falling" | "fading" | "idle"
    phaseStart:   0,
    fadeDuration: 1.0,
    holdDur:      0.6,
    startDir:     new THREE.Vector3(),
    startR:       0,
    currentR:     0,
    targetR:      0,
    timeAccel:    6.0,             // proper-time / wall-second
};

// Radial-redshift fade overlay for the horizon-crossing payoff. We ramp from
// transparent → deep red → black: physically motivated because everything
// the infalling observer sees from the receding hemisphere blueshifts away
// to nothing, and the forward sky is the last bright thing as the shadow
// closes around it.
const fadeEl = document.createElement("div");
fadeEl.id = "plungeFade";
fadeEl.style.cssText = [
    "position:fixed","top:0","left:0","width:100%","height:100%",
    "background:radial-gradient(ellipse at center, rgba(0,0,0,0) 0%, rgba(40,0,0,1) 60%, rgba(0,0,0,1) 100%)",
    "opacity:0","pointer-events:none",
    "transition:opacity 0.2s linear","z-index:998",
].join(";");
document.body.appendChild(fadeEl);

// Live β/γ/r HUD during the plunge (fades in/out with the dive).
const hudEl = document.createElement("div");
hudEl.id = "plungeHud";
hudEl.style.cssText = [
    "position:fixed","top:50%","right:24px","transform:translateY(-50%)",
    "padding:14px 18px","border:1px solid rgba(255,120,120,0.35)",
    "background:rgba(20,4,4,0.55)","border-radius:8px",
    "font-family:ui-monospace,monospace","font-size:13px",
    "color:#ffe0e0","line-height:1.55","letter-spacing:1px",
    "box-shadow:0 0 28px rgba(255,80,80,0.25)","backdrop-filter:blur(4px)",
    "pointer-events:none","opacity:0",
    "transition:opacity 0.4s linear","z-index:999",
    "min-width:130px","text-align:right",
].join(";");
hudEl.innerHTML = "";
document.body.appendChild(hudEl);

$("bPlunge").addEventListener("click", () => {
    // If user is already very close, give them a real fall by pulling back.
    const startR = Math.max(camera.position.length(), 14.0);
    plunge.startR    = startR;
    plunge.startDir  = (camera.position.lengthSq() > 0
                        ? camera.position.clone().normalize()
                        : new THREE.Vector3(0, 0.23, -0.97));
    plunge.currentR  = startR;
    // Schwarzschild horizon (a=0 in gold-standard mode).
    const M  = uniforms.uMass.value;
    const Rp = 2.0 * M;
    plunge.targetR = Rp * 1.02;

    camera.position.copy(plunge.startDir).multiplyScalar(startR);
    camera.lookAt(0, 0, 0);

    plunge.active     = true;
    plunge.phase      = "falling";
    plunge.phaseStart = performance.now() * 0.001;

    controls.enabled = false;
    if ($("cAuto").checked) {
        $("cAuto").checked = false;
        $("cAuto").dispatchEvent(new Event("change"));
    }
    fadeEl.style.transition = "opacity 0.2s linear";
    fadeEl.style.opacity    = "0";
    hudEl.style.transition  = "opacity 0.5s linear";
    hudEl.style.opacity     = "1";
});

// ============ Click-to-drop spaghetti cube ============
// Modifier-click on the canvas spawns a 0.4M cube of test particles at the
// closest approach of the click ray to the BH (clamped to r ≥ 8M). Plain
// click is reserved for OrbitControls (drag + dolly), so we use Shift+click
// to disambiguate.
canvas.addEventListener("click", (ev) => {
    if (!ev.shiftKey) return;
    const rect = canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
        ((ev.clientX - rect.left) / rect.width)  * 2 - 1,
       -((ev.clientY - rect.top)  / rect.height) * 2 + 1,
    );
    // Build the world-space ray for that pixel.
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(ndc, camera);
    const O = raycaster.ray.origin;
    const D = raycaster.ray.direction.clone().normalize();

    // Closest approach of the line {O + tD} to the BH at origin:
    //   t* = -(O·D); minimum distance = |O - t* D| (perpendicular component).
    // If the ray points away from the BH the closest approach is in the past;
    // we still spawn at t = max(t*, 4) so the cube lands in front of the camera.
    const tStar = -O.dot(D);
    const t = Math.max(tStar, 4.0);
    const point = new THREE.Vector3().copy(O).addScaledVector(D, t);
    // Clamp to r ≥ 6M so spawn isn't already inside the photon sphere.
    const r = point.length();
    if (r < 6.0) {
        point.multiplyScalar(6.0 / Math.max(r, 1e-3));
    }
    spawnSpaghetti(point);
});

// Sky URL
$("bLoadSky").addEventListener("click", () => {
    const url = $("rSkyUrl").value.trim();
    $("vSkyStat").textContent = "loading...";
    tryLoadSkyTexture(url);
});
$("bSkyClear").addEventListener("click", () => {
    uniforms.uHasSky.value = 0;
    if (uniforms.uSkyTex.value) {
        uniforms.uSkyTex.value.dispose();
        uniforms.uSkyTex.value = null;
    }
    $("vSkyStat").textContent = "procedural";
});
bindRange("rSkyRot", "vSkyRot", uniforms.uSkyRot, (v) => v.toFixed(2));

// Bloom strength
bindRange("rBloom", "vBloom", { value: 0 }, (v) => v.toFixed(2),
          (v) => { bloomPass.strength = v; });

// Render / time
const stepsState = { value: 600 };
bindRange("rSteps", "vSteps", stepsState, (v) => v.toFixed(0),
          (v) => { uniforms.uMaxSteps.value = Math.round(v); });

const timeSpeed = { value: 1.0 };
bindRange("rTs", "vTs", timeSpeed, (v) => v.toFixed(2));

const resScale = { value: 1.0 };
bindRange("rRes", "vRes", resScale, (v) => v.toFixed(2), () => resize());

// -------- presets --------
// Inner edge r_in is derived from prograde Kerr ISCO at apply-time
// (di = ISCO(a) * M).  Each preset only declares physical parameters.
const presets = {
    schwarz: {
        M:1.0,  a:0.0,   Mdot:1.0, do_:14.0, tau:1.5, dH:0.08,
        spin:true, incl:70, ao:0.6, fov:60, steps:500, bloom:0.55, boost:0,
    },
    stellar: {
        M:0.6,  a:0.7,   Mdot:1.6, do_:8.0,  tau:2.0, dH:0.05,
        spin:true, incl:75, ao:0.8, fov:55, steps:600, bloom:0.55, boost:0,
    },
    smbh: {
        M:2.4,  a:0.94,  Mdot:0.6, do_:48.0, tau:1.0, dH:0.10,
        spin:true, incl:65, ao:0.35, fov:65, steps:700, bloom:0.50, boost:0,
    },
    edge: {
        M:1.0,  a:0.5,   Mdot:1.2, do_:14.0, tau:1.6, dH:0.08,
        spin:true, incl:88, ao:0.4, fov:60, steps:600, bloom:0.7,  boost:0,
    },
    face: {
        M:1.0,  a:0.5,   Mdot:1.0, do_:14.0, tau:1.5, dH:0.08,
        spin:true, incl:5,  ao:0.6, fov:60, steps:500, bloom:0.45, boost:0,
    },
    cold: {
        M:1.0,  a:0.0,   Mdot:0.2, do_:18.0, tau:0.8, dH:0.03,
        spin:true, incl:70, ao:0.4, fov:55, steps:600, bloom:0.4,  boost:0,
    },
    quasar: {
        M:1.5,  a:0.998, Mdot:2.5, do_:30.0, tau:2.5, dH:0.12,
        spin:true, incl:60, ao:0.5, fov:60, steps:800, bloom:0.7,  boost:0,
    },
    relflyby: {
        M:1.0,  a:0.5,   Mdot:1.2, do_:14.0, tau:1.5, dH:0.08,
        spin:true, incl:80, ao:0.0, fov:75, steps:600, bloom:0.7,  boost:0.85,
    },
};
function applyPreset(p) {
    controls.target.set(0, 0, 0);
    const set  = (id, val) => { $(id).value = val; $(id).dispatchEvent(new Event("input")); };
    const setC = (id, val) => { $(id).checked = val; $(id).dispatchEvent(new Event("change")); };

    // Mass and spin must be set before deriving ISCO-based inner edge.
    set("rMass", p.M);  set("rSpin", p.a);
    set("rMdot", p.Mdot);

    const di = iscoKerr(p.a, true) * p.M;
    // Clamp into the slider's allowable range so we never set a value that
    // would land inside the Kerr horizon by rounding.
    const di_safe = Math.max(di, p.M * (1.0 + Math.sqrt(Math.max(1 - p.a*p.a, 0))) * 1.05);
    set("rDi",   di_safe.toFixed(2));
    set("rDo",   p.do_);
    set("rTau",  p.tau);  set("rDH", p.dH);
    setC("cSpin", p.spin);
    set("rIncl", p.incl);
    set("rAo",   p.ao);   set("rFov", p.fov);
    set("rSteps", p.steps); set("rBloom", p.bloom);
    // p.boost ignored — boost slider removed in gold-standard mode.
    setC("cAuto", true);   setC("cDisk", true);
    controls.update();
}
document.querySelectorAll(".presets button").forEach((btn) => {
    btn.addEventListener("click", () => {
        const name = btn.dataset.preset;
        if (presets[name]) applyPreset(presets[name]);
    });
});

// -------- tooltips --------
const tip = $("tip");
function attachTooltip(el) {
    if (!el.dataset.info) return;
    el.addEventListener("mouseenter", () => {
        tip.innerHTML = el.dataset.info;
        tip.style.display = "block";
    });
    el.addEventListener("mousemove", (e) => {
        const w = tip.offsetWidth, h = tip.offsetHeight;
        let x = e.clientX + 16, y = e.clientY + 18;
        if (x + w + 8 > window.innerWidth)  x = e.clientX - w - 12;
        if (y + h + 8 > window.innerHeight) y = e.clientY - h - 12;
        tip.style.left = x + "px";
        tip.style.top  = y + "px";
    });
    el.addEventListener("mouseleave", () => { tip.style.display = "none"; });
}
document.querySelectorAll("[data-info]").forEach(attachTooltip);

// -------- resize --------
const stats = $("stats");
function resize() {
    const w   = window.innerWidth;
    const h   = window.innerHeight;
    const s   = resScale.value || 1.0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W   = Math.round(w * dpr * s);
    const H   = Math.round(h * dpr * s);

    // Drive the renderer at its true pixel size (W×H) so the GL viewport
    // matches the canvas. Setting pixelRatio=1 stops Three.js from
    // multiplying again — we already folded dpr into W/H.
    renderer.setPixelRatio(1);
    renderer.setSize(W, H, false);
    composer.setSize(W, H);
    bloomPass.setSize(W, H);

    uniforms.uResolution.value.set(W, H);
    renderer.domElement.style.width  = w + "px";
    renderer.domElement.style.height = h + "px";

    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    uniforms.uTanFov.value = Math.tan((camera.fov * 0.5 * Math.PI) / 180);
}
window.addEventListener("resize", resize);
resize();
refreshDerived();

// -------- loop --------
let prev = performance.now();
let simT = 0;
let frames = 0, fpsClock = prev;

function tick(now) {
    const dt = (now - prev) * 0.001;
    prev = now;
    simT += dt * timeSpeed.value;

    controls.update();

    // ---- Pure-radial gold-standard plunge ----
    //
    // Free-fall from rest at infinity: dr/dτ = −sqrt(2M/r).
    // We update camera position by integrating r each frame, snap camera
    // direction to lookAt(origin), and write uCamRadialV so the shader's
    // tetrad uses the true physical 4-velocity. ALL relativistic effects
    // (aberration, Doppler, gravitational redshift) come from the shader's
    // tetrad math now — no JS-side cinematography (no FOV ramp, no exposure
    // boost, no corkscrew, no look-back).
    if (plunge.active) {
        const M  = uniforms.uMass.value;
        const Rp = 2.0 * M;

        if (plunge.phase === "falling") {
            const dtau  = dt * plunge.timeAccel;
            const rSafe = Math.max(plunge.currentR, Rp * 1.001);
            const v_r   = Math.min(0.998, Math.sqrt(2.0 * M / rSafe));
            plunge.currentR = Math.max(plunge.currentR - v_r * dtau,
                                       plunge.targetR);

            camera.position.copy(plunge.startDir).multiplyScalar(plunge.currentR);
            camera.lookAt(0, 0, 0);

            // Signed radial velocity in +r̂: inward = negative. The shader's
            // tetrad construction uses this scalar directly.
            uniforms.uCamRadialV.value = -v_r;

            const gamma = 1.0 / Math.sqrt(Math.max(1.0 - v_r * v_r, 1e-6));
            hudEl.innerHTML =
                "<span style='opacity:0.55;font-size:10px;letter-spacing:2px'>FREE FALL</span><br/>" +
                "β = <b style='color:#ffd6b0'>" + v_r.toFixed(3) + "</b><br/>" +
                "γ = <b style='color:#ffd6b0'>" + gamma.toFixed(2) + "</b><br/>" +
                "r/M = <b style='color:#ffd6b0'>" + plunge.currentR.toFixed(2) + "</b>";

            if (plunge.currentR <= plunge.targetR + 1e-3) {
                plunge.phase      = "fading";
                plunge.phaseStart = now * 0.001;
                fadeEl.style.transition = "opacity " + plunge.fadeDuration + "s linear";
                fadeEl.style.opacity    = "1";
                hudEl.innerHTML +=
                    "<br/><span style='opacity:0.55;font-size:10px;letter-spacing:2px;color:#ff8060'>" +
                    "HORIZON</span>";
            }
        } else if (plunge.phase === "fading") {
            // Camera held at horizon; fade to black, then restore.
            uniforms.uCamRadialV.value = -0.998;
            const elapsed = (now * 0.001) - plunge.phaseStart;
            if (elapsed >= plunge.fadeDuration + plunge.holdDur) {
                plunge.active = false;
                plunge.phase  = "idle";
                controls.enabled = true;
                camera.position.copy(plunge.startDir).multiplyScalar(plunge.startR);
                camera.lookAt(0, 0, 0);
                uniforms.uCamRadialV.value = 0.0;
                fadeEl.style.transition = "opacity 0.7s linear";
                fadeEl.style.opacity    = "0";
                hudEl.style.transition  = "opacity 0.4s linear";
                hudEl.style.opacity     = "0";
            }
        }
    } else {
        // No plunge: camera is static in BH rest frame. Tetrad sees vRad=0.
        uniforms.uCamRadialV.value = 0.0;
        // Legacy uCameraBeta stays zero — the gold-standard tetrad gets the
        // observer's 4-velocity from uCamRadialV, not from this slider.
        uniforms.uCameraBeta.value.set(0, 0, 0);
    }

    // ---- Spaghettification update ----
    // Schwarzschild radial proper-time geodesic: d²r/dτ² = -M/r². Use
    // semi-implicit Euler (update v first, then r) so motion bootstraps
    // correctly from v=0 — the explicit `v(r) = √(2M(1/r-1/r0))` form gets
    // stuck at r=r0 because v=0 there, which is what was happening before.
    // Inner particles see slightly larger acceleration → radial stretching.
    if (spag.active) {
        const M    = uniforms.uMass.value;
        const Rp   = 2.0 * M;
        const dtau = dt * spag.timeAccel;
        let alive  = 0;
        let xMin = Infinity, yMin = Infinity, zMin = Infinity;
        let xMax = -Infinity, yMax = -Infinity, zMax = -Infinity;
        for (const pt of spag.particles) {
            if (pt.r <= Rp * 1.02) continue;
            const a = -M / (pt.r * pt.r);
            pt.v_r += a * dtau;
            // Cap speed at 0.99c so visualization stays sane near horizon
            // (we're not modelling the full Schwarzschild correction here).
            pt.v_r = Math.max(pt.v_r, -0.99);
            pt.r += pt.v_r * dtau;
            if (pt.r < Rp * 1.02) pt.r = Rp * 1.02;
            pt.pos.copy(pt.dir).multiplyScalar(pt.r);
            alive++;
            if (pt.pos.x < xMin) xMin = pt.pos.x;
            if (pt.pos.y < yMin) yMin = pt.pos.y;
            if (pt.pos.z < zMin) zMin = pt.pos.z;
            if (pt.pos.x > xMax) xMax = pt.pos.x;
            if (pt.pos.y > yMax) yMax = pt.pos.y;
            if (pt.pos.z > zMax) zMax = pt.pos.z;
        }
        // Pad the bounding box by the Gaussian half-width so rays grazing
        // the cube edge still pick up emission.
        const pad = 0.5;
        uniforms.uPartBoxMin.value.set(xMin - pad, yMin - pad, zMin - pad);
        uniforms.uPartBoxMax.value.set(xMax + pad, yMax + pad, zMax + pad);

        // Pack the uniform arrays. Inactive slots get w=-1, vel=0.
        for (let i = 0; i < 64; i++) {
            const u = uniforms.uParticles.value[i];
            if (i < spag.particles.length) {
                const pt = spag.particles[i];
                if (pt.r > Rp * 1.025) {
                    u.set(pt.pos.x, pt.pos.y, pt.pos.z, pt.r);
                    uniforms.uParticleVel.value[i] = pt.v_r;
                } else {
                    u.set(0, 0, 0, -1);
                    uniforms.uParticleVel.value[i] = 0;
                }
            } else {
                u.set(0, 0, 0, -1);
                uniforms.uParticleVel.value[i] = 0;
            }
        }
        uniforms.uNumParticles.value = spag.particles.length;
        if (alive === 0) {
            spag.active = false;
            spag.particles.length = 0;
            uniforms.uNumParticles.value = 0;
        }
    } else {
        uniforms.uNumParticles.value = 0;
    }

    uniforms.uTime.value = simT;
    // Force matrixWorld refresh: when controls.enabled is false (plunge), or
    // when we set camera.position/quaternion manually, OrbitControls.update()
    // doesn't refresh the matrix and the shader reads last-frame's transform.
    camera.updateMatrixWorld();
    uniforms.uCameraPos.value.copy(camera.position);
    uniforms.uCameraMat.value.copy(camera.matrixWorld);

    // ---- Hotspot azimuthal phase: prograde Keplerian orbit at uHotR ----
    {
        const M     = uniforms.uMass.value;
        const a     = uniforms.uSpin.value;
        const rh    = uniforms.uHotR.value;
        // Same Ω the disk uses (prograde Kerr); sign follows uSpinDir.
        const sign  = uniforms.uSpinDir.value;
        const omega = sign / (Math.pow(rh, 1.5) / Math.sqrt(Math.max(M, 1e-3)) + sign * a);
        uniforms.uHotPhi.value = omega * simT;
    }

    // ---- Project BH center to NDC for the photon-ring overlay ----
    {
        camera.updateMatrixWorld();
        const bh = new THREE.Vector3(0, 0, 0).project(camera);
        uniforms.uBhScreen.value.set(bh.x, bh.y);
        uniforms.uBhDist.value = camera.position.length();
    }

    composer.render();

    frames++;
    if (now - fpsClock > 500) {
        const fps = (frames * 1000) / (now - fpsClock);
        stats.textContent = fps.toFixed(0) + " fps";
        frames = 0;
        fpsClock = now;
    }
    requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
