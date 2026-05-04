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
    uCameraBeta:  { value: new THREE.Vector3() },
    uTanFov:      { value: Math.tan((camera.fov * 0.5 * Math.PI) / 180) },
    uMass:        { value: 1.0 },
    uSpin:        { value: 0.0 },        // a/M
    uMaxSteps:    { value: 350 },
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
uniform vec3  uCameraBeta;
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

vec3 sampleSky(vec3 dirRest, vec3 nObs) {
    vec3 col = (uHasSky == 1) ? sampleSkyTexture(dirRest)
                              : sampleSkyProcedural(dirRest);
    // Doppler boost on incoming light (direction = +nObs in camera frame
    // is where camera looks; photon arrives travelling in -nObs).
    float D = dopplerFactor(nObs, uCameraBeta);
    return col * pow(D, 4.0);
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

    // Aberrate: camera-frame view direction -> BH-frame direction.
    vec3 dir = aberrate(nView, uCameraBeta);

    float a = uSpin * uMass;     // Kerr spin parameter (units of M)

    // Cartesian state: position x (BH-centered), momentum p with |p|=1 at
    // infinity. No coordinate singularity at the spin axis.
    vec3 x = uCameraPos;
    vec3 p = dir;

    float Rplus = uMass * (1.0 + sqrt(max(1.0 - uSpin * uSpin, 0.0)));

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
            // = E u_t − L_z u^φ, with E = 1 (gauge). Sky frequency is E,
            // so D = 1/(u_t − L_z u^φ), intensity ~ D⁴.
            float omegaEm = u_t - Lz * u_p;
            float D       = 1.0 / max(omegaEm, 1e-3);
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
            background = sampleSky(normalize(p), nView);
        } else {
            background = vec3(0.0);
        }
    }

    vec3 col = accumCol + background * (1.0 - accumA);

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

// Camera boost: fraction of c, applied tangentially along auto-orbit direction
const boostState = { value: 0 };
bindRange("rBoost", "vBoost", boostState, (v) => v.toFixed(2));

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
    camera.position.set(0, 4.0, -17);
    controls.target.set(0, 0, 0);
    $("rIncl").value = 70;  $("rIncl").dispatchEvent(new Event("input"));
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
const stepsState = { value: 500 };
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
    set("rBoost", p.boost);
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
    const w  = window.innerWidth;
    const h  = window.innerHeight;
    const s  = resScale.value || 1.0;
    const pr = renderer.getPixelRatio();
    const W  = Math.round(w * pr * s);
    const H  = Math.round(h * pr * s);

    renderer.setSize(w, h, false);
    composer.setSize(W, H);
    bloomPass.setSize(W, H);

    uniforms.uResolution.value.set(W, H);
    renderer.domElement.width  = W;
    renderer.domElement.height = H;
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

    // Camera boost: along auto-orbit tangent (BH y-axis cross to position).
    const beta = boostState.value;
    if (beta > 1e-4) {
        const r = camera.position.clone().normalize();
        const tangent = new THREE.Vector3(-r.z, 0.0, r.x).normalize();
        uniforms.uCameraBeta.value.copy(tangent.multiplyScalar(beta * 0.999));
    } else {
        uniforms.uCameraBeta.value.set(0, 0, 0);
    }

    uniforms.uTime.value = simT;
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
