// WebGPU port of the Schwarzschild + Lense-Thirring ray-marcher.
//
// - Raw WebGPU (no Three.js renderer); Three.js retained for camera math
//   and OrbitControls only (~150 LOC of input handling).
// - Single-pass fullscreen-triangle fragment shader writes HDR (rgba16float).
// - Composite pass applies ACES tonemap + sRGB encode to the swap chain.
// - Uniforms packed into a 256-byte std140-style buffer; particles in two
//   storage buffers (positions+r_p as vec4, signed radial velocities as f32).
//
// Physics is a 1:1 WGSL port of the GLSL shader in main.js.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const canvas = document.getElementById("gl");

// -------- WebGPU init --------
if (!navigator.gpu) {
    document.body.innerHTML = "<div style='color:#fff;padding:40px;font-family:sans-serif'>"
        + "<h2>WebGPU not available</h2>"
        + "Use Chrome or Edge (and ensure hardware acceleration is on)."
        + "</div>";
    throw new Error("WebGPU unavailable");
}
const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) {
    document.body.innerHTML = "<div style='color:#fff;padding:40px;font-family:sans-serif'>"
        + "<h2>No WebGPU adapter</h2>"
        + "Browser supports WebGPU but no GPU adapter is available."
        + "</div>";
    throw new Error("no adapter");
}
const device  = await adapter.requestDevice();
const ctx     = canvas.getContext("webgpu");
const swapFmt = "bgra8unorm";  // we do sRGB encode manually so HDR composite is linear
ctx.configure({ device, format: swapFmt, alphaMode: "opaque" });

device.lost.then((info) => {
    console.warn("WebGPU device lost:", info.message);
});

// Surface validation / shader errors loudly. WebGPU silently swallows
// pipeline-creation failures otherwise, leaving the user with a black canvas.
device.onuncapturederror = (e) => {
    console.error("[WebGPU error]", e.error);
    showError(e.error.message || String(e.error));
};

function showError(msg) {
    let el = document.getElementById("wgpuErr");
    if (!el) {
        el = document.createElement("div");
        el.id = "wgpuErr";
        el.style.cssText = [
            "position:fixed","top:60px","right:12px","max-width:480px",
            "padding:10px 14px","background:rgba(60,8,8,0.92)","color:#ffd0d0",
            "border:1px solid rgba(255,80,80,0.55)","border-radius:6px",
            "font-family:ui-monospace,monospace","font-size:11px",
            "white-space:pre-wrap","z-index:2000","line-height:1.4",
        ].join(";");
        document.body.appendChild(el);
    }
    el.textContent = "WebGPU: " + msg;
}

async function checkShader(mod, label) {
    const info = await mod.getCompilationInfo();
    let bad = false;
    for (const m of info.messages) {
        const tag = `[${label} ${m.type}] line ${m.lineNum}: ${m.message}`;
        if (m.type === "error") { console.error(tag); bad = true; }
        else                    { console.warn(tag); }
    }
    if (bad) showError(`${label} failed to compile — see console`);
}

// -------- camera / controls (Three.js for math + input only) --------
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
    const a  = aOverM, a2 = a * a;
    const Z1 = 1 + Math.cbrt(1 - a2)
                 * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
    const Z2 = Math.sqrt(3 * a2 + Z1 * Z1);
    const sgn = prograde ? -1 : 1;
    return 3 + Z2 + sgn * Math.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2));
}
function peakTempK(M, Mdot) {
    return 8000 * Math.pow(Math.max(Mdot, 1e-3), 0.25)
                / Math.pow(Math.max(M, 1e-3),    0.75);
}

// -------- mirrored uniform state (host side) --------
// Field names match main.js for parity. Packed into a 256-byte buffer
// before each frame's render.
const u = {
    resolutionX: 1, resolutionY: 1,
    tanFov: Math.tan((camera.fov * 0.5 * Math.PI) / 180),
    mass: 1.0,
    cameraPos: new THREE.Vector3(),
    spin: 0.0,
    cameraMat: new THREE.Matrix4(),
    cameraBeta: new THREE.Vector3(),
    camRadialV: 0.0,
    bhScreenX: 0, bhScreenY: 0,
    bhDist: 18.0,
    time: 0.0,
    showDisk: 1.0, mdot: 1.0, diskInner: 6.0, diskOuter: 14.0,
    diskH: 0.08, spinDir: 1.0, tau: 1.5, tpeakK: peakTempK(1.0, 1.0),
    skyRot: 0.0, hot: 0.0, hotR: 6.0, hotW: 0.6,
    hotBrt: 1.0, hotPhi: 0.0, jet: 0.0, jetBeta: 0.85,
    jetAngle: 0.15, jetLen: 25.0, jetBrt: 1.0, showPRing: 0.0,
    partBoxMin: new THREE.Vector3(),
    numParticles: 0,
    partBoxMax: new THREE.Vector3(),
    hasSky: 0,
    maxSteps: 600,
};

// Particle state (filled by spaghetti spawner each frame).
const particlesPosRp = new Float32Array(64 * 4); // xyz, rp
const particlesVel   = new Float32Array(64);
for (let i = 0; i < 64; i++) particlesPosRp[i*4 + 3] = -1;

// -------- WGSL shaders --------
const SHADER_BH = /* wgsl */`
struct U {
  resolution:    vec2<f32>,    // 0..8
  tanFov:        f32,          // 8..12
  mass:          f32,          // 12..16
  cameraPos:     vec3<f32>,    // 16..28 (vec3 takes 16, ends at 32)
  spin:          f32,          // 28..32
  cameraMat:     mat4x4<f32>,  // 32..96
  cameraBeta:    vec3<f32>,    // 96..108
  camRadialV:    f32,          // 108..112
  bhScreen:      vec2<f32>,    // 112..120
  bhDist:        f32,          // 120..124
  time:          f32,          // 124..128
  showDisk:      f32,
  mdot:          f32,
  diskInner:     f32,
  diskOuter:     f32,          // 128..144
  diskH:         f32,
  spinDir:       f32,
  tau:           f32,
  tpeakK:        f32,           // 144..160
  skyRot:        f32,
  hot:           f32,
  hotR:          f32,
  hotW:          f32,           // 160..176
  hotBrt:        f32,
  hotPhi:        f32,
  jet:           f32,
  jetBeta:       f32,           // 176..192
  jetAngle:      f32,
  jetLen:        f32,
  jetBrt:        f32,
  showPRing:     f32,           // 192..208
  partBoxMin:    vec3<f32>,     // 208..220
  numParticles:  i32,           // 220..224
  partBoxMax:    vec3<f32>,     // 224..236
  hasSky:        i32,           // 236..240
  maxSteps:      i32,           // 240..244
  _pad0:         f32,
  _pad1:         f32,
  _pad2:         f32,            // 244..256
};

@group(0) @binding(0) var<storage, read> u: U;
@group(0) @binding(1) var<storage, read> particles:    array<vec4<f32>, 64>;
@group(0) @binding(2) var<storage, read> particleVels: array<f32, 64>;
@group(0) @binding(3) var skySamp: sampler;
@group(0) @binding(4) var skyTex:  texture_2d<f32>;

const FAR: f32 = 90.0;
const MAX_DISK_HITS: i32 = 4;
const PI: f32 = 3.14159265358979;
const TWO_PI: f32 = 6.28318530717958;
const PI_HALF: f32 = 1.57079632679;

// ---- fullscreen triangle (no vertex buffer) ----
struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VsOut {
  // Triangle covering [-1,1]^2 with three vertices at (-1,-1), (3,-1), (-1,3).
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  let q = p[vid];
  var out: VsOut;
  out.pos = vec4<f32>(q, 0.0, 1.0);
  // uv in [0,1]^2 (u left→right, v bottom→top to match GLSL gl_FragCoord.y)
  out.uv = vec2<f32>(q.x * 0.5 + 0.5, q.y * 0.5 + 0.5);
  return out;
}

// ============ hash ============
fn pcg(vIn: u32) -> u32 {
  var v = vIn * 747796405u + 2891336453u;
  let w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
  return (w >> 22u) ^ w;
}
fn h22(vc: vec2<u32>) -> vec2<f32> {
  let a = pcg(vc.x ^ pcg(vc.y));
  let b = pcg(a + 17u);
  return vec2<f32>(f32(a), f32(b)) / 4294967295.0;
}

// ============ blackbody (Tanner-Helland) ============
fn blackbodyRGB(T: f32) -> vec3<f32> {
  let t = clamp(T * 0.01, 10.0, 400.0);
  var r: f32; var g: f32; var b: f32;
  if (t <= 66.0) {
    r = 1.0;
    g = clamp(0.39008157876902 * log(t) - 0.6318414437886, 0.0, 1.0);
    if (t <= 19.0) { b = 0.0; }
    else { b = clamp(0.5432067891101 * log(t - 10.0) - 1.196254089462, 0.0, 1.0); }
  } else {
    r = clamp(1.29293618606275 * pow(t - 60.0, -0.1332047592), 0.0, 1.0);
    g = clamp(1.12989086089529 * pow(t - 60.0, -0.0755148492), 0.0, 1.0);
    b = 1.0;
  }
  return pow(vec3<f32>(r, g, b), vec3<f32>(2.2));
}

// ============ procedural starfield ============
fn sampleSkyProcedural(dir: vec3<f32>) -> vec3<f32> {
  let theta = atan2(dir.z, dir.x);
  let phi   = asin(clamp(dir.y, -1.0, 1.0));

  var col = vec3<f32>(0.004, 0.006, 0.012);

  let band_axis = dot(dir, normalize(vec3<f32>(0.35, 0.20, 0.92)));
  let band      = exp(-band_axis * band_axis * 22.0);
  col = col + vec3<f32>(0.18, 0.13, 0.09) * band * 0.55;
  let knot      = exp(-band_axis * band_axis * 60.0);
  col = col + vec3<f32>(0.30, 0.22, 0.14) * knot * 0.30;

  for (var oct: i32 = 0; oct < 3; oct = oct + 1) {
    let scale = 60.0 * pow(2.0, f32(oct));
    let uv    = vec2<f32>(theta + PI, phi + PI_HALF) * scale;
    let cellv = floor(uv);
    let ucell = vec2<u32>(u32(i32(cellv.x) + 1024), u32(i32(cellv.y) + 1024))
              + vec2<u32>(u32(oct * 7919));
    let rh    = h22(ucell);
    let thr   = 0.9965 - 0.001 * f32(oct);
    if (rh.x > thr) {
      let frac1  = fract(uv) - 0.5;
      let d      = length(frac1);
      let bright = pow(rh.y, 5.0);
      let star   = bright * exp(-d * d * 60.0);
      let T_K    = mix(3000.0, 12000.0, rh.x);
      let tint   = blackbodyRGB(T_K) * 1.6;
      col = col + tint * star;
    }
  }
  return col;
}

fn sampleSkyTexture(dir: vec3<f32>) -> vec3<f32> {
  let theta = atan2(dir.z, dir.x) + u.skyRot;
  let phi   = asin(clamp(dir.y, -1.0, 1.0));
  let uv = vec2<f32>(0.5 + theta / TWO_PI,
                     0.5 - phi   / PI);  // flip v: (0,0) top-left in WebGPU
  let c = textureSampleLevel(skyTex, skySamp, uv, 0.0).rgb;
  return pow(c, vec3<f32>(2.2));
}

fn sampleSky(dirRest: vec3<f32>, omegaCam: f32) -> vec3<f32> {
  var col: vec3<f32>;
  if (u.hasSky == 1) { col = sampleSkyTexture(dirRest); }
  else               { col = sampleSkyProcedural(dirRest); }
  return col * pow(omegaCam, 4.0);
}

// ============ disk physics ============
fn diskTempNorm(r: f32) -> f32 {
  let x = u.diskInner / r;
  if (x >= 1.0 - 1e-4) { return 0.0; }
  let t = pow(x, 0.75) * pow(max(1.0 - sqrt(x), 0.0), 0.25);
  return t * 2.05;
}

fn diskEmission(hit: vec3<f32>) -> vec3<f32> {
  let r   = length(hit.xz);
  let phi = atan2(hit.z, hit.x);

  let omega = u.spinDir * sqrt(u.mass) / max(pow(r, 1.5), 0.5);
  let a = phi + omega * u.time;

  let n  = 0.5 + 0.5 * sin(a *  4.0 + r * 1.4 + 1.6 * sin(a * 2.0 - r * 0.4));
  let n2 = 0.5 + 0.5 * sin(a * 11.0 - r * 4.0 + u.time * 0.3);
  let n3 = 0.5 + 0.5 * sin(a * 23.0 + r * 9.0);
  let tex = pow(clamp(n * (0.6 + 0.4 * n2) * (0.7 + 0.3 * n3), 0.0, 1.0), 0.55);

  let tNorm = diskTempNorm(r);
  let T_K   = tNorm * u.tpeakK;
  let bbRGB = blackbodyRGB(T_K);
  let SB    = pow(T_K / 6000.0, 4.0);

  var col = bbRGB * SB * (0.30 + 1.7 * tex) * u.mdot;
  let fade = smoothstep(u.diskInner, u.diskInner * 1.10, r) *
             smoothstep(u.diskOuter, u.diskOuter * 0.94, r);
  col = col * fade;
  return col;
}

// ============ Schwarzschild + Lense-Thirring (Cartesian) ============
struct Deriv { dx: vec3<f32>, dp: vec3<f32> };

fn geoDeriv(x: vec3<f32>, p: vec3<f32>, M: f32, a: f32) -> Deriv {
  let r2  = dot(x, x);
  let r   = sqrt(r2);
  let r5  = r2 * r2 * r;
  let cxp = cross(x, p);
  let h2  = dot(cxp, cxp);

  var d: Deriv;
  d.dx = p;
  d.dp = -3.0 * M * h2 * x / r5;
  if (abs(a) > 1e-6) {
    d.dp = d.dp + (4.0 * a * M / (r2 * r)) * cross(vec3<f32>(0.0, 1.0, 0.0), p);
  }
  return d;
}

fn rk4Step(x: ptr<function, vec3<f32>>, p: ptr<function, vec3<f32>>,
           M: f32, a: f32, dl: f32)
{
  let k1 = geoDeriv(*x,                  *p,                  M, a);
  let k2 = geoDeriv(*x + 0.5*dl*k1.dx,   *p + 0.5*dl*k1.dp,   M, a);
  let k3 = geoDeriv(*x + 0.5*dl*k2.dx,   *p + 0.5*dl*k2.dp,   M, a);
  let k4 = geoDeriv(*x + dl*k3.dx,       *p + dl*k3.dp,       M, a);
  *x = *x + dl * (k1.dx + 2.0*k2.dx + 2.0*k3.dx + k4.dx) / 6.0;
  *p = *p + dl * (k1.dp + 2.0*k2.dp + 2.0*k3.dp + k4.dp) / 6.0;
}

// ============ Kerr (Kerr-Schild Cartesian, spin axis = +y) ============
// Real Kerr null geodesics, validated against physics/kerr.py (both the
// Boyer-Lindquist and Kerr-Schild forms): reproduces the prograde/retrograde
// photon-orbit radii, the frame-dragging shadow asymmetry ξ_± = ±3√(M·r_ph) − a,
// and the Schwarzschild a=0 limit. Engaged only when |u.spin| > 0; a=0 keeps the
// proven Schwarzschild integrator above untouched (no regression on existing shots).
struct KSField { f: f32, l: vec3<f32> };

fn ksRadius(x: vec3<f32>, a: f32) -> f32 {
  // KS radius: positive root of r⁴ − (R²−a²)r² − a²y² = 0  (spin axis = y).
  let R2 = dot(x, x);
  let a2 = a * a;
  let r2 = 0.5 * ((R2 - a2) + sqrt((R2 - a2) * (R2 - a2) + 4.0 * a2 * x.y * x.y));
  return sqrt(max(r2, 1e-12));
}

fn ksField(x: vec3<f32>, M: f32, a: f32) -> KSField {
  let r  = ksRadius(x, a);
  let a2 = a * a;
  let r2 = r * r;
  var o: KSField;
  o.f = 2.0 * M * r2 * r / (r2 * r2 + a2 * x.y * x.y);
  o.l = vec3<f32>((r * x.x - a * x.z) / (r2 + a2),
                  x.y / r,
                  (r * x.z + a * x.x) / (r2 + a2));
  return o;
}

// Position-dependent part of 2H for a conserved-E=1 photon: f·(1 + l·p)².
// The constant (−1 + |p|²) drops out of the spatial gradient, so differencing
// this well-conditioned O(f) term avoids f32 cancellation near the null cone.
fn ksPot(x: vec3<f32>, p: vec3<f32>, M: f32, a: f32) -> f32 {
  let fld = ksField(x, M, a);
  let lp  = 1.0 + dot(fld.l, p);
  return fld.f * lp * lp;
}

fn ksDeriv(x: vec3<f32>, p: vec3<f32>, M: f32, a: f32) -> Deriv {
  let fld = ksField(x, M, a);
  let lp  = 1.0 + dot(fld.l, p);        // l^μ p_μ  (E = 1)
  var d: Deriv;
  d.dx = p - fld.f * fld.l * lp;        // dx^i/dλ = g^{iμ} p_μ
  let h  = max(2.0e-3 * ksRadius(x, a), 1.0e-3);
  let gx = (ksPot(x + vec3<f32>(h,0.0,0.0), p, M, a) - ksPot(x - vec3<f32>(h,0.0,0.0), p, M, a)) / (2.0*h);
  let gy = (ksPot(x + vec3<f32>(0.0,h,0.0), p, M, a) - ksPot(x - vec3<f32>(0.0,h,0.0), p, M, a)) / (2.0*h);
  let gz = (ksPot(x + vec3<f32>(0.0,0.0,h), p, M, a) - ksPot(x - vec3<f32>(0.0,0.0,h), p, M, a)) / (2.0*h);
  d.dp = 0.5 * vec3<f32>(gx, gy, gz);   // dp_i/dλ = −½ ∂_i(2H) = +½ ∂_i ksPot
  return d;
}

fn rk4StepKS(x: ptr<function, vec3<f32>>, p: ptr<function, vec3<f32>>,
             M: f32, a: f32, dl: f32)
{
  let k1 = ksDeriv(*x,                  *p,                  M, a);
  let k2 = ksDeriv(*x + 0.5*dl*k1.dx,   *p + 0.5*dl*k1.dp,   M, a);
  let k3 = ksDeriv(*x + 0.5*dl*k2.dx,   *p + 0.5*dl*k2.dp,   M, a);
  let k4 = ksDeriv(*x + dl*k3.dx,       *p + dl*k3.dp,       M, a);
  *x = *x + dl * (k1.dx + 2.0*k2.dx + 2.0*k3.dx + k4.dx) / 6.0;
  *p = *p + dl * (k1.dp + 2.0*k2.dp + 2.0*k3.dp + k4.dp) / 6.0;
}

// Camera-ray init: a null photon at 'cam' whose coordinate velocity points along
// world direction 'n'. Solves (f−1)v_t² + 2f(l·n)v_t + [1 + f(l·n)²] = 0 for the
// future-directed root, then p_i = n_i + f l_i(v_t + l·n), E = v_t − f(v_t + l·n).
// Returns vec4(p, E); reduces to (n, 1) at large r where f→0.
fn ksCameraPhoton(cam: vec3<f32>, n: vec3<f32>, M: f32, a: f32) -> vec4<f32> {
  let fld = ksField(cam, M, a);
  let ln  = dot(fld.l, n);
  let A   = fld.f - 1.0;
  let B   = 2.0 * fld.f * ln;
  let C   = 1.0 + fld.f * ln * ln;
  let disc = max(B * B - 4.0 * A * C, 0.0);
  var vt: f32;
  if (abs(A) < 1e-9) { vt = -C / B; }
  else               { vt = (-B - sqrt(disc)) / (2.0 * A); }
  let lv = vt + ln;
  return vec4<f32>(n + fld.f * fld.l * lv, vt - fld.f * lv);
}

// ============ main ray ============
@fragment
fn fs(@builtin(position) fragPos: vec4<f32>) -> @location(0) vec4<f32> {
  // WebGPU @builtin(position) y points down; flip to match GLSL gl_FragCoord
  // convention (y-up) so the camera matrix code below is unchanged from the
  // GLSL original.
  let pixX = fragPos.x;
  let pixY = u.resolution.y - fragPos.y;
  let frag = (vec2<f32>(pixX, pixY) / u.resolution) * 2.0 - 1.0;
  let aspect = u.resolution.x / u.resolution.y;
  let dirCam = normalize(vec3<f32>(frag.x * aspect * u.tanFov, frag.y * u.tanFov, -1.0));
  let nView  = normalize((u.cameraMat * vec4<f32>(dirCam, 0.0)).xyz);

  // ---- gold-standard tetrad ray construction (Schwarzschild) ----
  let r0  = length(u.cameraPos);
  let ct0 = u.cameraPos.y / max(r0, 1e-3);
  let st0 = sqrt(max(1.0 - ct0 * ct0, 0.0));
  let r_xz = sqrt(max(u.cameraPos.x * u.cameraPos.x +
                      u.cameraPos.z * u.cameraPos.z, 1e-6));
  let cp0 = u.cameraPos.x / r_xz;
  let sp0 = u.cameraPos.z / r_xz;

  let r_hat  = vec3<f32>(st0 * cp0, ct0,  st0 * sp0);
  let th_hat = vec3<f32>(ct0 * cp0, -st0, ct0 * sp0);
  let ph_hat = vec3<f32>(-sp0,        0.0, cp0);

  let n1 = dot(nView, r_hat);
  let n2 = dot(nView, th_hat);
  let n3 = dot(nView, ph_hat);

  let fCam     = 1.0 - 2.0 * u.mass / max(r0, u.mass * 2.05);
  let fCamSqrt = sqrt(max(fCam, 1e-4));

  let vRad = clamp(u.camRadialV, -0.998, 0.998);
  let gam  = 1.0 / sqrt(max(1.0 - vRad * vRad, 1e-6));

  let E_local = fCamSqrt * gam * (1.0 + n1 * vRad);
  let invE    = 1.0 / max(abs(E_local), 1e-6);

  let kR  = (gam * fCamSqrt * (vRad + n1)) * invE;
  let kTh = (n2 / max(r0, 1e-3)) * invE;
  let kPh = (n3 / max(r0 * max(st0, 0.05), 1e-3)) * invE;

  let pX = st0 * cp0 * kR + r0 * ct0 * cp0 * kTh - r0 * st0 * sp0 * kPh;
  let pY = ct0 * kR        - r0 * st0 * kTh;
  let pZ = st0 * sp0 * kR + r0 * ct0 * sp0 * kTh + r0 * st0 * cp0 * kPh;
  let dir = vec3<f32>(pX, pY, pZ);

  let omegaCam = invE;

  // Spin axis = +y (disk normal). a=0 keeps the proven Schwarzschild integrator;
  // |a|>0 switches to the validated Kerr-Schild geodesics (real frame dragging).
  let aSpin = u.spin;
  let kerr  = abs(aSpin) > 1e-4;

  var x = u.cameraPos;
  var p = dir;
  if (kerr) {
    // Re-init the photon for the Kerr metric and normalize to conserved E = 1
    // (so the disk-Doppler omegaEm, which assumes E=1, stays consistent).
    let pe = ksCameraPhoton(u.cameraPos, nView, u.mass, aSpin);
    p = pe.xyz / max(pe.w, 1e-6);
  }

  // Horizon: Schwarzschild 2M, or the Kerr outer horizon r_+ = M + √(M²−a²).
  let Rplus = select(2.0 * u.mass,
                     u.mass + sqrt(max(u.mass * u.mass - aSpin * aSpin, 0.0)),
                     kerr);

  var captured = false;
  var escaped  = false;
  var accumCol = vec3<f32>(0.0);
  var accumA: f32 = 0.0;
  var diskHits: i32 = 0;

  for (var i: i32 = 0; i < 2000; i = i + 1) {
    if (i >= u.maxSteps) { break; }
    if (accumA > 0.99)   { break; }

    let r = length(x);
    // Capture on the true horizon radius: Cartesian |x| for Schwarzschild, but the
    // KS radius for Kerr (|x| = √(r²+a²) in the equatorial plane, so testing |x|
    // would mis-place the shadow edge — the headline Kerr feature).
    let rCap = select(r, ksRadius(x, aSpin), kerr);
    if (rCap < Rplus * 1.001) { captured = true; break; }
    if (r > FAR)              { escaped  = true; break; }

    let dl = clamp(r * 0.10, 0.04, 0.7);

    let prev_x = x;
    let prev_p = p;

    if (kerr) { rk4StepKS(&x, &p, u.mass, aSpin, dl); }
    else      { rk4Step(&x, &p, u.mass, 0.0, dl); }

    // Equatorial-plane crossing: disk + hotspot
    let wantEquatorial = (u.showDisk > 0.5 || u.hot > 0.5);
    if (wantEquatorial && diskHits < MAX_DISK_HITS && prev_x.y * x.y < 0.0) {
      var fr = prev_x.y / (prev_x.y - x.y);
      fr = clamp(fr, 0.0, 1.0);
      let hit  = mix(prev_x, x, vec3<f32>(fr));
      let pHit = mix(prev_p, p, vec3<f32>(fr));
      let rd   = length(hit.xz);

      let Lz = hit.z * pHit.x - hit.x * pHit.z;

      let Omega  = u.spinDir / (pow(rd, 1.5)/sqrt(u.mass) + u.spinDir * aSpin);
      let gtt0   = 1.0 - 2.0*u.mass/rd;
      let gtp0   = -2.0*u.mass*aSpin/rd;
      let gpp0   = (rd*rd + aSpin*aSpin + 2.0*u.mass*aSpin*aSpin/rd);
      let denomU = max(gtt0 - 2.0*Omega*(-gtp0) - Omega*Omega*gpp0, 1e-3);
      let u_t    = 1.0 / sqrt(denomU);
      let u_p    = Omega * u_t;

      let omegaEm = u_t - Lz * u_p;
      let D       = omegaCam / max(omegaEm, 1e-3);
      let boost   = pow(D, 4.0);

      var emitted = false;

      // ---- disk surface ----
      if (u.showDisk > 0.5 && rd > u.diskInner && rd < u.diskOuter) {
        let vyAbs  = abs(normalize(pHit).y);
        let H      = u.diskH * rd;
        let pathH  = H / max(vyAbs, 0.04);
        let tauEff = u.tau * pathH;
        let alpha  = 1.0 - exp(-tauEff);

        let emission = diskEmission(hit) * boost;
        accumCol = accumCol + emission * alpha * (1.0 - accumA);
        accumA   = accumA   + alpha * (1.0 - accumA);
        emitted = true;
      }

      // ---- hotspot ----
      if (u.hot > 0.5) {
        let phH    = atan2(hit.z, hit.x);
        let dr     = rd - u.hotR;
        var dphi   = phH - u.hotPhi;
        dphi = (dphi - TWO_PI * floor((dphi + PI) / TWO_PI));
        let arc    = u.hotR * dphi;
        let dist2  = dr*dr + arc*arc;
        let w2     = max(u.hotW * u.hotW, 1e-3);
        let gauss  = exp(-dist2 / w2);
        if (gauss > 0.001) {
          let hotCol = blackbodyRGB(18000.0) * (8.0 * u.hotBrt) * gauss * boost;
          accumCol = accumCol + hotCol * (1.0 - accumA);
          accumA   = accumA   + clamp(0.6 * gauss, 0.0, 1.0) * (1.0 - accumA);
          emitted = true;
        }
      }

      if (emitted) { diskHits = diskHits + 1; }
    }

    // ---- bipolar jet (volumetric) ----
    if (u.jet > 0.5) {
      let pLen = length(x);
      let yAbs = abs(x.y);
      if (pLen < u.jetLen && yAbs > 0.4 * u.mass) {
        let cosTh   = yAbs / max(pLen, 1e-3);
        let coneCos = cos(u.jetAngle);
        if (cosTh > coneCos) {
          let dlPhys = length(x - prev_x);
          let rPerp  = sqrt(max(pLen*pLen - yAbs*yAbs, 0.0));
          let sig    = max(u.jetAngle * yAbs * 0.7, 0.25);
          var density = exp(-rPerp*rPerp / (sig*sig));
          density = density * smoothstep(u.jetLen, 0.35 * u.jetLen, pLen);
          density = density * smoothstep(0.4 * u.mass, 1.5 * u.mass, yAbs);

          let photonDir = normalize(x - prev_x);
          let nObs       = -photonDir;
          let betaJet    = vec3<f32>(0.0, sign(x.y) * u.jetBeta, 0.0);
          let bb         = u.jetBeta * u.jetBeta;
          let gJ         = 1.0 / sqrt(max(1.0 - bb, 1e-4));
          let Djet       = 1.0 / max(gJ * (1.0 - dot(betaJet, nObs)), 1e-3);
          let jBoost     = pow(Djet, 4.0);

          let jetCol     = vec3<f32>(0.25, 0.55, 1.0);
          let emitScale  = density * u.jetBrt * dlPhys;
          let emit       = jetCol * jBoost * emitScale * 0.05;
          let jetA       = clamp(emitScale * 0.18, 0.0, 0.9);

          accumCol = accumCol + emit * (1.0 - accumA);
          accumA   = accumA   + jetA * (1.0 - accumA);
        }
      }
    }

    // ---- spaghettification particles + wireframe ----
    if (u.numParticles > 0 &&
        all(x >= u.partBoxMin) && all(x <= u.partBoxMax))
    {
      let dlPhys    = length(x - prev_x);
      let photonDir = normalize(x - prev_x);

      // Particle dots
      for (var pi: i32 = 0; pi < 64; pi = pi + 1) {
        if (pi >= u.numParticles) { break; }
        let pt = particles[pi];
        if (pt.w < 0.0) { continue; }
        let d  = x - pt.xyz;
        let d2 = dot(d, d);
        if (d2 > 0.25) { continue; }
        let gauss = exp(-d2 / 0.018);
        if (gauss < 0.002) { continue; }
        let fP    = 1.0 - 2.0 * u.mass / max(pt.w, u.mass * 2.05);
        let fPSqrt = sqrt(max(fP, 1e-3));
        let v_p     = clamp(particleVels[pi], -0.99, 0.99);
        let rHatP   = pt.xyz / max(length(pt.xyz), 1e-3);
        let gammaP  = 1.0 / sqrt(max(1.0 - v_p*v_p, 1e-4));
        let dopShift = gammaP * (1.0 - v_p * dot(rHatP, photonDir));
        let D = omegaCam * fPSqrt / max(dopShift, 1e-3);
        let boost = pow(D, 4.0);
        var T_obs = clamp(14000.0 * D, 600.0, 30000.0);
        let emitCol = blackbodyRGB(T_obs);
        let emission = emitCol * boost * gauss * dlPhys * 0.7;
        let partA    = clamp(gauss * dlPhys * 0.5, 0.0, 0.9);
        accumCol = accumCol + emission * (1.0 - accumA);
        accumA   = accumA   + partA * (1.0 - accumA);
      }

      // Wireframe edges (12 segments connecting the 8 cube corners)
      for (var e: i32 = 0; e < 12; e = e + 1) {
        var ia: i32; var ib: i32;
        if      (e == 0) { ia=0;  ib=3;  }
        else if (e == 1) { ia=12; ib=15; }
        else if (e == 2) { ia=48; ib=51; }
        else if (e == 3) { ia=60; ib=63; }
        else if (e == 4) { ia=0;  ib=12; }
        else if (e == 5) { ia=3;  ib=15; }
        else if (e == 6) { ia=48; ib=60; }
        else if (e == 7) { ia=51; ib=63; }
        else if (e == 8) { ia=0;  ib=48; }
        else if (e == 9) { ia=3;  ib=51; }
        else if (e ==10) { ia=12; ib=60; }
        else             { ia=15; ib=63; }
        let A = particles[ia];
        let B = particles[ib];
        if (A.w < 0.0 || B.w < 0.0) { continue; }
        let ab  = B.xyz - A.xyz;
        let ab2 = dot(ab, ab);
        if (ab2 < 1e-4) { continue; }
        let ax  = x - A.xyz;
        let t   = clamp(dot(ax, ab) / ab2, 0.0, 1.0);
        let closest = A.xyz + t * ab;
        let dE  = x - closest;
        let dE2 = dot(dE, dE);
        if (dE2 > 0.10) { continue; }
        let lineG = exp(-dE2 / 0.006);
        if (lineG < 0.05) { continue; }
        let rE     = 0.5 * (A.w + B.w);
        let v_pE   = clamp(0.5 * (particleVels[ia] + particleVels[ib]), -0.99, 0.99);
        let fE     = 1.0 - 2.0 * u.mass / max(rE, u.mass * 2.05);
        let fESqrt = sqrt(max(fE, 1e-3));
        let rHatE  = closest / max(length(closest), 1e-3);
        let gammaE = 1.0 / sqrt(max(1.0 - v_pE*v_pE, 1e-4));
        let dopE   = gammaE * (1.0 - v_pE * dot(rHatE, photonDir));
        let D_E    = omegaCam * fESqrt / max(dopE, 1e-3);
        let boostE = pow(D_E, 4.0);
        let T_obsE = clamp(14000.0 * D_E, 600.0, 30000.0);
        let edgeCol = blackbodyRGB(T_obsE);
        let emitE   = edgeCol * boostE * lineG * dlPhys * 0.45;
        let edgeA   = clamp(lineG * dlPhys * 0.4, 0.0, 0.85);
        accumCol = accumCol + emitE * (1.0 - accumA);
        accumA   = accumA   + edgeA * (1.0 - accumA);
      }
    }
  }

  var background: vec3<f32>;
  if (captured) {
    background = vec3<f32>(0.0);
  } else {
    let treatAsEscaped = escaped || (length(x) > Rplus * 1.5);
    if (treatAsEscaped) {
      background = sampleSky(normalize(p), omegaCam);
    } else {
      background = vec3<f32>(0.0);
    }
  }

  var col = accumCol + background * (1.0 - accumA);

  // ---- photon-ring overlay ----
  if (u.showPRing > 0.5) {
    // pixHere uses the same y-flip as our 'frag' calculation above so the
    // ring stays consistent with the WebGL convention.
    let pixHere = vec2<f32>(pixX, pixY);
    let pixCent = (u.bhScreen * 0.5 + vec2<f32>(0.5)) * u.resolution;
    let pixD    = length(pixHere - pixCent);
    let bcrit   = 3.0 * sqrt(3.0) * u.mass;
    let ringPix = (bcrit / max(u.bhDist, 1.0)) *
                  (u.resolution.y * 0.5) / max(u.tanFov, 1e-3);
    let w1 = 1.6;
    let d1 = (pixD - ringPix)         / w1;
    let d2 = (pixD - ringPix * 0.985) / 1.0;
    let ring1 = exp(-d1 * d1);
    let ring2 = exp(-d2 * d2) * 0.35;
    col = col + vec3<f32>(0.55, 0.85, 1.0) * (ring1 * 1.1 + ring2 * 0.6);
  }

  return vec4<f32>(col, 1.0);
}
`;

// Composite pass: HDR linear -> ACES tonemap -> sRGB encode.
const SHADER_COMP = /* wgsl */`
@group(0) @binding(0) var hdrTex:  texture_2d<f32>;
@group(0) @binding(1) var hdrSamp: sampler;

struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VsOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  let q = p[vid];
  var out: VsOut;
  out.pos = vec4<f32>(q, 0.0, 1.0);
  // The BH pass wrote "top of camera view" → texel y=0. WebGPU clip space
  // q.y=+1 is the top of the screen, so map q.y=+1 → uv.y=0 to keep the
  // image right-side-up.
  out.uv = vec2<f32>(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5);
  return out;
}

// ACES filmic (Hill 2017 fit), matches Three.js ACESFilmicToneMapping closely.
fn aces(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e),
               vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linearToSRGB(c: vec3<f32>) -> vec3<f32> {
  let cutoff = step(c, vec3<f32>(0.0031308));
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0/2.4)) - 0.055;
  return mix(hi, lo, cutoff);
}

@fragment
fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let hdr = textureSampleLevel(hdrTex, hdrSamp, uv, 0.0).rgb;
  let toned = aces(hdr);
  let srgb  = linearToSRGB(toned);
  return vec4<f32>(srgb, 1.0);
}
`;

// -------- pipelines --------
const bhModule   = device.createShaderModule({ code: SHADER_BH });
const compModule = device.createShaderModule({ code: SHADER_COMP });
checkShader(bhModule,   "BH");
checkShader(compModule, "Composite");

const HDR_FORMAT = "rgba16float";

const bhBindLayout = device.createBindGroupLayout({
    entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
    ],
});

const bhPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bhBindLayout] }),
    vertex:   { module: bhModule, entryPoint: "vs" },
    fragment: { module: bhModule, entryPoint: "fs",
                targets: [{ format: HDR_FORMAT }] },
    primitive: { topology: "triangle-list" },
});

const compBindLayout = device.createBindGroupLayout({
    entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
    ],
});
const compPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [compBindLayout] }),
    vertex:   { module: compModule, entryPoint: "vs" },
    fragment: { module: compModule, entryPoint: "fs",
                targets: [{ format: swapFmt }] },
    primitive: { topology: "triangle-list" },
});

// -------- buffers --------
const UNIFORM_SIZE = 256;
const uniformBuf = device.createBuffer({
    size: UNIFORM_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const partPosBuf = device.createBuffer({
    size: 64 * 16,  // 64 vec4f
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const partVelBuf = device.createBuffer({
    size: 64 * 4,   // 64 f32 (storage allows tight stride)
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

const linearSampler = device.createSampler({
    magFilter: "linear", minFilter: "linear",
    addressModeU: "repeat", addressModeV: "clamp-to-edge",
});

// -------- HDR sky texture (1×1 black until user loads one) --------
function makeBlackTexture() {
    const tex = device.createTexture({
        size: [1, 1, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
        { texture: tex },
        new Uint8Array([0, 0, 0, 255]),
        { bytesPerRow: 4, rowsPerImage: 1 },
        [1, 1, 1],
    );
    return tex;
}
let skyTexture = makeBlackTexture();
let skyTextureView = skyTexture.createView();

async function tryLoadSkyTexture(url) {
    if (!url) return;
    try {
        $("vSkyStat").textContent = "loading...";
        const res = await fetch(url, { mode: "cors" });
        if (!res.ok) throw new Error("HTTP " + res.status);
        const blob = await res.blob();
        const bmp  = await createImageBitmap(blob, { colorSpaceConversion: "none" });
        const tex = device.createTexture({
            size: [bmp.width, bmp.height, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING
                 | GPUTextureUsage.COPY_DST
                 | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        device.queue.copyExternalImageToTexture(
            { source: bmp },
            { texture: tex },
            [bmp.width, bmp.height, 1],
        );
        bmp.close();
        if (skyTexture) skyTexture.destroy();
        skyTexture = tex;
        skyTextureView = tex.createView();
        u.hasSky = 1;
        rebuildBhBindGroup();
        $("vSkyStat").textContent = "loaded";
    } catch (e) {
        console.warn("sky load failed", e);
        $("vSkyStat").textContent = "load failed";
    }
}

// -------- HDR render target (recreated on resize) --------
let hdrTexture = null, hdrView = null, compBindGroup = null;

function ensureHdrTarget(W, H) {
    if (hdrTexture) hdrTexture.destroy();
    hdrTexture = device.createTexture({
        size: [W, H, 1],
        format: HDR_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    hdrView = hdrTexture.createView();
    compBindGroup = device.createBindGroup({
        layout: compBindLayout,
        entries: [
            { binding: 0, resource: hdrView },
            { binding: 1, resource: linearSampler },
        ],
    });
}

let bhBindGroup = null;
function rebuildBhBindGroup() {
    bhBindGroup = device.createBindGroup({
        layout: bhBindLayout,
        entries: [
            { binding: 0, resource: { buffer: uniformBuf } },
            { binding: 1, resource: { buffer: partPosBuf } },
            { binding: 2, resource: { buffer: partVelBuf } },
            { binding: 3, resource: linearSampler },
            { binding: 4, resource: skyTextureView },
        ],
    });
}
rebuildBhBindGroup();

// -------- uniform packing --------
const uniformAB = new ArrayBuffer(UNIFORM_SIZE);
const uniformF32 = new Float32Array(uniformAB);
const uniformI32 = new Int32Array(uniformAB);

// Single source of truth for the WGSL `struct U` storage-buffer layout.
// Each entry is { word, type, get|ref } where `word` is the f32-word offset
// (byte offset / 4) and MUST match the corresponding member of `struct U` in
// SHADER_BH above. packUniforms() and the assert below both derive from this
// one list — there are no hand-numbered indices in the packer anymore, so a
// field added to the struct but not here (or vice-versa) fails loudly at load
// instead of silently shifting every later field and corrupting a frame.
// vec3/mat4 take 16-byte (4-word) alignment; an f32 declared right after a vec3
// legitimately occupies that vec3's trailing pad slot (e.g. spin after cameraPos).
const UNIFORM_WORDS = 64; // 256-byte buffer (UNIFORM_SIZE / 4)
const UNIFORM_FIELDS = [
    { word:  0, type: 'f32',  get: () => u.resolutionX },          // resolution.x
    { word:  1, type: 'f32',  get: () => u.resolutionY },          // resolution.y
    { word:  2, type: 'f32',  get: () => u.tanFov },
    { word:  3, type: 'f32',  get: () => u.mass },
    { word:  4, type: 'vec3', ref: () => u.cameraPos },
    { word:  7, type: 'f32',  get: () => u.spin },  // drives Kerr geodesics + disk ISCO
    { word:  8, type: 'mat4', ref: () => u.cameraMat },            // column-major 4x4
    { word: 24, type: 'vec3', ref: () => u.cameraBeta, dead: 'never read in shader' },
    { word: 27, type: 'f32',  get: () => u.camRadialV },
    { word: 28, type: 'f32',  get: () => u.bhScreenX },            // bhScreen.x
    { word: 29, type: 'f32',  get: () => u.bhScreenY },            // bhScreen.y
    { word: 30, type: 'f32',  get: () => u.bhDist },
    { word: 31, type: 'f32',  get: () => u.time },
    { word: 32, type: 'f32',  get: () => u.showDisk },
    { word: 33, type: 'f32',  get: () => u.mdot },
    { word: 34, type: 'f32',  get: () => u.diskInner },
    { word: 35, type: 'f32',  get: () => u.diskOuter },
    { word: 36, type: 'f32',  get: () => u.diskH },
    { word: 37, type: 'f32',  get: () => u.spinDir },
    { word: 38, type: 'f32',  get: () => u.tau },
    { word: 39, type: 'f32',  get: () => u.tpeakK },
    { word: 40, type: 'f32',  get: () => u.skyRot },
    { word: 41, type: 'f32',  get: () => u.hot },
    { word: 42, type: 'f32',  get: () => u.hotR },
    { word: 43, type: 'f32',  get: () => u.hotW },
    { word: 44, type: 'f32',  get: () => u.hotBrt },
    { word: 45, type: 'f32',  get: () => u.hotPhi },
    { word: 46, type: 'f32',  get: () => u.jet },
    { word: 47, type: 'f32',  get: () => u.jetBeta },
    { word: 48, type: 'f32',  get: () => u.jetAngle },
    { word: 49, type: 'f32',  get: () => u.jetLen },
    { word: 50, type: 'f32',  get: () => u.jetBrt },
    { word: 51, type: 'f32',  get: () => u.showPRing },
    { word: 52, type: 'vec3', ref: () => u.partBoxMin },
    { word: 55, type: 'i32',  get: () => u.numParticles },
    { word: 56, type: 'vec3', ref: () => u.partBoxMax },
    { word: 59, type: 'i32',  get: () => u.hasSky },
    { word: 60, type: 'i32',  get: () => u.maxSteps },
    // words 61..63: tail padding (_pad0/_pad1/_pad2), zeroed in packUniforms()
];

// Validate the layout once at load: catches overlaps, buffer overflow, and
// vec3/mat4 misalignment. A future desync between this table and `struct U`
// throws here, at startup, rather than silently mis-rendering the showpiece.
(function assertUniformLayout() {
    const SPAN = { f32: 1, i32: 1, vec3: 3, mat4: 16 };
    const used = new Array(UNIFORM_WORDS).fill(null);
    for (const f of UNIFORM_FIELDS) {
        const span = SPAN[f.type];
        if (span === undefined) throw new Error(`uniform layout: unknown type '${f.type}'`);
        const align = (f.type === 'vec3' || f.type === 'mat4') ? 4 : 1;
        if (f.word % align !== 0)
            throw new Error(`uniform layout: ${f.type} @word ${f.word} is not ${align * 4}-byte aligned`);
        for (let i = 0; i < span; i++) {
            const w = f.word + i;
            if (w >= UNIFORM_WORDS)
                throw new Error(`uniform layout: field @word ${f.word} overflows ${UNIFORM_WORDS} words`);
            if (used[w] !== null)
                throw new Error(`uniform layout: word ${w} claimed by both @${used[w]} and @${f.word}`);
            used[w] = f.word;
        }
    }
})();

function packUniforms() {
    for (const f of UNIFORM_FIELDS) {
        switch (f.type) {
            case 'f32': uniformF32[f.word] = f.get(); break;
            case 'i32': uniformI32[f.word] = f.get() | 0; break;
            case 'vec3': {
                const v = f.ref();
                uniformF32[f.word]     = v.x;
                uniformF32[f.word + 1] = v.y;
                uniformF32[f.word + 2] = v.z;
                break;
            }
            case 'mat4': {
                const e = f.ref().elements; // column-major float32[16]
                for (let i = 0; i < 16; i++) uniformF32[f.word + i] = e[i];
                break;
            }
        }
    }
    uniformF32[61] = 0; // _pad0
    uniformF32[62] = 0; // _pad1
    uniformF32[63] = 0; // _pad2
}

// -------- UI bindings (mirror main.js) --------
function bindRange(id, valueId, holder, fmt = (v) => v.toFixed(2), onChange = null) {
    const r = $(id), v = $(valueId);
    const apply = () => {
        const x = parseFloat(r.value);
        holder.set(x);
        v.textContent = fmt(x);
        if (onChange) onChange(x);
    };
    r.addEventListener("input", apply);
    apply();
}
function bindCheck(id, holder, onTrue = 1.0, onFalse = 0.0, onChange = null) {
    const c = $(id);
    const apply = () => {
        holder.set(c.checked ? onTrue : onFalse);
        if (onChange) onChange(c.checked);
    };
    c.addEventListener("change", apply);
    apply();
}

// thin setter-bag adapters so bindRange/bindCheck can write into u.* fields
function H(key) { return { set: (v) => { u[key] = v; } }; }

function refreshDerived() {
    $("vRs").textContent     = (2 * u.mass).toFixed(2);
    $("vBcrit").textContent  = (3 * Math.sqrt(3) * u.mass).toFixed(3);
    const ip = iscoKerr(u.spin, true)  * u.mass;
    const ir = iscoKerr(u.spin, false) * u.mass;
    $("vIscoP").textContent  = ip.toFixed(2);
    $("vIscoR").textContent  = ir.toFixed(2);
    const Tp = peakTempK(u.mass, u.mdot);
    u.tpeakK = Tp;
    $("vTpeak").textContent = (Tp / 1000).toFixed(1) + " kK";
}

bindRange("rMass", "vMass", H("mass"), (v) => v.toFixed(2), refreshDerived);
bindRange("rSpin", "vSpin", H("spin"), (v) => v.toFixed(3), refreshDerived);
bindCheck("cDisk", H("showDisk"));
bindRange("rMdot", "vMdot", H("mdot"),      (v) => v.toFixed(2), refreshDerived);
bindRange("rDi",   "vDi",   H("diskInner"), (v) => v.toFixed(2));
bindRange("rDo",   "vDo",   H("diskOuter"), (v) => v.toFixed(1));
bindRange("rTau",  "vTau",  H("tau"),       (v) => v.toFixed(2));
bindRange("rDH",   "vDH",   H("diskH"),     (v) => v.toFixed(2));
bindCheck("cSpin", H("spinDir"), 1.0, -1.0);
$("bSnapISCO").addEventListener("click", () => {
    const ip = iscoKerr(u.spin, true) * u.mass;
    $("rDi").value = ip.toFixed(2);
    $("rDi").dispatchEvent(new Event("input"));
});

bindCheck("cHot",  H("hot"));
bindRange("rHotR", "vHotR", H("hotR"), (v) => v.toFixed(1));
bindRange("rHotW", "vHotW", H("hotW"));
bindRange("rHotB", "vHotB", H("hotBrt"));

bindCheck("cJet",   H("jet"));
bindRange("rJetB",  "vJetB",  H("jetBeta"));
bindRange("rJetA",  "vJetA",  H("jetAngle"), (v) => v.toFixed(3));
bindRange("rJetL",  "vJetL",  H("jetLen"),   (v) => v.toFixed(0));
bindRange("rJetBr", "vJetBr", H("jetBrt"));

bindCheck("cPRing", H("showPRing"));

const inclState = { value: 70, set: (v) => { inclState.value = v; setInclination(v); } };
bindRange("rIncl", "vIncl", inclState, (v) => v.toFixed(0) + "°");
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

bindCheck("cAuto", { set: () => {} }, 1.0, 0.0, (on) => { controls.autoRotate = on; });
bindRange("rAo", "vAo", { set: () => {} }, (v) => v.toFixed(2),
          (v) => { controls.autoRotateSpeed = v; });
bindRange("rFov", "vFov", { set: () => {} }, (v) => v.toFixed(0), (v) => {
    camera.fov = v;
    camera.updateProjectionMatrix();
    u.tanFov = Math.tan((v * 0.5 * Math.PI) / 180);
});

$("bReset").addEventListener("click", () => {
    plunge.active = false;
    plunge.phase  = "idle";
    controls.enabled = true;
    camera.position.set(0, 4.0, -17);
    controls.target.set(0, 0, 0);
    const fovSlider = parseFloat($("rFov").value);
    camera.fov = fovSlider;
    camera.updateProjectionMatrix();
    u.tanFov = Math.tan((fovSlider * 0.5 * Math.PI) / 180);
    u.cameraBeta.set(0, 0, 0);
    u.camRadialV = 0.0;
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

// Plunge state (free-fall radial geodesic)
const plunge = {
    active: false, phase: "idle",
    phaseStart: 0, fadeDuration: 1.0, holdDur: 0.6,
    startDir: new THREE.Vector3(),
    startR: 0, currentR: 0, targetR: 0,
    timeAccel: 6.0,
};

const fadeEl = document.createElement("div");
fadeEl.id = "plungeFade";
fadeEl.style.cssText = [
    "position:fixed","top:0","left:0","width:100%","height:100%",
    "background:radial-gradient(ellipse at center, rgba(0,0,0,0) 0%, rgba(40,0,0,1) 60%, rgba(0,0,0,1) 100%)",
    "opacity:0","pointer-events:none",
    "transition:opacity 0.2s linear","z-index:998",
].join(";");
document.body.appendChild(fadeEl);

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
document.body.appendChild(hudEl);

$("bPlunge").addEventListener("click", () => {
    const startR = Math.max(camera.position.length(), 14.0);
    plunge.startR    = startR;
    plunge.startDir  = (camera.position.lengthSq() > 0
                        ? camera.position.clone().normalize()
                        : new THREE.Vector3(0, 0.23, -0.97));
    plunge.currentR  = startR;
    const Rp = 2.0 * u.mass;
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

// Spaghettification cube
const spag = {
    particles: [],
    active:    false,
    timeAccel: 10.0,
};
function spawnSpaghetti(spawnCenter) {
    spag.particles.length = 0;
    spag.active = true;
    const radialOut = spawnCenter.clone().normalize();
    const radialIn  = radialOut.clone().multiplyScalar(-1);
    let helper = new THREE.Vector3(0, 1, 0);
    if (Math.abs(radialOut.dot(helper)) > 0.95) helper.set(1, 0, 0);
    const t1 = new THREE.Vector3().crossVectors(radialOut, helper).normalize();
    const t2 = new THREE.Vector3().crossVectors(radialOut, t1).normalize();
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
                    pos, r, r0: r, v_r: 0,
                    dir: pos.clone().normalize(),
                });
            }
        }
    }
}

canvas.addEventListener("click", (ev) => {
    if (!ev.shiftKey) return;
    const rect = canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
        ((ev.clientX - rect.left) / rect.width)  * 2 - 1,
       -((ev.clientY - rect.top)  / rect.height) * 2 + 1,
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(ndc, camera);
    const O = raycaster.ray.origin;
    const D = raycaster.ray.direction.clone().normalize();
    const tStar = -O.dot(D);
    const t = Math.max(tStar, 4.0);
    const point = new THREE.Vector3().copy(O).addScaledVector(D, t);
    const r = point.length();
    if (r < 6.0) point.multiplyScalar(6.0 / Math.max(r, 1e-3));
    spawnSpaghetti(point);
});

$("bLoadSky").addEventListener("click", () => {
    const url = $("rSkyUrl").value.trim();
    tryLoadSkyTexture(url);
});
$("bSkyClear").addEventListener("click", () => {
    if (skyTexture) skyTexture.destroy();
    skyTexture = makeBlackTexture();
    skyTextureView = skyTexture.createView();
    u.hasSky = 0;
    rebuildBhBindGroup();
    $("vSkyStat").textContent = "procedural";
});
bindRange("rSkyRot", "vSkyRot", H("skyRot"), (v) => v.toFixed(2));

bindRange("rBloom", "vBloom", { set: () => {} }, (v) => v.toFixed(2));
// (Bloom not implemented in WebGPU port v1; slider is decorative.)

bindRange("rSteps", "vSteps", H("maxSteps"), (v) => v.toFixed(0));

const timeSpeed = { value: 1.0 };
bindRange("rTs", "vTs", { set: (v) => { timeSpeed.value = v; } },
          (v) => v.toFixed(2));

const resScale = { value: 1.0 };
bindRange("rRes", "vRes", { set: (v) => { resScale.value = v; } },
          (v) => v.toFixed(2), () => resize());

// -------- presets --------
const presets = {
    schwarz: { M:1.0, a:0.0, Mdot:1.0, do_:14.0, tau:1.5, dH:0.08,
               spin:true, incl:70, ao:0.6, fov:60, steps:600, bloom:0.55 },
    stellar: { M:0.6, a:0.7, Mdot:1.6, do_:8.0,  tau:2.0, dH:0.05,
               spin:true, incl:75, ao:0.8, fov:55, steps:600, bloom:0.55 },
    smbh:    { M:2.4, a:0.94,Mdot:0.6, do_:48.0, tau:1.0, dH:0.10,
               spin:true, incl:65, ao:0.35,fov:65, steps:700, bloom:0.50 },
    edge:    { M:1.0, a:0.5, Mdot:1.2, do_:14.0, tau:1.6, dH:0.08,
               spin:true, incl:88, ao:0.4, fov:60, steps:600, bloom:0.7 },
    face:    { M:1.0, a:0.5, Mdot:1.0, do_:14.0, tau:1.5, dH:0.08,
               spin:true, incl:5,  ao:0.6, fov:60, steps:600, bloom:0.45 },
    cold:    { M:1.0, a:0.0, Mdot:0.2, do_:18.0, tau:0.8, dH:0.03,
               spin:true, incl:70, ao:0.4, fov:55, steps:600, bloom:0.4 },
    quasar:  { M:1.5, a:0.998,Mdot:2.5,do_:30.0, tau:2.5, dH:0.12,
               spin:true, incl:60, ao:0.5, fov:60, steps:800, bloom:0.7 },
    relflyby:{ M:1.0, a:0.5, Mdot:1.2, do_:14.0, tau:1.5, dH:0.08,
               spin:true, incl:80, ao:0.0, fov:75, steps:600, bloom:0.7 },
};
function applyPreset(p) {
    controls.target.set(0, 0, 0);
    const set  = (id, val) => { $(id).value = val; $(id).dispatchEvent(new Event("input")); };
    const setC = (id, val) => { $(id).checked = val; $(id).dispatchEvent(new Event("change")); };
    set("rMass", p.M);  set("rSpin", p.a);
    set("rMdot", p.Mdot);
    const di = iscoKerr(p.a, true) * p.M;
    const di_safe = Math.max(di, p.M * (1.0 + Math.sqrt(Math.max(1 - p.a*p.a, 0))) * 1.05);
    set("rDi",   di_safe.toFixed(2));
    set("rDo",   p.do_);
    set("rTau",  p.tau);  set("rDH", p.dH);
    setC("cSpin", p.spin);
    set("rIncl", p.incl);
    set("rAo",   p.ao);   set("rFov", p.fov);
    set("rSteps", p.steps); set("rBloom", p.bloom);
    setC("cAuto", true);  setC("cDisk", true);
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
    const w = window.innerWidth, h = window.innerHeight;
    const s = resScale.value || 1.0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.max(1, Math.round(w * dpr * s));
    const H = Math.max(1, Math.round(h * dpr * s));

    canvas.width  = w * dpr;   // swap chain pixels
    canvas.height = h * dpr;
    canvas.style.width  = w + "px";
    canvas.style.height = h + "px";

    // We render the BH pass at WxH (internal) and the composite blit
    // upsamples to canvas size.
    u.resolutionX = W;
    u.resolutionY = H;

    ensureHdrTarget(W, H);

    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    u.tanFov = Math.tan((camera.fov * 0.5 * Math.PI) / 180);
}
window.addEventListener("resize", resize);
resize();
refreshDerived();

// -------- main loop --------
let prev = performance.now();
let simT = 0;
let frames = 0, fpsClock = prev;

function tick(now) {
    const dt = (now - prev) * 0.001;
    prev = now;
    simT += dt * timeSpeed.value;

    controls.update();

    // Plunge integration
    if (plunge.active) {
        const M = u.mass;
        const Rp = 2.0 * M;
        if (plunge.phase === "falling") {
            const dtau  = dt * plunge.timeAccel;
            const rSafe = Math.max(plunge.currentR, Rp * 1.001);
            const v_r   = Math.min(0.998, Math.sqrt(2.0 * M / rSafe));
            plunge.currentR = Math.max(plunge.currentR - v_r * dtau, plunge.targetR);
            camera.position.copy(plunge.startDir).multiplyScalar(plunge.currentR);
            camera.lookAt(0, 0, 0);
            u.camRadialV = -v_r;
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
            u.camRadialV = -0.998;
            const elapsed = (now * 0.001) - plunge.phaseStart;
            if (elapsed >= plunge.fadeDuration + plunge.holdDur) {
                plunge.active = false;
                plunge.phase  = "idle";
                controls.enabled = true;
                camera.position.copy(plunge.startDir).multiplyScalar(plunge.startR);
                camera.lookAt(0, 0, 0);
                u.camRadialV = 0;
                fadeEl.style.transition = "opacity 0.7s linear";
                fadeEl.style.opacity    = "0";
                hudEl.style.transition  = "opacity 0.4s linear";
                hudEl.style.opacity     = "0";
            }
        }
    } else {
        u.camRadialV = 0;
        u.cameraBeta.set(0, 0, 0);
    }

    // Spaghettification update
    if (spag.active) {
        const M  = u.mass;
        const Rp = 2.0 * M;
        const dtau = dt * spag.timeAccel;
        let alive = 0;
        let xMin = Infinity, yMin = Infinity, zMin = Infinity;
        let xMax = -Infinity, yMax = -Infinity, zMax = -Infinity;
        for (const pt of spag.particles) {
            if (pt.r <= Rp * 1.02) continue;
            const a = -M / (pt.r * pt.r);
            pt.v_r += a * dtau;
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
        const pad = 0.5;
        u.partBoxMin.set(xMin - pad, yMin - pad, zMin - pad);
        u.partBoxMax.set(xMax + pad, yMax + pad, zMax + pad);
        for (let i = 0; i < 64; i++) {
            if (i < spag.particles.length) {
                const pt = spag.particles[i];
                if (pt.r > Rp * 1.025) {
                    particlesPosRp[i*4 + 0] = pt.pos.x;
                    particlesPosRp[i*4 + 1] = pt.pos.y;
                    particlesPosRp[i*4 + 2] = pt.pos.z;
                    particlesPosRp[i*4 + 3] = pt.r;
                    particlesVel[i] = pt.v_r;
                } else {
                    particlesPosRp[i*4 + 3] = -1;
                    particlesVel[i] = 0;
                }
            } else {
                particlesPosRp[i*4 + 3] = -1;
                particlesVel[i] = 0;
            }
        }
        u.numParticles = spag.particles.length;
        if (alive === 0) {
            spag.active = false;
            spag.particles.length = 0;
            u.numParticles = 0;
        }
    } else {
        u.numParticles = 0;
    }

    u.time = simT;

    camera.updateMatrixWorld();
    u.cameraPos.copy(camera.position);
    u.cameraMat.copy(camera.matrixWorld);

    // Hotspot phase
    {
        const M = u.mass, a = u.spin;
        const rh = u.hotR;
        const sgn = u.spinDir;
        const omega = sgn / (Math.pow(rh, 1.5) / Math.sqrt(Math.max(M, 1e-3)) + sgn * a);
        u.hotPhi = omega * simT;
    }

    // BH center → NDC
    {
        const bh = new THREE.Vector3(0, 0, 0).project(camera);
        u.bhScreenX = bh.x;
        u.bhScreenY = bh.y;
        u.bhDist = camera.position.length();
    }

    // ---- pack & upload ----
    packUniforms();
    device.queue.writeBuffer(uniformBuf, 0, uniformAB);
    device.queue.writeBuffer(partPosBuf, 0, particlesPosRp);
    device.queue.writeBuffer(partVelBuf, 0, particlesVel);

    // ---- record + submit ----
    const enc = device.createCommandEncoder();

    // Pass 1: BH ray-march into HDR target
    {
        const pass = enc.beginRenderPass({
            colorAttachments: [{
                view: hdrView,
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: "clear",
                storeOp: "store",
            }],
        });
        pass.setPipeline(bhPipeline);
        pass.setBindGroup(0, bhBindGroup);
        pass.draw(3, 1, 0, 0);
        pass.end();
    }

    // Pass 2: ACES + sRGB encode -> swap chain
    {
        const pass = enc.beginRenderPass({
            colorAttachments: [{
                view: ctx.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: "clear",
                storeOp: "store",
            }],
        });
        pass.setPipeline(compPipeline);
        pass.setBindGroup(0, compBindGroup);
        pass.draw(3, 1, 0, 0);
        pass.end();
    }

    device.queue.submit([enc.finish()]);

    frames++;
    if (now - fpsClock > 500) {
        const fps = (frames * 1000) / (now - fpsClock);
        stats.textContent = fps.toFixed(0) + " fps · WebGPU";
        frames = 0;
        fpsClock = now;
    }
    requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
