// Terrell rotation demo.
//
// A solid object (cube or sphere) translates past the observer at velocity beta
// along the +x axis. The same object is rendered twice via a scissor split:
//
//   Left  half : "naive" view -- only Lorentz contraction along x is applied;
//                light travel time is ignored. This is what most textbooks
//                draw when they say "things get squished at high speed."
//
//   Right half : full Penrose-Terrell. For each vertex, the shader solves for
//                the EMISSION TIME t_e such that a photon leaving the vertex at
//                t_e arrives at the observer at observer-time t_now. The vertex
//                is then rendered at the position it had at t_e.
//
// Result: the right-half cube appears ROTATED, not contracted, and a sphere
// stays a sphere -- the famous Penrose-Terrell theorem (1959).
//
// Geometric units c = 1.

import * as THREE from "three";

// -------- renderer --------
const canvas   = document.getElementById("gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true,
                                           powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x050810);

const OBSERVER_POS = new THREE.Vector3(0, 0, 0);
const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 200);
camera.position.copy(OBSERVER_POS);
camera.lookAt(0, 0, -1);

// -------- subtle starfield backdrop --------
{
    const N = 1500;
    const positions = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
        // distribute on a far sphere
        const theta = Math.acos(2 * Math.random() - 1);
        const phi   = 2 * Math.PI * Math.random();
        const r     = 80;
        positions[i*3]     = r * Math.sin(theta) * Math.cos(phi);
        positions[i*3 + 1] = r * Math.sin(theta) * Math.sin(phi);
        positions[i*3 + 2] = r * Math.cos(theta);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const m = new THREE.PointsMaterial({ color: 0xc4d3e8, size: 0.07,
                                          transparent: true, opacity: 0.7,
                                          sizeAttenuation: true });
    scene.add(new THREE.Points(g, m));
}

// -------- shared shader --------
// Custom raw vertex shader does Lorentz contraction (along +x) + optional
// retarded-position evaluation. Per-face vertex colours selected via a
// vertex attribute.
const vertexShader = /* glsl */ `
// Three.js ShaderMaterial auto-injects:
//   uniform mat4 modelViewMatrix, projectionMatrix, ...
//   attribute vec3 position, normal, vec2 uv
//   precision highp float
// Only declare custom uniforms/attributes here.
uniform float uBeta;
uniform float uTime;
uniform float uMode;          // 0 = naive, 1 = Terrell-correct
uniform vec3  uCenterAtT0;    // object's center at t=0
uniform vec3  uObserverPos;

attribute vec3 color;         // per-vertex face color (custom)

varying vec3 vColor;
varying float vShade;

void main() {
    float beta  = uBeta;
    float beta2 = beta * beta;
    float gamma = 1.0 / sqrt(max(1.0 - beta2, 1e-6));

    // Object's center in observer frame at present (lab) time
    vec3 centerNow = uCenterAtT0 + vec3(beta * uTime, 0.0, 0.0);

    // Apply Lorentz contraction in x to the rest-frame vertex offset
    vec3 offset = position;
    offset.x   /= gamma;

    // Vertex's "instantaneous" position in observer frame
    vec3 P_now = centerNow + offset;

    vec3 P_show = P_now;
    if (uMode > 0.5) {
        // Solve for retarded emission time dt = t_now - t_emit:
        //   |P_now - (beta*dt, 0, 0) - obs| = c * dt   (c = 1)
        // Quadratic in dt, take + root.
        vec3  r   = P_now - uObserverPos;
        float r_x = r.x;
        float r2  = dot(r, r);
        float disc= max(beta2 * r_x * r_x + (1.0 - beta2) * r2, 0.0);
        float dt  = (-beta * r_x + sqrt(disc)) / max(1.0 - beta2, 1e-6);

        // Position when the photon was emitted -- this is what the eye sees.
        P_show = P_now - vec3(beta * dt, 0.0, 0.0);
    }

    // Cheap directional shading from a fixed key light
    vec3  lightDir = normalize(vec3(0.4, 0.5, 0.8));
    float ndotl    = max(dot(normalize(normal), lightDir), 0.0);
    vShade  = 0.45 + 0.55 * ndotl;

    vColor = color;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(P_show, 1.0);
}
`;

const fragmentShader = /* glsl */ `
varying vec3 vColor;
varying float vShade;
void main() {
    vec3 c = vColor * vShade;
    gl_FragColor = vec4(c, 1.0);
}
`;

function makeMaterial(initialMode) {
    return new THREE.ShaderMaterial({
        uniforms: {
            uBeta:        { value: 0.5 },
            uTime:        { value: 0.0 },
            uMode:        { value: initialMode },
            uCenterAtT0:  { value: new THREE.Vector3(-7, 0, -8) },
            uObserverPos: { value: OBSERVER_POS.clone() },
        },
        vertexShader, fragmentShader,
        side: THREE.DoubleSide,
    });
}

const matNaive = makeMaterial(0);
const matCorrect = makeMaterial(1);

// -------- cube geometry with face-coded colours and labels --------
function buildCubeGeom() {
    // 6 faces, each as 2 triangles (4 vertices, 6 indices)
    const FACES = [
        // [ normal, [v0,v1,v2,v3], color ]
        // +x  (R, red)
        [[ 1, 0, 0], [[+1,-1,-1], [+1,+1,-1], [+1,+1,+1], [+1,-1,+1]], [1.0, 0.30, 0.30]],
        // -x  (L, green)
        [[-1, 0, 0], [[-1,-1,-1], [-1,-1,+1], [-1,+1,+1], [-1,+1,-1]], [0.30, 1.0, 0.45]],
        // +y  (U, blue)
        [[ 0, 1, 0], [[-1,+1,-1], [-1,+1,+1], [+1,+1,+1], [+1,+1,-1]], [0.40, 0.55, 1.0]],
        // -y  (D, yellow)
        [[ 0,-1, 0], [[-1,-1,-1], [+1,-1,-1], [+1,-1,+1], [-1,-1,+1]], [1.0, 0.95, 0.35]],
        // +z  (F, magenta)
        [[ 0, 0, 1], [[-1,-1,+1], [+1,-1,+1], [+1,+1,+1], [-1,+1,+1]], [1.0, 0.45, 1.0]],
        // -z  (B, cyan)
        [[ 0, 0,-1], [[-1,-1,-1], [-1,+1,-1], [+1,+1,-1], [+1,-1,-1]], [0.40, 1.0, 1.0]],
    ];
    const positions = [];
    const colors    = [];
    const normals   = [];
    const indices   = [];
    let base = 0;
    for (const [nrm, vs, col] of FACES) {
        for (const v of vs) {
            positions.push(v[0], v[1], v[2]);
            colors.push(col[0], col[1], col[2]);
            normals.push(nrm[0], nrm[1], nrm[2]);
        }
        indices.push(base, base+1, base+2,  base, base+2, base+3);
        base += 4;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    g.setAttribute("color",    new THREE.Float32BufferAttribute(colors,    3));
    g.setAttribute("normal",   new THREE.Float32BufferAttribute(normals,   3));
    g.setIndex(indices);
    return g;
}

function buildSphereGeom(R = 1, segs = 48, rings = 32) {
    const positions = [];
    const colors    = [];
    const normals   = [];
    const indices   = [];

    for (let j = 0; j <= rings; j++) {
        const v = j / rings;
        const phi = v * Math.PI;
        for (let i = 0; i <= segs; i++) {
            const u = i / segs;
            const theta = u * 2 * Math.PI;
            const x = R * Math.sin(phi) * Math.cos(theta);
            const y = R * Math.cos(phi);
            const z = R * Math.sin(phi) * Math.sin(theta);
            positions.push(x, y, z);
            normals.push(x / R, y / R, z / R);
            // Latitude/longitude tartan colour for visual reference
            const stripe = ((Math.floor(u * 12) + Math.floor(v * 12)) % 2 === 0) ? 1.0 : 0.55;
            colors.push(0.55 * stripe + 0.25,
                        0.7 * stripe + 0.2,
                        1.0 * stripe);
        }
    }
    for (let j = 0; j < rings; j++) {
        for (let i = 0; i < segs; i++) {
            const a = j * (segs + 1) + i;
            const b = a + 1;
            const c = a + (segs + 1);
            const d = c + 1;
            indices.push(a, c, b);
            indices.push(b, c, d);
        }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    g.setAttribute("color",    new THREE.Float32BufferAttribute(colors,    3));
    g.setAttribute("normal",   new THREE.Float32BufferAttribute(normals,   3));
    g.setIndex(indices);
    return g;
}

let currentGeom = buildCubeGeom();
const meshNaive   = new THREE.Mesh(currentGeom, matNaive);
const meshCorrect = new THREE.Mesh(currentGeom, matCorrect);
scene.add(meshNaive);
scene.add(meshCorrect);

// "Ghost" of true (non-retarded) position, semi-transparent, only shown on
// the right (Terrell-correct) side.
const ghostMat = matNaive.clone();
ghostMat.transparent = true;
ghostMat.opacity     = 0.18;
ghostMat.depthWrite  = false;
const ghostMesh = new THREE.Mesh(currentGeom, ghostMat);
ghostMesh.visible = false;
scene.add(ghostMesh);

function setObject(kind) {
    currentGeom.dispose();
    currentGeom = (kind === "sphere") ? buildSphereGeom() : buildCubeGeom();
    meshNaive.geometry   = currentGeom;
    meshCorrect.geometry = currentGeom;
    ghostMesh.geometry   = currentGeom;
}

// -------- UI --------
const $ = (id) => document.getElementById(id);
const state = { beta: 0.5, time: 0, timeSpeed: 1, fov: 60, dist: 8,
                paused: false, ghost: false };

// Must precede any bindRange that uses refreshDerived (TDZ).
const takeawayEl = document.getElementById("takeaway");

function bindRange(id, valId, key, fmt = (v) => v.toFixed(2), onChange = null) {
    const r = $(id), v = $(valId);
    const apply = () => {
        state[key] = parseFloat(r.value);
        v.textContent = fmt(state[key]);
        if (onChange) onChange(state[key]);
    };
    r.addEventListener("input", apply);
    apply();
}
function bindCheck(id, key, onChange = null) {
    const c = $(id);
    const apply = () => {
        state[key] = c.checked;
        if (onChange) onChange(state[key]);
    };
    c.addEventListener("change", apply);
    apply();
}

bindRange("rBeta", "vBeta", "beta", (v) => v.toFixed(2), refreshDerived);
bindRange("rTs",   "vTs",   "timeSpeed");
bindRange("rDist", "vDist", "dist", (v) => v.toFixed(1), updateDistance);
bindRange("rFov",  "vFov",  "fov",  (v) => v.toFixed(0), updateFov);
bindCheck("cPause", "paused");
bindCheck("cGhost", "ghost", (v) => { ghostMesh.visible = v; });

$("sObj").addEventListener("change", (e) => setObject(e.target.value));

document.querySelectorAll(".presets button").forEach((btn) => {
    btn.addEventListener("click", () => {
        const v = parseFloat(btn.dataset.preset);
        if (!Number.isNaN(v)) {
            $("rBeta").value = v;
            $("rBeta").dispatchEvent(new Event("input"));
        }
    });
});

function updateFov(v) {
    camera.fov = v;
    camera.updateProjectionMatrix();
}
function updateDistance(d) {
    matNaive.uniforms.uCenterAtT0.value.z   = -d;
    matCorrect.uniforms.uCenterAtT0.value.z = -d;
    ghostMat.uniforms.uCenterAtT0.value.z   = -d;
}

function refreshDerived() {
    const b = state.beta;
    const g = 1 / Math.sqrt(Math.max(1 - b*b, 1e-12));
    const rotDeg = (Math.asin(Math.min(b, 1)) * 180) / Math.PI;
    $("vGamma").textContent = g.toFixed(3);
    $("vRot").textContent   = rotDeg.toFixed(1) + "°";

    if (b < 0.01) {
        takeawayEl.textContent = "At rest, both halves are identical.";
    } else {
        takeawayEl.textContent =
            "Naive view squishes the object by 1/γ = " + (1/g).toFixed(2) +
            ".  Real view rotates it by ≈" + rotDeg.toFixed(0) +
            "° instead.  Light from the back face takes longer to reach you, " +
            "so you see the object's earlier configuration.";
    }
}

// Tooltips
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
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// -------- loop --------
let prev = performance.now();
let frames = 0, fpsClock = prev;

const X_LOOP = 7.0;   // wrap when |center.x| > X_LOOP

function tick(now) {
    const dt_real = (now - prev) * 0.001;
    prev = now;
    if (!state.paused) state.time += dt_real * state.timeSpeed;

    // Compute current center.x (wrap when out of range)
    const beta = state.beta;
    let cx = -X_LOOP + ((state.time * beta + X_LOOP) % (2 * X_LOOP));
    if (Number.isNaN(cx) || beta < 1e-6) {
        cx = 0;
        state.time = 0;
    }
    // We want the shader to compute centerNow = uCenterAtT0 + (beta*time, 0, 0)
    // and have centerNow.x == cx. So set uCenterAtT0.x = cx - beta*time.
    const x0  = cx - beta * state.time;

    [matNaive, matCorrect, ghostMat].forEach((m) => {
        m.uniforms.uBeta.value         = beta;
        m.uniforms.uTime.value         = state.time;
        m.uniforms.uCenterAtT0.value.x = x0;
    });

    // Render twice with scissor
    const w = window.innerWidth, h = window.innerHeight;
    renderer.setScissorTest(true);

    // LEFT half: naive
    renderer.setScissor(0, 0, Math.floor(w / 2), h);
    renderer.setViewport(0, 0, Math.floor(w / 2), h);
    meshNaive.visible = true;
    meshCorrect.visible = false;
    ghostMesh.visible = false;
    renderer.render(scene, camera);

    // RIGHT half: Terrell + optional ghost
    renderer.setScissor(Math.floor(w / 2), 0, w - Math.floor(w / 2), h);
    renderer.setViewport(Math.floor(w / 2), 0, w - Math.floor(w / 2), h);
    meshNaive.visible = false;
    meshCorrect.visible = true;
    ghostMesh.visible = state.ghost;
    renderer.render(scene, camera);

    renderer.setScissorTest(false);

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
