// Geodesic deviation demo.
//
// Two marbles dropped from rest. Two side-by-side panes:
//
//   LEFT  : "if gravity pointed down" -- a uniform field. Marbles fall along
//           parallel straight lines; their separation stays constant. This is
//           the kid-intuitive view ("things just fall down").
//
//   RIGHT : real Newtonian gravity that points toward Earth's center, with
//           a continuous transition between exterior (1/r^2) and interior
//           (linear, uniform-density) solutions. Marbles converge.
//
// The convergence on the right is the simplest visualization of geodesic
// deviation -- it tells you, without ever computing a Christoffel symbol,
// that spacetime around mass is curved.
//
// Geometric units: G*M_Earth = 1, R_Earth = 1.

import * as THREE from "three";

// -------- renderer --------
const canvas   = document.getElementById("gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true,
                                           powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

// -------- two scenes --------
const sceneL = new THREE.Scene(); sceneL.background = new THREE.Color(0x070b16);
const sceneR = new THREE.Scene(); sceneR.background = new THREE.Color(0x070b16);

let cameraL, cameraR;
const VIEW_HALF = 4.5;
function setupCameras() {
    const w = window.innerWidth, h = window.innerHeight;
    const halfH = VIEW_HALF;
    const halfW = halfH * (w / 2) / h;
    cameraL = new THREE.OrthographicCamera(-halfW, halfW, halfH, -halfH, 0.1, 100);
    cameraL.position.set(0, 0, 10); cameraL.lookAt(0, 0, 0);
    cameraR = cameraL.clone();
}
setupCameras();

// -------- helpers --------
function makeLine(points, color, opacity = 1.0) {
    const g = new THREE.BufferGeometry().setFromPoints(points);
    const m = new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity });
    return new THREE.Line(g, m);
}
function makeDisk(r, color, opacity = 1.0, segs = 36) {
    const g = new THREE.CircleGeometry(r, segs);
    const m = new THREE.MeshBasicMaterial({ color, transparent: opacity < 1,
                                             opacity, side: THREE.DoubleSide });
    return new THREE.Mesh(g, m);
}
function makeRing(r, color, opacity = 1.0, segs = 96) {
    const pts = [];
    for (let i = 0; i <= segs; i++) {
        const t = (i / segs) * Math.PI * 2;
        pts.push(new THREE.Vector3(Math.cos(t) * r, Math.sin(t) * r, 0));
    }
    return makeLine(pts, color, opacity);
}

// -------- left pane (uniform "down" gravity) --------
//
// Show a flat ground at y = 0; marbles fall along parallel lines toward it.
const groundL = new THREE.Mesh(
    new THREE.PlaneGeometry(40, 0.5),
    new THREE.MeshBasicMaterial({ color: 0x1a2738 }),
);
groundL.position.set(0, -0.5, 0);
sceneL.add(groundL);
const groundLine = makeLine(
    [new THREE.Vector3(-20, -0.25, 0), new THREE.Vector3(20, -0.25, 0)],
    0x4a607c, 0.7
);
sceneL.add(groundLine);

// Vertical reference rails to make 'parallel' visually obvious
const railMatL = new THREE.LineBasicMaterial({ color: 0x223047, transparent: true, opacity: 0.55 });
function addRailL(x) {
    const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x, -0.5, 0), new THREE.Vector3(x, 5, 0),
    ]);
    sceneL.add(new THREE.Line(g, railMatL));
}

// -------- right pane (real radial gravity) --------
const earthR = makeDisk(1.0, 0x2a4a72, 0.9, 64);
sceneR.add(earthR);
const earthAtmoR = makeRing(1.04, 0x6da3d8, 0.45);
sceneR.add(earthAtmoR);
const earthCenter = makeDisk(0.04, 0xffe8a0, 1.0);
earthCenter.position.set(0, 0, 0);
sceneR.add(earthCenter);

// -------- marbles + trails --------
class Trail {
    constructor(scene, color, max = 600) {
        this.max = max;
        this.points = [];
        this.geom = new THREE.BufferGeometry();
        this.posAttr = new THREE.Float32BufferAttribute(new Float32Array(max * 3), 3);
        this.geom.setAttribute("position", this.posAttr);
        const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.6 });
        this.line = new THREE.Line(this.geom, mat);
        scene.add(this.line);
    }
    push(x, y) {
        this.points.push([x, y]);
        if (this.points.length > this.max) this.points.shift();
        const arr = this.posAttr.array;
        for (let i = 0; i < this.points.length; i++) {
            arr[i*3]   = this.points[i][0];
            arr[i*3+1] = this.points[i][1];
            arr[i*3+2] = 0;
        }
        this.posAttr.needsUpdate = true;
        this.geom.setDrawRange(0, this.points.length);
    }
    clear() { this.points = []; this.geom.setDrawRange(0, 0); }
    setVisible(v) { this.line.visible = v; }
}

class Marble {
    constructor(scene, color, sceneTrail) {
        this.dot  = makeDisk(0.06, color, 1.0);
        this.glow = makeDisk(0.16, color, 0.3);
        this.glow.material.blending = THREE.AdditiveBlending;
        scene.add(this.glow);
        scene.add(this.dot);
        this.trail = new Trail(sceneTrail, color);
        this.x = 0; this.y = 0; this.vx = 0; this.vy = 0;
        this.color = color;
    }
    setPos(x, y) {
        this.x = x; this.y = y;
        this.dot.position.set(x, y, 0);
        this.glow.position.set(x, y, 0);
    }
    update(dt, accelFn) {
        // Symplectic-Euler integration (good for orbits)
        const a = accelFn(this.x, this.y);
        this.vx += a[0] * dt;
        this.vy += a[1] * dt;
        this.x  += this.vx * dt;
        this.y  += this.vy * dt;
        this.dot.position.set(this.x, this.y, 0);
        this.glow.position.set(this.x, this.y, 0);
        this.trail.push(this.x, this.y);
    }
}

const marbleLA = new Marble(sceneL, 0xffa3a3, sceneL);
const marbleLB = new Marble(sceneL, 0xa3c8ff, sceneL);
const marbleRA = new Marble(sceneR, 0xffa3a3, sceneR);
const marbleRB = new Marble(sceneR, 0xa3c8ff, sceneR);

// Acceleration arrows
const arrowMat = new THREE.LineBasicMaterial({ color: 0xffe8a0, transparent: true, opacity: 0.7 });
function makeArrow(scene) {
    const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0,0,0), new THREE.Vector3(0,-0.3,0),
    ]);
    const line = new THREE.Line(g, arrowMat);
    scene.add(line);
    return line;
}
const arrowsL = [makeArrow(sceneL), makeArrow(sceneL)];
const arrowsR = [makeArrow(sceneR), makeArrow(sceneR)];
function setArrow(line, fromX, fromY, ax, ay) {
    const arr = line.geometry.attributes.position.array;
    arr[0] = fromX; arr[1] = fromY; arr[2] = 0;
    arr[3] = fromX + ax; arr[4] = fromY + ay; arr[5] = 0;
    line.geometry.attributes.position.needsUpdate = true;
}

// "Flat-space" reference paths (right pane only)
const flatRefMat = new THREE.LineDashedMaterial({
    color: 0x7080a0, dashSize: 0.10, gapSize: 0.10,
    transparent: true, opacity: 0.55,
});
let flatRefA = null, flatRefB = null;
function rebuildFlatRefs(showFlat) {
    [flatRefA, flatRefB].forEach((l) => {
        if (l) { sceneR.remove(l); l.geometry.dispose(); l.material.dispose(); }
    });
    flatRefA = flatRefB = null;
    if (!showFlat) return;
    const xa = -state.sep / 2, xb = +state.sep / 2;
    const yTop = state.h, yBot = -3;
    const ga = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(xa, yTop, 0), new THREE.Vector3(xa, yBot, 0),
    ]);
    const gb = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(xb, yTop, 0), new THREE.Vector3(xb, yBot, 0),
    ]);
    flatRefA = new THREE.Line(ga, flatRefMat); flatRefA.computeLineDistances();
    flatRefB = new THREE.Line(gb, flatRefMat); flatRefB.computeLineDistances();
    sceneR.add(flatRefA); sceneR.add(flatRefB);
}

// -------- physics --------
// Right pane: continuous gravity (G M = 1, R = 1)
const G = 1.0, R = 1.0, M = 1.0;
function gravityRadial(x, y) {
    const r = Math.hypot(x, y);
    if (r < 1e-6) return [0, 0];
    if (r >= R) {
        // Exterior: -GM r̂ / r^2 = -GM (x,y)/r^3
        const k = -G * M / (r * r * r);
        return [k * x, k * y];
    } else {
        // Interior (uniform density): -GM r̂ * r / R^3 = -GM (x,y) / R^3
        const k = -G * M / (R * R * R);
        return [k * x, k * y];
    }
}
// Left pane: uniform "down" with magnitude g0 chosen so initial accel matches
// the right pane's surface value (so the timescales feel similar).
function gravityUniform(x, y) {
    return [0, -1.0];
}

// -------- state / UI --------
const $ = (id) => document.getElementById(id);
const state = {
    sep: 0.5,
    h: 2.0,
    timeSpeed: 1.0,
    playing: true,
    showTrail: true,
    showArrows: false,
    showFlat: false,
    t: 0,
};

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

bindRange("rSep", "vSep", "sep", (v) => v.toFixed(2), () => { resetMarbles(); });
bindRange("rH",   "vH",   "h",   (v) => v.toFixed(2), () => { resetMarbles(); });
bindRange("rTs",  "vTs",  "timeSpeed");
bindCheck("cTrail",  "showTrail",  (v) => {
    [marbleLA, marbleLB, marbleRA, marbleRB].forEach((m) => m.trail.setVisible(v));
});
bindCheck("cArrows", "showArrows", (v) => {
    arrowsL.concat(arrowsR).forEach((a) => a.visible = v);
});
bindCheck("cFlatRef", "showFlat", (v) => rebuildFlatRefs(v));

$("bPlay").textContent = "❚❚ Pause";
$("bPlay").addEventListener("click", () => {
    state.playing = !state.playing;
    $("bPlay").textContent = state.playing ? "❚❚ Pause" : "▶ Play";
});
$("bRestart").addEventListener("click", () => {
    resetMarbles();
    state.playing = false;
    $("bPlay").textContent = "▶ Play";
});

function resetMarbles() {
    state.t = 0;
    const xa = -state.sep / 2, xb = +state.sep / 2;
    const y0 = state.h;
    marbleLA.setPos(xa, y0); marbleLA.vx = 0; marbleLA.vy = 0; marbleLA.trail.clear();
    marbleLB.setPos(xb, y0); marbleLB.vx = 0; marbleLB.vy = 0; marbleLB.trail.clear();
    marbleRA.setPos(xa, y0); marbleRA.vx = 0; marbleRA.vy = 0; marbleRA.trail.clear();
    marbleRB.setPos(xb, y0); marbleRB.vx = 0; marbleRB.vy = 0; marbleRB.trail.clear();
    rebuildFlatRefs(state.showFlat);
}
resetMarbles();

const takeawayEl = $("takeaway");
function refreshDerivedDisplays() {
    const dx = marbleRB.x - marbleRA.x;
    const dy = marbleRB.y - marbleRA.y;
    const sep = Math.hypot(dx, dy);
    $("vDist").textContent = sep.toFixed(3);

    // Tidal acceleration / separation = -GM/r^3 (exterior) or -GM/R^3 (interior)
    const r = Math.hypot((marbleRA.x + marbleRB.x) / 2,
                         (marbleRA.y + marbleRB.y) / 2);
    const tidalPerSep = (r >= R) ? (-G * M / (r * r * r)) : (-G * M / (R * R * R));
    $("vTidal").textContent = tidalPerSep.toFixed(3);

    if (sep < state.sep * 0.9) {
        takeawayEl.textContent =
            "On the right, the marbles' world-lines are bending toward each other. " +
            "No force is pulling them together — gravity is the geometry of spacetime.";
    } else {
        takeawayEl.textContent =
            "Two marbles, both in free fall. Watch what happens on the right side.";
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
    setupCameras();
}
window.addEventListener("resize", resize);
resize();

// -------- loop --------
let prev = performance.now();
let frames = 0, fpsClock = prev;

function tick(now) {
    const dt_real = (now - prev) * 0.001;
    prev = now;
    if (state.playing) {
        // sub-step for stable integration at high time speeds
        const stepDt = 0.005;
        const total = dt_real * state.timeSpeed;
        const N = Math.min(40, Math.ceil(total / stepDt));
        const dt = total / Math.max(N, 1);
        for (let i = 0; i < N; i++) {
            marbleLA.update(dt, gravityUniform);
            marbleLB.update(dt, gravityUniform);
            marbleRA.update(dt, gravityRadial);
            marbleRB.update(dt, gravityRadial);
            state.t += dt;
        }
        // ground collision left side
        [marbleLA, marbleLB].forEach((m) => {
            if (m.y < -0.3) { m.y = -0.3; m.vy = -0.4 * m.vy; }
        });
    }

    // Update arrows
    if (state.showArrows) {
        const sa = 0.6;
        [marbleLA, marbleLB].forEach((m, i) => {
            const a = gravityUniform(m.x, m.y);
            setArrow(arrowsL[i], m.x, m.y, a[0] * sa, a[1] * sa);
        });
        [marbleRA, marbleRB].forEach((m, i) => {
            const a = gravityRadial(m.x, m.y);
            setArrow(arrowsR[i], m.x, m.y, a[0] * sa, a[1] * sa);
        });
    }

    refreshDerivedDisplays();

    // Render with scissor split
    const w = window.innerWidth, h = window.innerHeight;
    renderer.setScissorTest(true);
    renderer.setScissor(0, 0, Math.floor(w / 2), h);
    renderer.setViewport(0, 0, Math.floor(w / 2), h);
    renderer.render(sceneL, cameraL);

    renderer.setScissor(Math.floor(w / 2), 0, w - Math.floor(w / 2), h);
    renderer.setViewport(Math.floor(w / 2), 0, w - Math.floor(w / 2), h);
    renderer.render(sceneR, cameraR);
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
