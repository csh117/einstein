// Light-clock time-dilation demo.
//
// Two physically-identical light clocks side by side:
//
//   Left  : at rest in observer's frame  -> photon path vertical, period 2L/c
//   Right : moving with velocity beta    -> photon path diagonal in observer
//                                            frame, period 2L/(c sqrt(1-beta^2))
//                                            = gamma * (rest period)
//
// In geometric units c = 1 and mirror separation 2L = 2.
// One "tick" = one round trip (photon hits bottom mirror).
//
// The moving clock drifts to the right with v_x = beta and wraps back to its
// start whenever it crosses the right edge, so the diagonal photon path
// stays visible.

import * as THREE        from "three";
import { Line2 }         from "three/addons/lines/Line2.js";
import { LineMaterial }  from "three/addons/lines/LineMaterial.js";
import { LineGeometry }  from "three/addons/lines/LineGeometry.js";

// -------- renderer --------
const canvas   = document.getElementById("gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true,
                                           powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050810);

// Orthographic camera looking at the xz plane (y is "into the screen").
// VIEW_WIDTH instead of VIEW_HEIGHT: we cap to a fixed horizontal world width
// so both clocks always remain on-screen regardless of window aspect.
const VIEW_WIDTH_MIN = 9.5;       // world units horizontally (minimum)
let   camera;
function setupCamera() {
    const w = window.innerWidth, h = window.innerHeight;
    const aspect  = w / h;
    // Use whichever bound makes both clocks visible: width >= VIEW_WIDTH_MIN,
    // height >= 4.5 so the photon's vertical bounce always fits.
    const half_w  = Math.max(VIEW_WIDTH_MIN / 2, (4.5 / 2) * aspect);
    const half_h  = half_w / aspect;
    camera = new THREE.OrthographicCamera(-half_w, half_w, half_h, -half_h, 0.1, 100);
    // Look at the xz plane from -y, so world +x maps to screen right and
    // world +z maps to screen up.
    camera.position.set(0, -10, 0);
    camera.up.set(0, 0, 1);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
}
setupCamera();

// -------- reference grid (toggleable) --------
const gridGroup = new THREE.Group();
const gridMat = new THREE.LineBasicMaterial({ color: 0x202a3a,
                                              transparent: true, opacity: 0.55 });
for (let x = -8; x <= 8; x += 1) {
    const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x, 0, -1.4),
        new THREE.Vector3(x, 0, +1.4),
    ]);
    gridGroup.add(new THREE.Line(g, gridMat));
}
for (const z of [-1.4, -0.5, 0, 0.5, 1.4]) {
    const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-9, 0, z),
        new THREE.Vector3(+9, 0, z),
    ]);
    gridGroup.add(new THREE.Line(g, gridMat));
}
scene.add(gridGroup);

// -------- trail helper --------
//
// Three.js's basic `LineBasicMaterial` has a `linewidth` property but most
// platforms ignore it (WebGL gl.lineWidth is only required to support 1px).
// `Line2` from the addons works around this with a fragment-shader-based
// thick line: width is specified in pixels and rendered as quads.
class Trail {
    constructor(color, max = 360, pixelWidth = 6) {
        this.max = max;
        this.points = [];
        this._buf   = new Float32Array(max * 3);
        this._n     = 0;
        this.length = 0;

        this.geom = new LineGeometry();
        // Initialize with two coincident points so the buffer exists.
        this.geom.setPositions([0, 0, 0, 0, 0, 0]);

        const w = window.innerWidth, h = window.innerHeight;
        this.mat = new LineMaterial({
            color: new THREE.Color(color),
            linewidth: pixelWidth,         // pixels (Line2)
            transparent: true,
            opacity: 0.95,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            resolution: new THREE.Vector2(w, h),
        });

        this.line = new Line2(this.geom, this.mat);
        this.line.computeLineDistances();
        this.line.frustumCulled = false;     // bounding sphere goes stale every push
        scene.add(this.line);
    }

    push(x, y, z) {
        if (this._n > 0) {
            const i = (this._n - 1) * 3;
            const dx = x - this._buf[i],
                  dz = z - this._buf[i + 2];
            this.length += Math.sqrt(dx*dx + dz*dz);
        }
        if (this._n < this.max) {
            const i = this._n * 3;
            this._buf[i] = x; this._buf[i + 1] = y; this._buf[i + 2] = z;
            this._n++;
        } else {
            // ring-shift the buffer one slot left
            this._buf.copyWithin(0, 3);
            const i = (this.max - 1) * 3;
            this._buf[i] = x; this._buf[i + 1] = y; this._buf[i + 2] = z;
        }
        if (this._n < 2) return;
        // LineGeometry.setPositions expects a flat array of [x,y,z,...]
        const slice = this._buf.subarray(0, this._n * 3);
        this.geom.setPositions(slice);
    }

    clear() {
        this._n     = 0;
        this.length = 0;
        // Reset to a degenerate two-point line so geometry stays valid.
        this.geom.setPositions([0, 0, 0, 0, 0, 0]);
    }

    setVisible(v) { this.line.visible = v; }

    // Call from resize() to keep the screen-pixel width consistent.
    setResolution(w, h) {
        this.mat.resolution.set(w, h);
    }
}

// -------- clock --------
class LightClock {
    constructor(baseX, color, label, trailColor) {
        this.baseX = baseX;
        this.label = label;
        this.color = color;
        this.group = new THREE.Group();
        this.group.position.set(baseX, 0, 0);

        // Mirrors: thicker, with a glowing emissive cap on top/bottom faces
        const mMat   = new THREE.MeshBasicMaterial({ color });
        const mGeom  = new THREE.BoxGeometry(2.4, 0.12, 0.08);
        this.top     = new THREE.Mesh(mGeom, mMat);
        this.bot     = new THREE.Mesh(mGeom, mMat);
        this.top.position.set(0, 0, +1);
        this.bot.position.set(0, 0, -1);
        this.group.add(this.top);
        this.group.add(this.bot);

        // Mirror flash sprites: brief bright pulse when photon strikes a mirror
        const flashGeom = new THREE.PlaneGeometry(3.6, 0.6);
        const flashMat  = new THREE.MeshBasicMaterial({
            color, transparent: true, opacity: 0.0,
            blending: THREE.AdditiveBlending, depthWrite: false });
        this.topFlash = new THREE.Mesh(flashGeom, flashMat.clone());
        this.botFlash = new THREE.Mesh(flashGeom, flashMat.clone());
        this.topFlash.position.set(0, 0, +1);
        this.botFlash.position.set(0, 0, -1);
        this.topFlash.rotation.x = -Math.PI / 2;
        this.botFlash.rotation.x = -Math.PI / 2;
        this.group.add(this.topFlash);
        this.group.add(this.botFlash);

        // Side rails (more visible than before)
        const railMat = new THREE.LineBasicMaterial({ color, transparent: true,
                                                       opacity: 0.30 });
        for (const dx of [-1.2, +1.2]) {
            const g = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(dx, 0, -1),
                new THREE.Vector3(dx, 0, +1),
            ]);
            this.group.add(new THREE.Line(g, railMat));
        }

        // Photon: bigger sphere with prominent glow
        this.photon = new THREE.Mesh(
            new THREE.SphereGeometry(0.13, 20, 20),
            new THREE.MeshBasicMaterial({ color: 0xfff2a0 })
        );
        this.glow = new THREE.Mesh(
            new THREE.SphereGeometry(0.34, 20, 20),
            new THREE.MeshBasicMaterial({ color: 0xfff2a0,
                transparent: true, opacity: 0.45,
                blending: THREE.AdditiveBlending, depthWrite: false }),
        );
        this.photon.add(this.glow);
        const haloMat = new THREE.MeshBasicMaterial({ color: 0xfff2a0,
            transparent: true, opacity: 0.18,
            blending: THREE.AdditiveBlending, depthWrite: false });
        this.halo = new THREE.Mesh(new THREE.SphereGeometry(0.6, 18, 14), haloMat);
        this.photon.add(this.halo);
        this.group.add(this.photon);

        scene.add(this.group);

        // Trail (lives in world space, not group, since the moving clock wraps).
        // Pixel-width Line2 so the diagonal-vs-vertical comparison is bold.
        this.trail = new Trail(trailColor || 0xfff2a0, 360, 7);

        // State
        this.photonZ  = -1.0;     // start at bottom mirror
        this.photonVz = +1.0;     // moving up
        this.ticks    = 0;
        this.offsetX  = 0;        // dynamic x drift (only nonzero for moving clock)
        this.flashT   = 0;        // remaining flash time after a tick
        this.flashSide = 0;       // -1 = bottom, +1 = top, 0 = none
    }

    // dt: simulation time elapsed (in observer frame).
    // beta: this clock's velocity in observer frame, 0..1.
    // wrapMin/Max: x-bounds at which the moving clock wraps back.
    update(dt, beta, wrapMin = -100, wrapMax = +100) {
        const vz = Math.sqrt(Math.max(1.0 - beta*beta, 0));   // photon's |z-speed|

        // Drift: clock's center moves with beta in +x
        this.offsetX += beta * dt;
        if (this.offsetX > wrapMax - this.baseX) {
            this.offsetX = wrapMin - this.baseX;
            this.trail.clear();
        }

        // Photon advance + bounce.
        // Snap to the mirror on the frame the photon would cross it instead
        // of reflecting past it. The reflection (velocity flip) is applied
        // here, but the photon's *position* sits on the mirror this frame.
        // Next frame it advances away with the new direction. This makes the
        // visible tap coincide with the counter increment and the flash.
        let z = this.photonZ + this.photonVz * vz * dt;
        let bouncedSide = 0;

        if (z >= 1.0 && this.photonVz > 0) {
            z = 1.0;
            this.photonVz = -1.0;
            this.flashT  = 0.32;
            this.flashSide = +1;
            bouncedSide = +1;
        }
        if (z <= -1.0 && this.photonVz < 0) {
            z = -1.0;
            this.photonVz = +1.0;
            this.ticks++;
            this.flashT  = 0.42;
            this.flashSide = -1;
            bouncedSide = -1;
        }
        this.photonZ = z;

        // Three.js positions
        this.group.position.x = this.baseX + this.offsetX;
        this.photon.position.set(0, 0, z);

        // Trail in world coords (one vertex per frame is enough; the snap
        // above already guarantees we hit the mirror Z exactly on the tap
        // frame, so the trail naturally passes through ±1 on bounces).
        const wx = this.group.position.x + this.photon.position.x;
        const wz = this.photon.position.z;
        this.trail.push(wx, 0, wz);

        // Flash decay
        if (this.flashT > 0) {
            this.flashT = Math.max(0, this.flashT - dt);
            const alpha = Math.min(1, this.flashT / 0.20);
            this.topFlash.material.opacity = (this.flashSide > 0) ? alpha : 0;
            this.botFlash.material.opacity = (this.flashSide < 0) ? alpha : 0;
        } else {
            this.topFlash.material.opacity = 0;
            this.botFlash.material.opacity = 0;
        }
        // Acknowledge the bounce flag (consumed implicitly by flash setup).
        void bouncedSide;
    }

    reset() {
        this.photonZ  = -1.0;
        this.photonVz = +1.0;
        this.ticks    = 0;
        this.offsetX  = 0;
        this.flashT   = 0;
        this.trail.clear();
    }
}

// Stationary clock on the left, moving clock starting just right of center.
// Both clock centers stay inside the visible window (~9.5 wide x 5.3 tall).
const clockA = new LightClock(-3.0,  0x9ed1ff, "Stationary", 0x5fffe0);
const clockB = new LightClock(+0.5,  0xff9adf, "Moving",     0xff6ad5);

// -------- text labels (HTML overlay, screen-positioned) --------
//
// Two pieces per clock: a small caption *below* the apparatus, plus a
// large bold tick-counter card *above* it for the at-a-glance comparison.

function makeOverlay(cssExtra) {
    const el = document.createElement("div");
    el.style.cssText = `
        position: fixed; pointer-events: none;
        font-family: ui-sans-serif, system-ui, sans-serif;
        text-shadow: 0 1px 4px rgba(0,0,0,0.8);
    ` + (cssExtra || "");
    document.body.appendChild(el);
    return el;
}

// big tick cards
const cardA = makeOverlay(`
    text-align: center; padding: 8px 14px; border-radius: 10px;
    border: 1.5px solid rgba(95, 255, 224, 0.45);
    background: rgba(8, 32, 28, 0.55);
    box-shadow: 0 0 18px rgba(95, 255, 224, 0.25);
    backdrop-filter: blur(6px);
    min-width: 110px;
`);
const cardB = makeOverlay(`
    text-align: center; padding: 8px 14px; border-radius: 10px;
    border: 1.5px solid rgba(255, 106, 213, 0.45);
    background: rgba(40, 12, 30, 0.55);
    box-shadow: 0 0 18px rgba(255, 106, 213, 0.25);
    backdrop-filter: blur(6px);
    min-width: 110px;
`);

// caption beneath each apparatus
const labelA = makeOverlay(`font-size: 12px; color: #cdd6e0; opacity: 0.9; text-align: center;`);
const labelB = makeOverlay(`font-size: 12px; color: #cdd6e0; opacity: 0.9; text-align: center;`);
labelA.innerHTML = "Stationary clock<br/><span style='opacity:0.6'>at rest in your frame</span>";
function updateMovingLabel(beta, gamma) {
    labelB.innerHTML = "Moving clock<br/>" +
      "<span style='opacity:0.6'>β = " + beta.toFixed(2) +
      ", γ = " + gamma.toFixed(3) + "</span>";
}

function renderCard(el, ticks, pathPerTick, accent, hot) {
    el.innerHTML =
        "<div style='font-size:10px; letter-spacing:2px; opacity:0.75; " +
              "color:" + accent + ";'>TICKS</div>" +
        "<div style='font-size:36px; font-weight:900; line-height:1; " +
              "color:" + hot + "; text-shadow:0 0 10px " + accent + ";' >" +
            ticks +
        "</div>" +
        "<div style='font-size:10px; letter-spacing:1px; opacity:0.75; margin-top:4px;'>" +
            "path / tick: <b style='color:" + hot + "'>" + pathPerTick.toFixed(2) + "</b>" +
        "</div>";
}

// project a world point to screen pixels and place an overlay
function placeOverlayAt(el, worldX, worldZ, screenYOffset = 0) {
    const v = new THREE.Vector3(worldX, 0, worldZ);
    v.project(camera);
    const w = window.innerWidth, h = window.innerHeight;
    const sx = (v.x + 1) * 0.5 * w;
    const sy = (-v.y + 1) * 0.5 * h;
    el.style.left = (sx - el.offsetWidth / 2) + "px";
    el.style.top  = (sy + screenYOffset) + "px";
}

// keep old call site working
function placeLabelAt(el, worldX, screenYOffset) {
    placeOverlayAt(el, worldX, 0, screenYOffset);
}

// -------- UI bindings --------
const $ = (id) => document.getElementById(id);

const state = {
    beta: 0.5,
    timeSpeed: 1.0,
    showTrail: true,
    showGrid: true,
};

// Must precede any bindRange that uses refreshDerived (TDZ).
const takeawayEl = document.getElementById("takeaway");

function bindRange(id, valId, key, fmt = (v) => v.toFixed(2), onChange = null) {
    const r = $(id), vl = $(valId);
    const apply = () => {
        state[key] = parseFloat(r.value);
        vl.textContent = fmt(state[key]);
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
bindCheck("cTrail", "showTrail", (v) => {
    clockA.trail.setVisible(v); clockB.trail.setVisible(v);
});
bindCheck("cGrid",  "showGrid",  (v) => { gridGroup.visible = v; });

document.querySelectorAll(".presets button").forEach((btn) => {
    btn.addEventListener("click", () => {
        const v = parseFloat(btn.dataset.preset);
        if (!Number.isNaN(v)) {
            $("rBeta").value = v;
            $("rBeta").dispatchEvent(new Event("input"));
            // Reset clocks on preset change so the moving clock's new path is clean
            clockA.reset(); clockB.reset();
        }
    });
});

function refreshDerived() {
    const b = state.beta;
    const g = 1.0 / Math.sqrt(Math.max(1 - b*b, 1e-12));
    $("vGamma").textContent = g.toFixed(3);
    $("vRatio").textContent = (1/g).toFixed(3);
    updateMovingLabel(b, g);
    updateTakeaway(b, g);
}

function updateTakeaway(beta, gamma) {
    if (beta < 0.001) {
        takeawayEl.innerHTML =
            "Both clocks tick identically. β = 0, γ = 1. " +
            "<b>Speed up the right one</b> and watch the photon's path stretch.";
    } else {
        const slow = (1 - 1/gamma) * 100;
        takeawayEl.innerHTML =
            "Same clock. Same light. The <span class='pink'>moving</span> photon " +
            "traces a <b>diagonal — γ = " + gamma.toFixed(2) + "× longer</b> than " +
            "the <span class='cyan'>vertical</span> bounce. " +
            "So the moving clock ticks <b>" + slow.toFixed(1) + "% slower</b>.";
    }
}

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
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h, false);
    setupCamera();
    // Keep Line2 pixel widths in sync after a window resize.
    clockA.trail.setResolution(w, h);
    clockB.trail.setResolution(w, h);
}
window.addEventListener("resize", resize);
resize();

// -------- loop --------
let prev = performance.now();
let frames = 0, fpsClock = prev;

function tick(now) {
    const dt_real = (now - prev) * 0.001;
    prev = now;
    const dt = dt_real * state.timeSpeed;

    // Stationary clock: beta = 0, never wraps
    clockA.update(dt, 0,                 -7.0, +7.0);
    // Moving clock: drifts in +x, wraps when exceeding +3.5 back to -1.0 so
    // the diagonal photon path always stays inside the visible window.
    clockB.update(dt, state.beta,        -1.0, +3.5);

    $("vTicksA").textContent = clockA.ticks.toString();
    $("vTicksB").textContent = clockB.ticks.toString();
    $("vDrift").textContent  = (clockA.ticks - clockB.ticks).toString();

    // Big bold tick cards above each apparatus.
    //   Stationary: path-per-round-trip = 2L = 2.0 (vertical bounce).
    //   Moving:     path-per-round-trip = 2L * γ (diagonal hypotenuse).
    const beta  = state.beta;
    const gamma = 1 / Math.sqrt(Math.max(1 - beta * beta, 1e-12));
    renderCard(cardA, clockA.ticks, 2.0,           "#5fffe0", "#d3fff5");
    renderCard(cardB, clockB.ticks, 2.0 * gamma,   "#ff6ad5", "#ffd3f1");

    // Position cards above mirrors, captions below.
    const aX = clockA.baseX,                  bX = clockB.baseX + clockB.offsetX;
    placeOverlayAt(cardA,  aX, +1, -90);    // above top mirror
    placeOverlayAt(cardB,  bX, +1, -90);
    placeOverlayAt(labelA, aX, -1, +20);    // below bottom mirror
    placeOverlayAt(labelB, bX, -1, +20);

    renderer.render(scene, camera);

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
