// "Light is Stubborn" -- demo 01.
//
// The pedagogical point: light's speed is invariant. No matter how fast
// you fly, you always measure light at exactly c.
//
// Side-by-side runner's-frame view:
//   LEFT  -- Newton's world. A baseball at v_b = 0.6 c. As you speed up,
//            its apparent speed = v_b - beta drops; at beta = v_b you
//            "catch" it; at beta > v_b it appears to drift backward.
//   RIGHT -- Einstein's world. A photon at c. Apparent speed = c always.
//            No matter how fast you go, the gap keeps growing.
//
// In both panels the SHIP sits stationary at center, while the world
// (starfield, neon grid, distant pillars) streams past at -beta.
// Geometric units c = 1.

import * as THREE from "three";

// ============ DOM hooks ============
const $        = (id) => document.getElementById(id);
const canvas   = document.getElementById("gl");
const stats    = $("stats");
const takeawayEl = $("takeaway");

// ============ state ============
const state = {
    beta:  0.5,        // ship speed in c-units
    vBall: 0.6,        // baseball speed in c-units (Newton's world)
    t:     0.0,        // sim clock (seconds)
    timeSpeed: 1.0,
    playing:   true,
};

// ============ renderer ============
const renderer = new THREE.WebGLRenderer({
    canvas, antialias: true, powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace      = THREE.SRGBColorSpace;
renderer.toneMapping           = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure   = 1.15;
renderer.autoClear             = true;

const camera = new THREE.PerspectiveCamera(46, 1.0, 0.05, 400);
camera.position.set(0, 1.4, 9);
camera.lookAt(0, 0.6, 0);

// ============ scene helpers ============

// Procedural neon synthwave grid: a horizontal plane with glowing magenta/
// cyan grid lines that scroll along +x as the ship "moves." We use a
// custom ShaderMaterial here because we need the lines to glow uniformly
// regardless of scene lighting.
function makeNeonGrid() {
    const geo = new THREE.PlaneGeometry(160, 60, 1, 1);
    const mat = new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        uniforms: {
            uOffsetX:  { value: 0 },
            uColorA:   { value: new THREE.Color(0xff5fc8) },  // magenta
            uColorB:   { value: new THREE.Color(0x5fffe0) },  // cyan
        },
        vertexShader: `
            varying vec2 vUv;
            varying vec3 vWorld;
            void main() {
                vUv = uv;
                vec4 wp = modelMatrix * vec4(position, 1.0);
                vWorld = wp.xyz;
                gl_Position = projectionMatrix * viewMatrix * wp;
            }`,
        fragmentShader: `
            uniform float uOffsetX;
            uniform vec3 uColorA;
            uniform vec3 uColorB;
            varying vec2 vUv;
            varying vec3 vWorld;

            float gridLine(float x, float spacing, float thickness) {
                float f = abs(fract(x / spacing - 0.5) - 0.5);
                return smoothstep(thickness, 0.0, f);
            }

            void main() {
                // Lines parallel to z (constant x): scroll along x with offset
                float gx = gridLine(vWorld.x + uOffsetX, 1.5, 0.04);
                // Lines parallel to x (constant z): static
                float gz = gridLine(vWorld.z, 1.5, 0.05);

                // Major lines every 6 units brighter
                float gxMajor = gridLine(vWorld.x + uOffsetX, 6.0, 0.06);
                float gzMajor = gridLine(vWorld.z, 6.0, 0.06);

                // Distance-fade to horizon
                float d = length(vWorld.xz - vec2(0.0, 0.0));
                float fade = 1.0 - smoothstep(20.0, 70.0, d);

                vec3 col = uColorA * (gx + gxMajor * 1.4)
                         + uColorB * (gz + gzMajor * 1.4);
                float a = (gx * 0.8 + gxMajor + gz * 0.8 + gzMajor) * fade;
                gl_FragColor = vec4(col * fade * 1.4, clamp(a, 0.0, 1.0));
            }`,
    });
    const m = new THREE.Mesh(geo, mat);
    m.rotation.x = -Math.PI / 2;
    m.position.y = -1.0;
    m.userData.mat = mat;
    return m;
}

// Solid dark ground beneath the neon grid for contrast.
function makeFloor() {
    const geo = new THREE.PlaneGeometry(160, 60);
    const mat = new THREE.MeshBasicMaterial({ color: 0x05071a });
    const m = new THREE.Mesh(geo, mat);
    m.rotation.x = -Math.PI / 2;
    m.position.y = -1.005;
    return m;
}

// Synthwave horizon "sun" -- emissive concentric arcs at the back.
function makeHorizonSun() {
    const grp = new THREE.Group();
    const sunGeo = new THREE.CircleGeometry(7, 64);
    const sunMat = new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        uniforms: { uTopCol: { value: new THREE.Color(0xff77c8) },
                    uBotCol: { value: new THREE.Color(0xffd070) } },
        vertexShader: `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }`,
        fragmentShader: `
            uniform vec3 uTopCol, uBotCol;
            varying vec2 vUv;
            void main() {
                vec2 p = vUv - 0.5;
                vec3 col = mix(uBotCol, uTopCol, smoothstep(-0.05, 0.5, p.y));
                // horizontal emissive bands (scan lines)
                float bands = step(0.04, fract((p.y + 0.5) * 12.0));
                col *= mix(0.6, 1.0, bands);
                // soft circular falloff
                float r = length(p) * 2.0;
                float a = 1.0 - smoothstep(0.85, 1.0, r);
                gl_FragColor = vec4(col, a);
            }`,
    });
    const sun = new THREE.Mesh(sunGeo, sunMat);
    sun.position.set(0, 5, -45);
    grp.add(sun);

    // glow halo
    const halo = new THREE.Mesh(
        new THREE.CircleGeometry(11, 64),
        new THREE.ShaderMaterial({
            transparent: true, depthWrite: false,
            uniforms: { uCol: { value: new THREE.Color(0xff77c8) } },
            vertexShader: `varying vec2 vUv; void main(){
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);
            }`,
            fragmentShader: `
                uniform vec3 uCol; varying vec2 vUv;
                void main() {
                    vec2 p = vUv - 0.5;
                    float r = length(p) * 2.0;
                    float a = pow(1.0 - smoothstep(0.0, 1.0, r), 2.0) * 0.55;
                    gl_FragColor = vec4(uCol, a);
                }`,
        }));
    halo.position.copy(sun.position);
    halo.position.z -= 0.1;
    grp.add(halo);
    return grp;
}

// Streaming starfield: many bright dots scattered above the horizon.
// Wraps along x so stars stream past.
function makeStarfield(count = 1500) {
    const positions = new Float32Array(count * 3);
    const colors    = new Float32Array(count * 3);
    const sizes     = new Float32Array(count);
    for (let i = 0; i < count; i++) {
        positions[i*3+0] = (Math.random() * 2 - 1) * 60;
        positions[i*3+1] = Math.random() * 22 + 1.5;
        positions[i*3+2] = -Math.random() * 50 - 6;
        // color: cool blues / pinks / whites
        const hue = Math.random();
        let r=1, g=1, b=1;
        if (hue < 0.4)      { r = 0.85 + Math.random()*0.15; g = 0.7  + Math.random()*0.2; b = 1.0; }
        else if (hue < 0.7) { r = 1.0; g = 0.6 + Math.random()*0.2; b = 0.8 + Math.random()*0.2; }
        else                { r = 1.0; g = 1.0; b = 0.95 + Math.random()*0.05; }
        colors[i*3+0] = r;
        colors[i*3+1] = g;
        colors[i*3+2] = b;
        sizes[i] = Math.random() * 0.06 + 0.02;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("aColor",   new THREE.BufferAttribute(colors,    3));
    geo.setAttribute("size",     new THREE.BufferAttribute(sizes,     1));

    const mat = new THREE.ShaderMaterial({
        transparent: true, depthWrite: false,
        blending: THREE.AdditiveBlending,
        vertexShader: `
            attribute float size;
            attribute vec3 aColor;
            varying vec3 vColor;
            void main() {
                vColor = aColor;
                vec4 mv = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = size * (260.0 / -mv.z);
                gl_Position  = projectionMatrix * mv;
            }`,
        fragmentShader: `
            varying vec3 vColor;
            void main() {
                vec2 p = gl_PointCoord - 0.5;
                float r = length(p);
                float a = smoothstep(0.5, 0.0, r);
                gl_FragColor = vec4(vColor, a);
            }`,
    });
    const points = new THREE.Points(geo, mat);
    points.userData.positions = positions;
    return points;
}

// Streaming "warp" lines -- thin streaks parallel to x-axis. They give
// the strong sense of forward motion in space.
function makeWarpLines(count = 60) {
    const grp = new THREE.Group();
    const mat = new THREE.MeshBasicMaterial({
        color: 0xffffff, transparent: true, opacity: 0.65,
    });
    for (let i = 0; i < count; i++) {
        const len = 0.5 + Math.random() * 2.0;
        const geo = new THREE.BoxGeometry(len, 0.02, 0.02);
        const m = new THREE.Mesh(geo, mat);
        m.position.set(
            (Math.random() * 2 - 1) * 50,
            Math.random() * 16 + 0.5,
           -Math.random() * 30 - 4);
        m.userData.lifetime = Math.random();
        grp.add(m);
    }
    return grp;
}

// Sleek neon racing ship (the "you" pov).
function makeShip() {
    const grp = new THREE.Group();

    const hullMat = new THREE.MeshStandardMaterial({
        color: 0x1a1a2e, roughness: 0.25, metalness: 0.85,
        emissive: 0x110a1c, emissiveIntensity: 0.4,
    });
    const accentMat = new THREE.MeshStandardMaterial({
        color: 0xff6ad5, emissive: 0xff6ad5, emissiveIntensity: 1.4,
        roughness: 0.3, metalness: 0.5,
    });
    const accent2Mat = new THREE.MeshStandardMaterial({
        color: 0x5fffe0, emissive: 0x5fffe0, emissiveIntensity: 1.3,
        roughness: 0.3, metalness: 0.5,
    });
    const cockpitMat = new THREE.MeshPhysicalMaterial({
        color: 0x0c1426, roughness: 0.05, metalness: 0.0,
        transmission: 0.4, transparent: true, opacity: 0.85,
        clearcoat: 1.0,
    });
    const exhaustMat = new THREE.MeshBasicMaterial({
        color: 0xb6e4ff,
    });

    // hull body (wedge)
    const body = new THREE.Mesh(
        new THREE.BoxGeometry(2.2, 0.45, 0.9), hullMat);
    body.position.y = 0.0;
    grp.add(body);

    // forward nose cone pointing +x
    const nose = new THREE.Mesh(
        new THREE.ConeGeometry(0.45, 1.2, 16), hullMat);
    nose.rotation.z = -Math.PI / 2;
    nose.position.set(1.7, 0, 0);
    grp.add(nose);

    // cockpit dome
    const cockpit = new THREE.Mesh(
        new THREE.SphereGeometry(0.32, 24, 18, 0, Math.PI * 2, 0, Math.PI * 0.55),
        cockpitMat);
    cockpit.position.set(0.2, 0.20, 0);
    cockpit.scale.set(1.4, 1.0, 1.0);
    grp.add(cockpit);

    // wings (swept back)
    for (const sgn of [-1, 1]) {
        const wing = new THREE.Mesh(
            new THREE.BoxGeometry(1.3, 0.06, 0.5), hullMat);
        wing.position.set(-0.3, -0.05, sgn * 0.7);
        wing.rotation.y = sgn * 0.3;
        grp.add(wing);

        // glowing wing edge accent
        const edge = new THREE.Mesh(
            new THREE.BoxGeometry(1.3, 0.04, 0.05),
            sgn > 0 ? accentMat : accent2Mat);
        edge.position.set(-0.3, -0.04, sgn * 0.93);
        edge.rotation.y = sgn * 0.3;
        grp.add(edge);
    }

    // body underbelly accent line (alternating colors front/back)
    const accentLineFront = new THREE.Mesh(
        new THREE.BoxGeometry(1.3, 0.04, 0.04), accent2Mat);
    accentLineFront.position.set(0.5, -0.18, 0);
    grp.add(accentLineFront);
    const accentLineBack = new THREE.Mesh(
        new THREE.BoxGeometry(0.6, 0.04, 0.04), accentMat);
    accentLineBack.position.set(-0.7, -0.18, 0);
    grp.add(accentLineBack);

    // engines (back) — hot blue exhaust
    for (const sgn of [-1, 1]) {
        const eng = new THREE.Mesh(
            new THREE.CylinderGeometry(0.18, 0.20, 0.30, 14),
            hullMat);
        eng.rotation.z = Math.PI / 2;
        eng.position.set(-1.18, -0.02, sgn * 0.30);
        grp.add(eng);

        const flame = new THREE.Mesh(
            new THREE.ConeGeometry(0.16, 1.2, 14), exhaustMat);
        flame.rotation.z = Math.PI / 2;
        flame.position.set(-1.95, -0.02, sgn * 0.30);
        flame.userData.isFlame = true;
        grp.add(flame);
    }

    grp.position.y = 0.4;
    return grp;
}

// Baseball: red sphere with white stitching texture. (Canvas2D drawn.)
function makeBaseballTexture() {
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    const g = c.getContext("2d");
    // base
    g.fillStyle = "#d22835"; g.fillRect(0, 0, 256, 256);
    // soft shading
    const grad = g.createRadialGradient(96, 96, 8, 128, 128, 200);
    grad.addColorStop(0,    "rgba(255,180,180,0.7)");
    grad.addColorStop(0.5,  "rgba(0,0,0,0.0)");
    grad.addColorStop(1,    "rgba(0,0,0,0.45)");
    g.fillStyle = grad; g.fillRect(0, 0, 256, 256);
    // stitches: two arcs
    g.strokeStyle = "#ffffff"; g.lineWidth = 4;
    g.setLineDash([10, 6]);
    g.beginPath(); g.arc(128, 128, 90, -0.85, 0.85); g.stroke();
    g.beginPath(); g.arc(128, 128, 90, Math.PI - 0.85, Math.PI + 0.85); g.stroke();
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
}

function makeBaseball() {
    const grp = new THREE.Group();
    const ball = new THREE.Mesh(
        new THREE.SphereGeometry(0.35, 28, 22),
        new THREE.MeshStandardMaterial({
            map: makeBaseballTexture(),
            emissive: 0x331010, emissiveIntensity: 0.2,
            roughness: 0.6, metalness: 0.0,
        }));
    ball.position.y = 0;
    grp.add(ball);

    // bright halo behind so it pops on the dark scene
    const halo = new THREE.Mesh(
        new THREE.SphereGeometry(0.55, 18, 14),
        new THREE.MeshBasicMaterial({
            color: 0xff8060, transparent: true, opacity: 0.18,
        }));
    halo.position.copy(ball.position);
    grp.add(halo);

    // motion-trail streak (cone pointing -x)
    const trailMat = new THREE.MeshBasicMaterial({
        color: 0xff7050, transparent: true, opacity: 0.55,
    });
    const trail = new THREE.Mesh(
        new THREE.ConeGeometry(0.30, 1.6, 14), trailMat);
    trail.rotation.z = Math.PI / 2;
    trail.position.x = -0.95;
    grp.add(trail);

    return grp;
}

function makePhoton() {
    const grp = new THREE.Group();

    // bright white-yellow core
    const core = new THREE.Mesh(
        new THREE.SphereGeometry(0.32, 28, 22),
        new THREE.MeshBasicMaterial({ color: 0xfff7c0 }));
    grp.add(core);

    // inner glow
    const glow1 = new THREE.Mesh(
        new THREE.SphereGeometry(0.55, 22, 16),
        new THREE.MeshBasicMaterial({
            color: 0xfff080, transparent: true, opacity: 0.55,
        }));
    grp.add(glow1);

    // outer glow
    const glow2 = new THREE.Mesh(
        new THREE.SphereGeometry(1.1, 22, 16),
        new THREE.MeshBasicMaterial({
            color: 0xfff08a, transparent: true, opacity: 0.18,
        }));
    grp.add(glow2);

    // long stretched motion trail
    const trail = new THREE.Mesh(
        new THREE.ConeGeometry(0.40, 4.0, 18),
        new THREE.MeshBasicMaterial({
            color: 0xfff8a0, transparent: true, opacity: 0.55,
        }));
    trail.rotation.z = Math.PI / 2;
    trail.position.x = -2.2;
    grp.add(trail);

    // even longer faint trail behind
    const trail2 = new THREE.Mesh(
        new THREE.ConeGeometry(0.55, 7.0, 18),
        new THREE.MeshBasicMaterial({
            color: 0xfff080, transparent: true, opacity: 0.18,
        }));
    trail2.rotation.z = Math.PI / 2;
    trail2.position.x = -3.8;
    grp.add(trail2);

    // a point light so the photon actually lights up the ship subtly
    const pl = new THREE.PointLight(0xfff080, 1.5, 8, 1.4);
    grp.add(pl);

    return grp;
}

// ============ build the two scenes ============
function buildScene(label) {
    const sc = new THREE.Scene();
    sc.background = new THREE.Color(0x02030c);
    sc.fog        = new THREE.Fog(0x02030c, 30, 80);

    sc.add(makeHorizonSun());
    const grid    = makeNeonGrid();      sc.add(grid);
    sc.add(makeFloor());
    const stars   = makeStarfield(1200); sc.add(stars);
    const warp    = makeWarpLines(70);   sc.add(warp);

    // Soft fill light + faint key light so the ship has subtle shading.
    sc.add(new THREE.HemisphereLight(0xa080ff, 0x110820, 0.6));
    const key = new THREE.DirectionalLight(0xfff0f8, 0.6);
    key.position.set(2, 4, 2);
    sc.add(key);
    const rim = new THREE.DirectionalLight(0x5fffe0, 0.4);
    rim.position.set(-3, 2, -2);
    sc.add(rim);

    // Ship at center.
    const ship = makeShip();
    sc.add(ship);

    // Projectile: baseball on left scene, photon on right.
    const projectile = (label === "ball") ? makeBaseball() : makePhoton();
    projectile.position.set(0, 0.4, 0);
    sc.add(projectile);

    sc.userData = { ship, projectile, grid, stars, warp, label };
    return sc;
}

const sceneLeft  = buildScene("ball");
const sceneRight = buildScene("light");

// ============ animation logic ============
const X_MIN = -16, X_MAX = +18;     // wrap range for projectile in runner frame
const PROJECTILE_LOOP = X_MAX - X_MIN;

function applyFrame(scene, t) {
    const beta  = state.beta;
    const vBall = state.vBall;
    const isBall = scene.userData.label === "ball";

    const { projectile, grid, stars, warp } = scene.userData;

    // Apparent speed of the projectile in the runner's frame.
    // For the ball: Galilean v_b - beta. For the photon: ALWAYS c.
    const vRel = isBall ? (vBall - beta) : 1.0;

    // Projectile position in runner frame: spawn at x=0 at t=0; loop along x.
    let x = vRel * t;
    // wrap into the visible range so we always see the projectile flying
    x = ((x - X_MIN) % PROJECTILE_LOOP + PROJECTILE_LOOP) % PROJECTILE_LOOP + X_MIN;
    projectile.position.x = x;

    // For the ball: when it's moving backward (vRel < 0), flip its trail
    // so the streak still points opposite to its motion.
    if (isBall) {
        for (const ch of projectile.children) {
            // child[2] is the trail cone (placed at -0.95 by default)
            // Identify by its negative initial x
            if (ch.geometry && ch.geometry.type === "ConeGeometry") {
                ch.scale.x = vRel < 0 ? -1 : 1;
            }
        }
    }

    // Scroll the neon grid to convey ship motion through the world.
    grid.userData.mat.uniforms.uOffsetX.value = -beta * t * 6.0;

    // Stream the starfield along +x at speed proportional to beta.
    const positions = stars.geometry.attributes.position.array;
    const dx = beta * 1.0 / 60.0;     // approx per-frame shift (cosmetic)
    for (let i = 0; i < positions.length; i += 3) {
        positions[i] -= dx * 60 * 0.016;
        if (positions[i] < -60) positions[i] += 120;
    }
    stars.geometry.attributes.position.needsUpdate = true;

    // Stream the warp lines (more dramatic at high beta).
    for (const m of warp.children) {
        m.position.x -= beta * 0.55;
        if (m.position.x < -55) {
            m.position.x = 55;
            m.position.y = Math.random() * 16 + 0.5;
            m.position.z = -Math.random() * 30 - 4;
        }
        // brightness scales with beta
        m.material.opacity = 0.25 + 0.6 * Math.min(1, beta * 1.6);
    }
}

// ============ HUD update ============
function updateHUD() {
    const beta  = state.beta;
    const vBall = state.vBall;

    $("vBeta").textContent = beta.toFixed(2) + " c";

    const vL = vBall - beta;
    const vR = 1.0;     // c, always

    $("vSpeedL").textContent =
        (vL >= 0 ? "" : "-") + Math.abs(vL).toFixed(2) + " c";
    $("vSpeedR").textContent = vR.toFixed(2) + " c";

    // Bars: scaled to vBall (left) and 1.0 (right).
    const barL = $("barL");
    const barR = $("barR");
    barL.style.width = (Math.max(0, Math.min(1, Math.abs(vL))) * 100) + "%";
    barL.style.background = vL >= 0
        ? "linear-gradient(90deg, #ff6ad5, #ffd3f1)"
        : "linear-gradient(90deg, #ff8a5f, #ffc97a)";
    barR.style.width = "100%";

    // verdict
    const vL_el = $("verdictL");
    if (vL > 0.05)            vL_el.textContent = "▶ ball drifts ahead at " + vL.toFixed(2) + " c";
    else if (vL > -0.05)      vL_el.textContent = "✋ caught it! ball is at rest in your frame";
    else                       vL_el.textContent = "◀ you've overtaken it!";
    $("verdictR").textContent = "▶ light escapes at full c — always";

    // takeaway message
    if (beta < 0.02) {
        takeawayEl.innerHTML =
            "Drag the slider. <span class='punch'>The ball slows, the light doesn't.</span>";
    } else if (beta < vBall - 0.02) {
        takeawayEl.innerHTML =
            "Speed up — you're closing in on the ball but " +
            "<span class='punch'>not on the light</span>.";
    } else if (beta < vBall + 0.05) {
        takeawayEl.innerHTML =
            "<b>You caught the ball</b> — but the light is still racing away at c. " +
            "<span class='punch'>That's the postulate.</span>";
    } else {
        takeawayEl.innerHTML =
            "You've <b>overtaken the ball</b>. " +
            "Light still pulls away at <span class='punch'>exactly c.</span> " +
            "No matter how fast you fly.";
    }
}

// ============ UI ============
function bindRange(id, valId, key, fmt = (v) => v.toFixed(2), onChange = null) {
    const r = $(id);
    const apply = () => {
        state[key] = parseFloat(r.value);
        if (onChange) onChange(state[key]);
        updateHUD();
    };
    r.addEventListener("input", apply);
    apply();
}
bindRange("rBeta", "vBeta", "beta");

// ============ resize ============
function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h, true);
    camera.aspect = (w / 2) / h;
    camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// ============ render loop ============
let prev = performance.now();
let frames = 0, fpsClock = prev;

function render() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const halfW = Math.floor(w / 2);

    renderer.setScissorTest(true);

    // LEFT: classical ball
    renderer.setViewport(0, 0, halfW, h);
    renderer.setScissor (0, 0, halfW, h);
    renderer.render(sceneLeft, camera);

    // RIGHT: light beam
    renderer.setViewport(halfW, 0, w - halfW, h);
    renderer.setScissor (halfW, 0, w - halfW, h);
    renderer.render(sceneRight, camera);

    renderer.setScissorTest(false);
}

function tick(now) {
    const dt_real = (now - prev) * 0.001;
    prev = now;

    if (state.playing) {
        state.t += dt_real * state.timeSpeed;
        if (state.t > 60) state.t = 0;
    }

    applyFrame(sceneLeft,  state.t);
    applyFrame(sceneRight, state.t);

    render();

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

// initial
updateHUD();
