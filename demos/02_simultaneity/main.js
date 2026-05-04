// Relativity of simultaneity -- 3D Three.js dual-frame demo.
//
// Two side-by-side panels (split-screen via WebGL scissor):
//   LEFT  (Lab frame)   - Dad stands on platform; train zooms past at speed beta.
//   RIGHT (Train frame) - Jack stands on the train; the world slides past.
//
// Both panels show the same two events (lightning bolts at the train's
// front and back). In the Lab frame the strikes are simultaneous; in the
// Train frame the front strike precedes the back strike by Delta t' = beta * L.
//
// Geometric units c = 1.

import * as THREE from "three";

// ============ DOM hooks ============
const $       = (id) => document.getElementById(id);
const canvas  = document.getElementById("gl");
const tip     = $("tip");
const stats   = $("stats");
const takeawayEl = $("takeaway");
const verdictGEl = $("verdictG");
const verdictTEl = $("verdictT");

// ============ state ============
const state = {
    beta:      0.5,
    L:         6.0,
    t:        -3.0,
    timeSpeed: 1.0,
    playing:   true,
};

const FLASH_DUR = 0.45;       // simulation seconds the lightning bolt is visible
const PULSE_FADE = 8.0;       // distance over which a pulse ring fades out
const T_RESET_HI =  10.0;
const T_RESET_LO = -3.0;

// ============ renderer ============
const renderer = new THREE.WebGLRenderer({
    canvas, antialias: true, powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping      = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type    = THREE.PCFSoftShadowMap;
renderer.autoClear = true;

// One camera reused for both halves.
const camera = new THREE.PerspectiveCamera(42, 1.0, 0.1, 200);
camera.position.set(0, 1.7, 20);
camera.lookAt(0, 0.4, 0);

// ============ scene builders ============
function makeSky() {
    // Surreal dusk/twilight palette: deep magenta zenith → coral mid → amber horizon
    const skyGeo = new THREE.SphereGeometry(80, 48, 24);
    const skyMat = new THREE.ShaderMaterial({
        side: THREE.BackSide,
        uniforms: {
            uTop:    { value: new THREE.Color(0x4a2380) },   // deep purple
            uHi:     { value: new THREE.Color(0x9a3f9c) },   // magenta band
            uMid:    { value: new THREE.Color(0xff8a6a) },   // coral
            uBottom: { value: new THREE.Color(0xffd58a) },   // amber horizon
        },
        vertexShader: `
            varying vec3 vP;
            void main() {
                vP = position;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }`,
        fragmentShader: `
            uniform vec3 uTop, uHi, uMid, uBottom;
            varying vec3 vP;
            // simple hash-noise for subtle gradient banding
            float h21(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5);}
            void main() {
                float h = normalize(vP).y;
                vec3 c = mix(uBottom, uMid, smoothstep(-0.10, 0.25, h));
                c      = mix(c,       uHi,  smoothstep( 0.20, 0.55, h));
                c      = mix(c,       uTop, smoothstep( 0.55, 0.95, h));
                // subtle painterly noise to break up the gradient banding
                vec2 n2 = vP.xz * 0.04;
                c += (h21(floor(n2 * 12.0)) - 0.5) * 0.015;
                gl_FragColor = vec4(c, 1.0);
            }`,
        depthWrite: false,
    });
    return new THREE.Mesh(skyGeo, skyMat);
}

// Three colored moons floating in the sky.
function makeMoons() {
    const grp = new THREE.Group();
    const moons = [
        { pos: [-26, 16, -32], r: 2.4, c: 0xffd87a, em: 0x6a4a18 }, // big amber
        { pos: [ 22, 22, -38], r: 1.6, c: 0xe6a3d4, em: 0x5a2a4a }, // pink
        { pos: [  6, 28, -45], r: 1.0, c: 0xa9c8ff, em: 0x36486a }, // lavender
    ];
    for (const m of moons) {
        const moon = new THREE.Mesh(
            new THREE.SphereGeometry(m.r, 36, 24),
            new THREE.MeshStandardMaterial({
                color: m.c, emissive: m.em, emissiveIntensity: 0.45,
                roughness: 0.9,
            }));
        moon.position.set(...m.pos);
        grp.add(moon);
    }
    // ringed planet (Saturn-like) high in the sky
    const planet = new THREE.Mesh(
        new THREE.SphereGeometry(1.4, 32, 22),
        new THREE.MeshStandardMaterial({
            color: 0xffb674, emissive: 0x4a2410, emissiveIntensity: 0.35,
            roughness: 0.8,
        }));
    planet.position.set(-15, 24, -42);
    grp.add(planet);
    const ring = new THREE.Mesh(
        new THREE.RingGeometry(1.9, 2.7, 64, 1),
        new THREE.MeshBasicMaterial({
            color: 0xf2c890, transparent: true, opacity: 0.55,
            side: THREE.DoubleSide,
        }));
    ring.position.copy(planet.position);
    ring.rotation.x = Math.PI / 2.4;
    ring.rotation.z = 0.3;
    grp.add(ring);

    return grp;
}

// A few Dali-esque clocks floating in the air, slowly rotating. Cute nod
// to the relativity-of-time theme.
function makeFloatingClocks() {
    const grp = new THREE.Group();

    const faceTex = makeClockFaceTexture();
    const faceMat = new THREE.MeshStandardMaterial({
        map: faceTex, roughness: 0.55, metalness: 0.25,
    });
    const sideMat = new THREE.MeshStandardMaterial({
        color: 0xc9b06b, roughness: 0.4, metalness: 0.7,
    });
    const handMat = new THREE.MeshStandardMaterial({
        color: 0x1a1a1a, roughness: 0.7,
    });

    const placements = [
        { pos: [ -8,  9, -16], r: 0.85, melt: 0.3 },
        { pos: [  9, 11, -18], r: 0.65, melt: 0.5 },
        { pos: [ -2, 14, -22], r: 0.50, melt: 0.0 },
    ];

    for (const p of placements) {
        const clk = new THREE.Group();

        // face: short cylinder textured front + flat side
        const face = new THREE.Mesh(
            new THREE.CylinderGeometry(p.r, p.r, 0.12, 36),
            [sideMat, faceMat, faceMat]);
        face.rotation.x = Math.PI / 2;     // face front toward camera
        clk.add(face);

        // hour hand
        const hourHand = new THREE.Mesh(
            new THREE.BoxGeometry(0.04, p.r * 0.55, 0.02), handMat);
        hourHand.position.set(0, p.r * 0.27, 0.08);
        clk.add(hourHand);
        // minute hand (rotated)
        const minuteHand = new THREE.Mesh(
            new THREE.BoxGeometry(0.03, p.r * 0.78, 0.02), handMat);
        minuteHand.rotation.z = 0.7;
        minuteHand.position.set(p.r * 0.27, p.r * 0.30, 0.08);
        clk.add(minuteHand);

        // center pin (gold)
        const pin = new THREE.Mesh(
            new THREE.SphereGeometry(0.06, 14, 10), sideMat);
        pin.position.set(0, 0, 0.08);
        clk.add(pin);

        // optional Dali "melt" — a hanging droplet drape under the clock
        if (p.melt > 0) {
            const drape = new THREE.Mesh(
                new THREE.ConeGeometry(p.r * 0.25, p.r * (1.0 + p.melt), 8),
                sideMat);
            drape.position.set(p.r * 0.4, -p.r * (0.5 + p.melt * 0.5), 0);
            drape.rotation.z = Math.PI;
            clk.add(drape);
        }

        clk.position.set(...p.pos);
        clk.rotation.y = (Math.random() - 0.5) * 0.6;
        clk.userData.spinAxis = new THREE.Vector3(
            (Math.random() - 0.5),
            1,
            (Math.random() - 0.5)).normalize();
        clk.userData.spinSpeed = 0.05 + Math.random() * 0.10;
        clk.userData.bobPhase  = Math.random() * Math.PI * 2;
        clk.userData.bobAmp    = 0.08 + Math.random() * 0.10;     // smaller bob
        clk.userData.bobBase   = p.pos[1];
        grp.add(clk);
    }

    return grp;
}

function makeClockFaceTexture() {
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    const g = c.getContext("2d");
    const S = 256, cx = S / 2, cy = S / 2;

    // ivory face background
    g.fillStyle = "#f4ead2"; g.fillRect(0, 0, S, S);
    // outer ring
    g.lineWidth = 8; g.strokeStyle = "#a07a30";
    g.beginPath(); g.arc(cx, cy, S * 0.46, 0, Math.PI * 2); g.stroke();
    // hour marks (12)
    g.strokeStyle = "#1a1208";
    for (let i = 0; i < 12; i++) {
        const a = (i / 12) * Math.PI * 2 - Math.PI / 2;
        const r1 = S * 0.40, r2 = S * 0.43;
        const w = (i % 3 === 0) ? 6 : 3;
        g.lineWidth = w;
        g.beginPath();
        g.moveTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
        g.lineTo(cx + Math.cos(a) * r2, cy + Math.sin(a) * r2);
        g.stroke();
    }
    // roman numerals at 12, 3, 6, 9
    g.fillStyle = "#1a1208";
    g.font = "bold 28px serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText("XII", cx, cy - S * 0.34);
    g.fillText("III", cx + S * 0.34, cy);
    g.fillText("VI",  cx, cy + S * 0.34);
    g.fillText("IX",  cx - S * 0.34, cy);

    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
}

// Dr. Seuss spiral-stripe cone mountains in the distance.
function makeSpiralStripeTex(baseHex, stripeHex) {
    const c = document.createElement("canvas");
    c.width = 256; c.height = 256;
    const g = c.getContext("2d");
    g.fillStyle = baseHex; g.fillRect(0, 0, 256, 256);
    g.strokeStyle = stripeHex;
    g.lineWidth = 22;
    g.lineCap = "round";
    for (let i = -300; i < 600; i += 50) {
        g.beginPath();
        g.moveTo(i, 0);
        g.lineTo(i + 256, 256);
        g.stroke();
    }
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    return tex;
}

function makeSurrealMountains() {
    const grp = new THREE.Group();
    const rng = mulberry32(54321);

    const palettes = [
        ["#5a2a8c", "#c478ff"],   // purple
        ["#1a4a7a", "#5fb8e6"],   // teal-blue
        ["#7a2a5a", "#ff8acc"],   // magenta
        ["#3a6a3a", "#a8e07a"],   // lime
        ["#7a3a1a", "#e6a85a"],   // copper
    ];
    const textures = palettes.map(([a, b]) => makeSpiralStripeTex(a, b));

    for (let i = 0; i < 16; i++) {
        const tx = -38 + i * 4.6 + (rng() - 0.5) * 1.8;
        const tz = -22 - rng() * 8;
        const th = 4 + rng() * 6;
        const tr = 0.9 + rng() * 0.6;

        const tex = textures[Math.floor(rng() * textures.length)];
        const mat = new THREE.MeshStandardMaterial({
            map: tex, roughness: 0.9,
        });
        const cone = new THREE.Mesh(
            new THREE.ConeGeometry(tr, th, 18, 6), mat);
        cone.position.set(tx, -2.0 + th / 2, tz);
        // playful lean
        cone.rotation.z = (rng() - 0.5) * 0.45;
        cone.rotation.x = (rng() - 0.5) * 0.10;
        cone.castShadow    = true;
        cone.receiveShadow = true;
        grp.add(cone);
    }

    return grp;
}

// Procedural grass canvas texture: hi-res, subtle, multi-octave variation.
function makeGrassTexture(size) {
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const g = c.getContext("2d");

    // base gradient (slight variation in base hue)
    const grad = g.createLinearGradient(0, 0, size, size);
    grad.addColorStop(0,   "#4d7a3a");
    grad.addColorStop(0.5, "#587f3d");
    grad.addColorStop(1,   "#4f7438");
    g.fillStyle = grad; g.fillRect(0, 0, size, size);

    // large soft color patches for low-frequency variation
    for (let i = 0; i < 80; i++) {
        const x = Math.random() * size;
        const y = Math.random() * size;
        const r = size * (0.06 + Math.random() * 0.16);
        const hue = 95 + Math.random() * 30;
        const sat = 25 + Math.random() * 25;
        const lum = 22 + Math.random() * 20;
        const rg = g.createRadialGradient(x, y, 0, x, y, r);
        rg.addColorStop(0,   `hsla(${hue},${sat}%,${lum}%,0.55)`);
        rg.addColorStop(1,   `hsla(${hue},${sat}%,${lum}%,0.0)`);
        g.fillStyle = rg;
        g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill();
    }

    // medium speckles (small clumps)
    for (let i = 0; i < 1400; i++) {
        const x = Math.random() * size;
        const y = Math.random() * size;
        const r = 2 + Math.random() * 4;
        const hue = 80 + Math.random() * 50;
        const lum = 20 + Math.random() * 30;
        g.fillStyle = `hsla(${hue},45%,${lum}%,0.5)`;
        g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill();
    }

    // fine grass-blade flecks
    for (let i = 0; i < 6000; i++) {
        const x = Math.random() * size;
        const y = Math.random() * size;
        const len = 1.5 + Math.random() * 4;
        const hue = 88 + Math.random() * 45;
        const lum = 26 + Math.random() * 32;
        g.strokeStyle = `hsla(${hue},55%,${lum}%,0.7)`;
        g.lineWidth = 0.7 + Math.random() * 0.6;
        g.beginPath();
        g.moveTo(x, y);
        g.lineTo(x + (Math.random() - 0.5) * 2.5, y - len);
        g.stroke();
    }

    // tiny dirt specks
    for (let i = 0; i < 400; i++) {
        const x = Math.random() * size;
        const y = Math.random() * size;
        const r = 0.6 + Math.random() * 1.6;
        g.fillStyle = `hsla(${30 + Math.random()*15},35%,${22 + Math.random()*15}%,0.55)`;
        g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill();
    }

    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 16;
    return tex;
}

function makeGround() {
    const groundTex = makeGrassTexture(2048);
    groundTex.wrapS = groundTex.wrapT = THREE.RepeatWrapping;
    groundTex.repeat.set(3, 1.5);     // less obvious tiling

    const g   = new THREE.PlaneGeometry(180, 90, 1, 1);
    const mat = new THREE.MeshStandardMaterial({
        map: groundTex, roughness: 0.95, metalness: 0.0,
    });
    const m = new THREE.Mesh(g, mat);
    m.rotation.x = -Math.PI / 2;
    m.position.y = -2.0;
    m.receiveShadow = true;
    m.userData.tex = groundTex;
    return m;
}

// Bushes + distant trees as a separate group so they can slide past
// in the train frame.
function makeFoliage() {
    const grp = new THREE.Group();

    const bushMat  = new THREE.MeshStandardMaterial({ color: 0x355d24, roughness: 0.95 });
    const bushMat2 = new THREE.MeshStandardMaterial({ color: 0x4f8233, roughness: 0.95 });
    const rng = mulberry32(12345);
    for (let i = 0; i < 28; i++) {
        const bx = (rng() * 2 - 1) * 28;
        const bz = 6 + rng() * 18;
        const sgn = rng() > 0.5 ? 1 : -1;
        const cluster = new THREE.Group();
        const r = 0.35 + rng() * 0.5;
        const a = new THREE.Mesh(new THREE.IcosahedronGeometry(r, 0), bushMat);
        cluster.add(a);
        const b = new THREE.Mesh(new THREE.IcosahedronGeometry(r * 0.7, 0), bushMat2);
        b.position.set(r * 0.6, r * 0.1, r * 0.2);
        cluster.add(b);
        const c = new THREE.Mesh(new THREE.IcosahedronGeometry(r * 0.55, 0), bushMat);
        c.position.set(-r * 0.5, r * 0.05, r * 0.3);
        cluster.add(c);
        cluster.position.set(bx, -2.0 + r * 0.7, bz * sgn);
        cluster.children.forEach(ch => { ch.castShadow = true;
                                         ch.receiveShadow = true; });
        grp.add(cluster);
    }

    grp.add(makeSurrealMountains());

    return grp;
}

// Small deterministic RNG so bushes/trees don't twitch frame-to-frame.
function mulberry32(seed) {
    let s = seed >>> 0;
    return function() {
        s = (s + 0x6D2B79F5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function makePlatform() {
    const grp = new THREE.Group();

    const slab = new THREE.Mesh(
        new THREE.BoxGeometry(36, 0.6, 4),
        new THREE.MeshStandardMaterial({ color: 0xb0b6c0, roughness: 0.7 })
    );
    slab.position.set(0, -1.7, -3.2);
    slab.receiveShadow = true;
    slab.castShadow    = true;
    grp.add(slab);

    // yellow safety stripe
    const stripe = new THREE.Mesh(
        new THREE.BoxGeometry(36, 0.02, 0.25),
        new THREE.MeshStandardMaterial({
            color: 0xffd54a, emissive: 0x4a3c0a, roughness: 0.6,
        })
    );
    stripe.position.set(0, -1.39, -1.3);
    grp.add(stripe);

    // pillars
    for (let i = -3; i <= 3; i++) {
        const p = new THREE.Mesh(
            new THREE.CylinderGeometry(0.10, 0.10, 2.4, 12),
            new THREE.MeshStandardMaterial({ color: 0x8a8f97, roughness: 0.8 })
        );
        p.position.set(i * 5, -0.4, -4.5);
        p.castShadow = true;
        grp.add(p);
    }

    return grp;
}

function makeTrack() {
    const grp = new THREE.Group();

    const railMat = new THREE.MeshStandardMaterial({
        color: 0x9aa1aa, roughness: 0.4, metalness: 0.7,
    });
    const railGeo = new THREE.BoxGeometry(36, 0.08, 0.12);
    for (const z of [-0.5, 0.5]) {
        const r = new THREE.Mesh(railGeo, railMat);
        r.position.set(0, -1.55, z);
        r.receiveShadow = true;
        grp.add(r);
    }

    // sleepers
    const sleepGeo = new THREE.BoxGeometry(0.4, 0.09, 1.6);
    const sleepMat = new THREE.MeshStandardMaterial({
        color: 0x4a3520, roughness: 0.9,
    });
    for (let i = -16; i <= 16; i++) {
        const s = new THREE.Mesh(sleepGeo, sleepMat);
        s.position.set(i * 1.1, -1.62, 0);
        s.receiveShadow = true;
        grp.add(s);
    }

    return grp;
}

// Modern bullet-train. Returns a Group whose origin is the train's center.
function makeTrain() {
    const grp = new THREE.Group();

    const bodyLen = 1.0;       // unit length; scaled by group at apply time
    const bodyHi  = 1.05;
    const bodyDp  = 1.4;

    const bodyMat = new THREE.MeshStandardMaterial({
        color: 0xf0f4f8, roughness: 0.35, metalness: 0.55,
    });
    const accentMat = new THREE.MeshStandardMaterial({
        color: 0x2d6ee0, roughness: 0.45, metalness: 0.6,
    });
    const noseMat = new THREE.MeshStandardMaterial({
        color: 0xeef2f6, roughness: 0.3, metalness: 0.7,
    });

    // body (will be scaled along x by the apply step).
    // body.position.y is chosen so wheel bottoms land on the rail tops:
    //   wheel bottom = body.position.y - bodyHi/2 - 0.12 - wheelR(0.18) = -1.51
    //   => body.position.y = -0.685
    const body = new THREE.Mesh(
        new THREE.BoxGeometry(bodyLen, bodyHi, bodyDp),
        bodyMat
    );
    body.position.y = -0.685;
    body.castShadow = true;
    body.receiveShadow = true;
    body.name = "body";
    grp.add(body);

    // blue accent stripe along the side
    const stripe = new THREE.Mesh(
        new THREE.BoxGeometry(bodyLen, 0.18, bodyDp + 0.01),
        accentMat
    );
    stripe.position.y = body.position.y + 0.06;
    stripe.name = "stripe";
    grp.add(stripe);

    // skirt below windows (dark)
    const skirt = new THREE.Mesh(
        new THREE.BoxGeometry(bodyLen, 0.20, bodyDp + 0.005),
        new THREE.MeshStandardMaterial({ color: 0x1a2230, roughness: 0.7 })
    );
    skirt.position.y = body.position.y - bodyHi / 2 + 0.10;
    skirt.name = "skirt";
    grp.add(skirt);

    // nose cones at both ends (we will reposition with apply scale)
    const noseGeo = new THREE.ConeGeometry(0.55, 0.7, 16);
    const noseFront = new THREE.Mesh(noseGeo, noseMat);
    noseFront.rotation.z = -Math.PI / 2;
    noseFront.position.y = body.position.y - 0.05;
    noseFront.castShadow = true;
    noseFront.name = "noseFront";
    grp.add(noseFront);

    const noseBack = new THREE.Mesh(noseGeo, noseMat);
    noseBack.rotation.z = Math.PI / 2;
    noseBack.position.y = body.position.y - 0.05;
    noseBack.castShadow = true;
    noseBack.name = "noseBack";
    grp.add(noseBack);

    // windows: a row of small tinted panes (procedurally placed in apply)
    const windowsGroup = new THREE.Group();
    windowsGroup.name = "windows";
    grp.add(windowsGroup);

    // headlight beam plates (front + back)
    const headlightGeo = new THREE.PlaneGeometry(0.14, 0.06);
    const headlightMat = new THREE.MeshBasicMaterial({
        color: 0xfffbe5, transparent: true, opacity: 0.9,
    });
    const hF = new THREE.Mesh(headlightGeo, headlightMat); hF.name = "headF"; grp.add(hF);
    const hB = new THREE.Mesh(headlightGeo, headlightMat); hB.name = "headB"; grp.add(hB);

    // wheels (procedurally placed in apply)
    const wheelGroup = new THREE.Group();
    wheelGroup.name = "wheels";
    grp.add(wheelGroup);

    return grp;
}

// Lay out a train group's geometry for a given rest length L (world units).
function applyTrainLength(trainGrp, L) {
    const bodyLen = L;          // total straight body length (excluding nose)
    const bodyHi  = 1.05;
    const bodyDp  = 1.4;

    const body  = trainGrp.getObjectByName("body");
    const stripe = trainGrp.getObjectByName("stripe");
    const skirt = trainGrp.getObjectByName("skirt");
    body.scale.x = bodyLen;
    stripe.scale.x = bodyLen;
    skirt.scale.x = bodyLen;

    const noseF = trainGrp.getObjectByName("noseFront");
    const noseB = trainGrp.getObjectByName("noseBack");
    noseF.position.x =  bodyLen / 2 + 0.35;
    noseB.position.x = -bodyLen / 2 - 0.35;

    const hF = trainGrp.getObjectByName("headF");
    const hB = trainGrp.getObjectByName("headB");
    hF.position.set( bodyLen / 2 + 0.71, body.position.y - 0.05,  0.71);
    hF.rotation.y = -Math.PI / 2;
    hB.position.set(-bodyLen / 2 - 0.71, body.position.y - 0.05,  0.71);
    hB.rotation.y =  Math.PI / 2;

    // windows + door divisions + roof LED
    const win = trainGrp.getObjectByName("windows");
    win.clear();
    const winMat = new THREE.MeshPhysicalMaterial({
        color: 0x16242e, roughness: 0.05, metalness: 0.0,
        transparent: true, opacity: 0.85,
    });
    const winFrameMat = new THREE.MeshStandardMaterial({
        color: 0xc7ced8, roughness: 0.35, metalness: 0.65,
    });
    const winGeo   = new THREE.BoxGeometry(0.55, 0.42, 0.02);
    const winFrame = new THREE.BoxGeometry(0.60, 0.46, 0.014);
    const ny = body.position.y + 0.18;
    const margin = 0.55;
    const gap    = 0.18;
    const each   = 0.55;
    const span   = bodyLen - margin * 2;
    const count  = Math.max(1, Math.floor((span + gap) / (each + gap)));
    for (let i = 0; i < count; i++) {
        const x = -bodyLen / 2 + margin + each / 2 + i * (each + gap);
        for (const sgn of [+1, -1]) {
            const fr = new THREE.Mesh(winFrame, winFrameMat);
            fr.position.set(x, ny, sgn * (bodyDp / 2 + 0.003));
            win.add(fr);
            const w = new THREE.Mesh(winGeo, winMat);
            w.position.set(x, ny, sgn * (bodyDp / 2 + 0.012));
            win.add(w);
        }
    }

    // door divisions (vertical seam at midpoint and ends)
    const seamGeo = new THREE.BoxGeometry(0.025, 1.0, 0.01);
    const seamMat = new THREE.MeshStandardMaterial({
        color: 0x0f1822, roughness: 0.6, metalness: 0.3,
    });
    const seamXs = [-bodyLen * 0.33, 0.0, bodyLen * 0.33];
    for (const sx of seamXs) {
        for (const sz of [+1, -1]) {
            const s = new THREE.Mesh(seamGeo, seamMat);
            s.position.set(sx, body.position.y, sz * (bodyDp / 2 + 0.008));
            win.add(s);
        }
    }

    // roof LED accent strip (emissive teal)
    const ledMat = new THREE.MeshStandardMaterial({
        color: 0x55c8ff, emissive: 0x1f7fb0, emissiveIntensity: 0.8,
        roughness: 0.4,
    });
    for (const sz of [+1, -1]) {
        const led = new THREE.Mesh(
            new THREE.BoxGeometry(bodyLen - 0.4, 0.04, 0.04), ledMat);
        led.position.set(0, body.position.y + 0.46, sz * (bodyDp / 2 + 0.002));
        win.add(led);
    }

    // wheels (cylinders below skirt at four positions)
    const wheels = trainGrp.getObjectByName("wheels");
    wheels.clear();
    const wMat = new THREE.MeshStandardMaterial({
        color: 0x202830, roughness: 0.5, metalness: 0.6,
    });
    const wGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.10, 18);
    const wy   = body.position.y - bodyHi / 2 - 0.12;
    const wxs  = [-bodyLen * 0.35, -bodyLen * 0.10, bodyLen * 0.10, bodyLen * 0.35];
    for (const wx of wxs) {
        for (const wz of [-0.5, 0.5]) {     // sit on the rails (z = ±0.5)
            const w = new THREE.Mesh(wGeo, wMat);
            w.rotation.x = Math.PI / 2;     // axle along z so wheel rolls in x
            w.position.set(wx, wy, wz);
            w.castShadow = true;
            wheels.add(w);
        }
    }
}

// Cowboy-Bebop-style face texture — narrow almond eyes with sharp upper-lid
// strokes, angular brows, and a cool smirk. Drawn onto a transparent canvas
// so it can sit on a flat plane in front of a sphere head.
function makeAnimeFaceTexture(kind) {
    const c = document.createElement("canvas");
    c.width = c.height = 512;
    const g = c.getContext("2d");
    const S = 512;
    const cx = S / 2, cy = S / 2;
    const isKid = kind === "kid";

    const irisCol  = "#5a3a20";       // brown for both
    const irisHi   = "#8a5e36";       // warm rim highlight
    const browCol  = "#0d0a06";
    const lidCol   = "#0c0c0c";

    const eyeY    = cy + 22;
    const eyeOff  = 82;
    const eyeW    = 70;
    const eyeH    = isKid ? 30 : 24;     // kid slightly taller (more open)

    // ---- Eyes (almond shape via bezier) ----
    for (const sgn of [-1, 1]) {
        const ex = cx + sgn * eyeOff;

        // sclera (white almond)
        g.fillStyle = "#fafafa";
        g.beginPath();
        g.moveTo(ex - eyeW, eyeY + 4);
        g.bezierCurveTo(
            ex - eyeW * 0.55, eyeY - eyeH * 1.05,
            ex + eyeW * 0.55, eyeY - eyeH * 1.05,
            ex + eyeW, eyeY + 4);
        g.bezierCurveTo(
            ex + eyeW * 0.55, eyeY + eyeH * 0.65,
            ex - eyeW * 0.55, eyeY + eyeH * 0.65,
            ex - eyeW, eyeY + 4);
        g.fill();

        // iris (warm brown), large and partly under upper lid
        const irisR = Math.min(eyeW * 0.32, eyeH * 1.1);
        g.fillStyle = irisCol;
        g.beginPath();
        g.ellipse(ex, eyeY + 2, irisR, irisR * 1.25, 0, 0, Math.PI * 2);
        g.fill();

        // warm iris rim highlight (gives that hand-drawn brown depth)
        const ig = g.createRadialGradient(ex - irisR * 0.3, eyeY - irisR * 0.4, 1,
                                          ex, eyeY + 2, irisR * 1.3);
        ig.addColorStop(0,    irisHi);
        ig.addColorStop(0.45, irisCol);
        ig.addColorStop(1,    "#2a180a");
        g.fillStyle = ig;
        g.beginPath();
        g.ellipse(ex, eyeY + 2, irisR * 0.95, irisR * 1.18, 0, 0, Math.PI * 2);
        g.fill();

        // pupil (vertical slit-ish ellipse)
        g.fillStyle = "#080604";
        g.beginPath();
        g.ellipse(ex, eyeY + 2, irisR * 0.32, irisR * 0.55, 0, 0, Math.PI * 2);
        g.fill();

        // upper-lid shadow under the heavy line (subtle)
        g.fillStyle = "rgba(40,25,15,0.40)";
        g.beginPath();
        g.moveTo(ex - eyeW * 0.95, eyeY - 2);
        g.bezierCurveTo(
            ex - eyeW * 0.5, eyeY - eyeH * 0.95,
            ex + eyeW * 0.5, eyeY - eyeH * 0.95,
            ex + eyeW * 0.95, eyeY - 2);
        g.lineTo(ex + eyeW * 0.95, eyeY + 4);
        g.bezierCurveTo(
            ex + eyeW * 0.5, eyeY - eyeH * 0.45,
            ex - eyeW * 0.5, eyeY - eyeH * 0.45,
            ex - eyeW * 0.95, eyeY + 4);
        g.closePath();
        g.fill();

        // heavy upper eyelid stroke (Bebop trademark)
        g.lineWidth   = 8;
        g.lineCap     = "round";
        g.strokeStyle = lidCol;
        g.beginPath();
        g.moveTo(ex - eyeW * 1.05, eyeY + 4);
        g.bezierCurveTo(
            ex - eyeW * 0.5, eyeY - eyeH * 1.10,
            ex + eyeW * 0.5, eyeY - eyeH * 1.10,
            ex + eyeW * 1.05, eyeY - 2);
        g.stroke();

        // outer corner flick
        g.lineWidth = 5;
        g.beginPath();
        g.moveTo(ex + eyeW * sgn, eyeY + sgn * 4);
        g.lineTo(ex + eyeW * sgn * 1.30, eyeY + 10);
        g.stroke();

        // tiny single highlight (the cool, restrained kind)
        g.fillStyle = "#ffffff";
        g.beginPath();
        g.ellipse(ex - irisR * 0.45, eyeY - irisR * 0.55,
                  irisR * 0.18, irisR * 0.28, 0, 0, Math.PI * 2);
        g.fill();
    }

    // ---- Eyebrows: angular, slightly slanted (Spike-style) ----
    g.lineWidth   = 13;
    g.lineCap     = "round";
    g.lineJoin    = "round";
    g.strokeStyle = browCol;
    for (const sgn of [-1, 1]) {
        const ex = cx + sgn * eyeOff;
        const lift = isKid ? 6 : 0;
        g.beginPath();
        g.moveTo(ex - eyeW * 0.95, eyeY - eyeH - 30 + lift);
        g.lineTo(ex - eyeW * 0.10, eyeY - eyeH - 38 + lift);
        g.lineTo(ex + eyeW * 0.95, eyeY - eyeH - 22 + lift);
        g.stroke();
    }

    // ---- Nose: minimalist crease ----
    g.lineWidth   = 3;
    g.strokeStyle = "rgba(150,100,70,0.85)";
    g.beginPath();
    g.moveTo(cx - 1, cy + 64);
    g.lineTo(cx + 5, cy + 92);
    g.stroke();
    // tiny nostril hint
    g.fillStyle = "rgba(120,80,55,0.45)";
    g.beginPath(); g.ellipse(cx + 6, cy + 95, 3, 1.6, 0, 0, Math.PI * 2); g.fill();

    // ---- Mouth ----
    g.lineWidth   = 5;
    g.lineCap     = "round";
    g.strokeStyle = "#2a1410";
    g.beginPath();
    if (isKid) {
        // confident close-mouth grin
        g.moveTo(cx - 28, cy + 118);
        g.quadraticCurveTo(cx, cy + 136, cx + 28, cy + 118);
    } else {
        // asymmetric smirk — one corner higher (Spike style)
        g.moveTo(cx - 30, cy + 122);
        g.quadraticCurveTo(cx - 4, cy + 116, cx + 32, cy + 102);
    }
    g.stroke();
    // subtle lower-lip shadow
    g.lineWidth   = 2;
    g.strokeStyle = "rgba(60,30,25,0.35)";
    g.beginPath();
    g.moveTo(cx - 20, cy + 132);
    g.quadraticCurveTo(cx, cy + 138, cx + 22, cy + 132);
    g.stroke();

    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    return tex;
}

// Chibi-anime character.
//   "adult" -> Dad: blue button-down + slacks + flat cap, beard
//   "kid"   -> Jack: blue hoodie + red pants + sneakers + backpack
function makeCharacter(kind) {
    const grp = new THREE.Group();
    const isKid = kind === "kid";

    // Palette
    const shirtCol  = isKid ? 0x274472 : 0x2e6dd6;
    const shirtCol2 = isKid ? 0x1d3358 : 0x1f4a99;
    const pantsCol  = isKid ? 0xe53935 : 0x222a36;
    const skinCol   = 0xffe0c2;
    const hairCol   = isKid ? 0x4a2e1e : 0x2b1c12;
    const shoeCol   = isKid ? 0xffffff : 0x141414;
    const shoeAcc   = 0xff3344;
    const beltCol   = 0x2a2014;

    // Chibi-anime proportions: oversized head, simple body shapes.
    const headR     = isKid ? 0.30 : 0.32;
    const torsoH    = isKid ? 0.50 : 0.65;
    const torsoR    = isKid ? 0.24 : 0.30;
    const torsoDp   = torsoR * 0.85;
    const legH      = isKid ? 0.40 : 0.55;
    const legR      = isKid ? 0.10 : 0.115;
    const armH      = isKid ? 0.40 : 0.55;
    const armR      = isKid ? 0.085 : 0.10;

    // ---- Legs ----
    const legMat = new THREE.MeshStandardMaterial({ color: pantsCol, roughness: 0.92 });
    const legXOffset = legR * 1.2;
    for (const lx of [-legXOffset, legXOffset]) {
        const l = new THREE.Mesh(
            new THREE.CylinderGeometry(legR * 0.95, legR * 1.05, legH, 18), legMat);
        l.position.set(lx, legH / 2, 0);
        l.castShadow = true;
        grp.add(l);

        const shoe = new THREE.Mesh(
            new THREE.BoxGeometry(legR * 2.6, 0.10, legR * 3.2),
            new THREE.MeshStandardMaterial({
                color: shoeCol, roughness: 0.5, metalness: 0.05,
            }));
        shoe.position.set(lx, 0.05, 0.04);
        shoe.castShadow = true;
        grp.add(shoe);

        if (isKid) {
            const stripe = new THREE.Mesh(
                new THREE.BoxGeometry(legR * 2.65, 0.025, legR * 3.3),
                new THREE.MeshStandardMaterial({ color: shoeAcc, roughness: 0.5 }));
            stripe.position.set(lx, 0.03, 0.04);
            grp.add(stripe);
        }
    }

    // ---- Torso (rounded box for chibi look) ----
    const torsoMat = new THREE.MeshStandardMaterial({
        color: shirtCol, roughness: isKid ? 0.92 : 0.65,
    });
    const torso = new THREE.Mesh(
        new THREE.BoxGeometry(torsoR * 2, torsoH, torsoDp * 2),
        torsoMat);
    torso.position.y = legH + torsoH / 2;
    torso.castShadow = true;
    grp.add(torso);

    // shoulder caps (round off the top corners + provide attachment for arms)
    for (const sgn of [-1, 1]) {
        const sc = new THREE.Mesh(
            new THREE.SphereGeometry(armR * 1.45, 20, 16),
            new THREE.MeshStandardMaterial({
                color: shirtCol, roughness: isKid ? 0.92 : 0.65,
            }));
        sc.position.set(sgn * (torsoR - armR * 0.05),
                        legH + torsoH * 0.92, 0);
        sc.castShadow = true;
        grp.add(sc);
    }

    if (isKid) {
        // ---- Jack: hoodie front panel + pouch + back hood + backpack ----
        const front = new THREE.Mesh(
            new THREE.BoxGeometry(torsoR * 1.85, torsoH * 0.94, 0.04),
            new THREE.MeshStandardMaterial({ color: shirtCol2, roughness: 0.95 }));
        front.position.set(0, legH + torsoH / 2, torsoDp + 0.005);
        grp.add(front);

        const pouch = new THREE.Mesh(
            new THREE.BoxGeometry(torsoR * 1.4, torsoH * 0.30, 0.05),
            new THREE.MeshStandardMaterial({ color: 0x142540, roughness: 0.95 }));
        pouch.position.set(0, legH + torsoH * 0.30, torsoDp + 0.020);
        grp.add(pouch);

        // hood: a flat fabric panel rising at the back of the neck
        const hoodMat = new THREE.MeshStandardMaterial({
            color: shirtCol2, roughness: 0.95,
        });
        const hoodBack = new THREE.Mesh(
            new THREE.BoxGeometry(torsoR * 1.5, torsoH * 0.50, 0.06),
            hoodMat);
        hoodBack.position.set(0, legH + torsoH * 0.95, -torsoDp - 0.04);
        hoodBack.rotation.x = -0.18;
        grp.add(hoodBack);

        // hood collar ring around the back of the neck
        const hoodRing = new THREE.Mesh(
            new THREE.TorusGeometry(torsoR * 0.55, 0.05, 12, 22, Math.PI),
            hoodMat);
        hoodRing.position.set(0, legH + torsoH * 0.96, -torsoDp * 0.3);
        hoodRing.rotation.x = Math.PI / 2;
        hoodRing.rotation.z = Math.PI;
        grp.add(hoodRing);

        // backpack
        const packCol = 0x2e7d32;
        const pack = new THREE.Mesh(
            new THREE.BoxGeometry(torsoR * 1.6, torsoH * 0.65, 0.20),
            new THREE.MeshStandardMaterial({ color: packCol, roughness: 0.85 }));
        pack.position.set(0, legH + torsoH * 0.50, -torsoDp - 0.12);
        pack.castShadow = true;
        grp.add(pack);
    } else {
        // ---- Dad: shirt placket + buttons + belt ----
        const placket = new THREE.Mesh(
            new THREE.BoxGeometry(0.10, torsoH * 0.88, 0.02),
            new THREE.MeshStandardMaterial({ color: shirtCol2, roughness: 0.6 }));
        placket.position.set(0, legH + torsoH * 0.50, torsoDp + 0.005);
        grp.add(placket);

        for (let i = 0; i < 4; i++) {
            const btn = new THREE.Mesh(
                new THREE.SphereGeometry(0.022, 10, 8),
                new THREE.MeshStandardMaterial({
                    color: 0xeeeeee, roughness: 0.4, metalness: 0.2,
                }));
            btn.position.set(0,
                legH + torsoH * 0.80 - i * 0.16,
                torsoDp + 0.014);
            grp.add(btn);
        }

        // belt
        const belt = new THREE.Mesh(
            new THREE.BoxGeometry(torsoR * 2.05, 0.07, torsoDp * 2.05),
            new THREE.MeshStandardMaterial({ color: beltCol, roughness: 0.8 }));
        belt.position.y = legH + 0.04;
        grp.add(belt);
        const buckle = new THREE.Mesh(
            new THREE.BoxGeometry(0.10, 0.07, 0.03),
            new THREE.MeshStandardMaterial({
                color: 0xc9b06b, roughness: 0.3, metalness: 0.85,
            }));
        buckle.position.set(0, legH + 0.04, torsoDp + 0.015);
        grp.add(buckle);
    }

    // ---- Arms (attached at shoulder, no gap) ----
    const armMat = new THREE.MeshStandardMaterial({
        color: shirtCol, roughness: isKid ? 0.92 : 0.65,
    });
    const armX = torsoR + armR * 0.05;     // touching torso edge
    for (const sgn of [-1, 1]) {
        const a = new THREE.Mesh(
            new THREE.CylinderGeometry(armR, armR * 1.1, armH, 16), armMat);
        a.position.set(sgn * armX,
                       legH + torsoH * 0.85 - armH / 2, 0);
        a.castShadow = true;
        grp.add(a);

        const hand = new THREE.Mesh(
            new THREE.SphereGeometry(armR * 1.18, 16, 14),
            new THREE.MeshStandardMaterial({ color: skinCol, roughness: 0.75 }));
        hand.position.set(sgn * armX,
                          legH + torsoH * 0.85 - armH - 0.03, 0);
        hand.scale.set(1.0, 1.05, 0.85);
        grp.add(hand);
    }

    // ---- Head (chibi: large) ----
    const headMat = new THREE.MeshStandardMaterial({ color: skinCol, roughness: 0.85 });
    const head = new THREE.Mesh(new THREE.SphereGeometry(headR, 32, 24), headMat);
    head.position.y = legH + torsoH + 0.04 + headR * 0.95;
    head.scale.set(1.0, 1.05, 0.95);
    head.castShadow = true;
    grp.add(head);
    const hY = head.position.y;

    // ---- Anime face (canvas texture on a flat plane in front of head) ----
    const faceTex = makeAnimeFaceTexture(kind);
    const facePlane = new THREE.Mesh(
        new THREE.PlaneGeometry(headR * 1.85, headR * 1.85),
        new THREE.MeshBasicMaterial({
            map: faceTex, transparent: true, alphaTest: 0.05,
            depthWrite: false,
        }));
    // Slightly in front of the sphere surface to avoid z-fighting.
    facePlane.position.set(0, hY + 0.01, headR * 1.005);
    grp.add(facePlane);

    // ---- Hair / hat ----
    const hairMat = new THREE.MeshStandardMaterial({ color: hairCol, roughness: 0.95 });

    if (isKid) {
        // Jack: anime spiky hair
        const baseHair = new THREE.Mesh(
            new THREE.SphereGeometry(headR * 1.04, 24, 18,
                                     0, Math.PI * 2, 0, Math.PI * 0.42),
            hairMat);
        baseHair.position.copy(head.position);
        baseHair.position.y -= 0.005;
        grp.add(baseHair);

        // five spike tufts at the front-top
        for (const a of [-0.7, -0.3, 0.0, 0.3, 0.7]) {
            const spike = new THREE.Mesh(
                new THREE.ConeGeometry(headR * 0.18, headR * 0.55, 6),
                hairMat);
            spike.position.set(
                Math.sin(a) * headR * 0.7,
                hY + headR * 0.7,
                Math.cos(a) * headR * 0.4 + headR * 0.4);
            spike.rotation.z = -a * 0.6;
            spike.rotation.x = -0.5;
            grp.add(spike);
        }
        // side fringe
        const fringe = new THREE.Mesh(
            new THREE.BoxGeometry(headR * 1.0, headR * 0.25, headR * 0.6),
            hairMat);
        fringe.position.set(headR * 0.20, hY + headR * 0.55, headR * 0.55);
        fringe.rotation.z = 0.30;
        grp.add(fringe);
    } else {
        // Dad: short hair + flat cap
        const baseHair = new THREE.Mesh(
            new THREE.SphereGeometry(headR * 1.02, 24, 18,
                                     0, Math.PI * 2, 0, Math.PI * 0.40),
            hairMat);
        baseHair.position.copy(head.position);
        grp.add(baseHair);

        const capMat = new THREE.MeshStandardMaterial({
            color: 0x4a5563, roughness: 0.9,
        });
        const cap = new THREE.Mesh(
            new THREE.SphereGeometry(headR * 1.10, 24, 16,
                                     0, Math.PI * 2, 0, Math.PI * 0.40),
            capMat);
        cap.position.copy(head.position);
        cap.position.y += 0.02;
        cap.scale.set(1.05, 0.85, 1.05);
        grp.add(cap);
        const brim = new THREE.Mesh(
            new THREE.CylinderGeometry(headR * 1.18, headR * 1.22, 0.03, 24,
                                       1, false, -Math.PI * 0.30, Math.PI * 0.6),
            capMat);
        brim.position.set(0, hY + headR * 0.42, headR * 0.55);
        brim.rotation.x = 0.18;
        grp.add(brim);
    }

    return grp;
}

// One lightning bolt: jagged Line + glowing core + shadow-casting flash light.
function makeLightning() {
    const grp = new THREE.Group();
    grp.visible = false;

    // a jagged path from sky down to the train roof
    const pts = [];
    const segs = 8;
    const yTop = 9.0, yBot = -0.05;       // just above new train roof (~-0.16)
    let x = 0, z = 0;
    for (let i = 0; i <= segs; i++) {
        const tNorm = i / segs;
        const y = yTop + (yBot - yTop) * tNorm;
        x += (Math.random() - 0.5) * 0.6;
        z += (Math.random() - 0.5) * 0.4;
        pts.push(new THREE.Vector3(x, y, z));
    }
    pts[pts.length - 1].set(0, yBot, 0);

    const curve = new THREE.CatmullRomCurve3(pts);
    const tube  = new THREE.Mesh(
        new THREE.TubeGeometry(curve, 60, 0.06, 6, false),
        new THREE.MeshBasicMaterial({ color: 0xffffff }));
    grp.add(tube);

    const halo = new THREE.Mesh(
        new THREE.TubeGeometry(curve, 60, 0.20, 6, false),
        new THREE.MeshBasicMaterial({
            color: 0xa9d6ff, transparent: true, opacity: 0.45,
        }));
    grp.add(halo);

    const flash = new THREE.PointLight(0xc0e8ff, 0.0, 18, 1.6);
    flash.position.set(0, 0.6, 0);     // sits above the train roof
    grp.add(flash);

    grp.userData.tube  = tube;
    grp.userData.halo  = halo;
    grp.userData.flash = flash;
    return grp;
}

// Light pulse: an expanding ring of light (TorusGeometry stretched to a thin disk).
function makePulse(color = 0xffe89a) {
    const geo = new THREE.RingGeometry(0.92, 1.0, 64);
    const mat = new THREE.MeshBasicMaterial({
        color, side: THREE.DoubleSide, transparent: true, opacity: 0.0,
        depthWrite: false,
    });
    const m = new THREE.Mesh(geo, mat);
    m.rotation.x = -Math.PI / 2;
    m.position.y = 0.05;
    m.userData.baseColor = new THREE.Color(color);
    return m;
}

// ============ build the two scenes ============
function buildScene(label) {
    const sc = new THREE.Scene();

    sc.add(makeSky());
    sc.add(makeMoons());
    const clocks = makeFloatingClocks();
    sc.add(clocks);

    // Ground stays on the scene (we'll scroll its texture for motion in
    // the train frame). Bushes/trees go in staticWorld and move bodily.
    const ground = makeGround();
    sc.add(ground);

    // Static-world group: everything attached to the platform that should
    // slide past in the train frame.
    const staticWorld = new THREE.Group();
    staticWorld.add(makePlatform());
    staticWorld.add(makeTrack());
    staticWorld.add(makeFoliage());
    sc.add(staticWorld);

    // Lights
    const hemi = new THREE.HemisphereLight(0xeaf2ff, 0x4a4030, 0.55);
    sc.add(hemi);
    const sun = new THREE.DirectionalLight(0xfff2d0, 1.4);
    sun.position.set(7, 12, 5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.left   = -16;
    sun.shadow.camera.right  =  16;
    sun.shadow.camera.top    =   8;
    sun.shadow.camera.bottom =  -4;
    sun.shadow.camera.near   =  0.5;
    sun.shadow.camera.far    =  40;
    sc.add(sun);

    // Train
    const train = makeTrain();
    sc.add(train);
    applyTrainLength(train, state.L);

    // Characters
    const dad  = makeCharacter("adult");
    const jack = makeCharacter("kid");
    sc.add(dad);
    sc.add(jack);

    // Lightning bolts (one at front, one at back of train)
    const boltFront = makeLightning();
    const boltBack  = makeLightning();
    sc.add(boltFront);
    sc.add(boltBack);

    // Light pulses
    const pulseFront = makePulse(0xffe89a);
    const pulseBack  = makePulse(0xa9d6ff);
    sc.add(pulseFront);
    sc.add(pulseBack);

    sc.userData = { train, dad, jack, boltFront, boltBack,
                    pulseFront, pulseBack, ground, staticWorld, clocks, label };

    return sc;
}

const sceneLab   = buildScene("lab");
const sceneTrain = buildScene("train");

// ============ animation logic ============
//
// Lab frame events (we set t=0 at simultaneous strike):
//   front strike: (t = 0,        x = +L_lab/2 = +L/(2 gamma))
//   back  strike: (t = 0,        x = -L_lab/2 = -L/(2 gamma))
//
// Train frame events (Lorentz transform t' = gamma(t - beta x)):
//   front strike: t' = -beta L / 2,  x' = +L/2
//   back  strike: t' = +beta L / 2,  x' = -L/2
//
// In each panel we use the scrubber as that frame's time variable
// (t in lab, t' in train).
function applyFrame(scene, t) {
    const beta  = state.beta;
    const L     = state.L;
    const gamma = 1 / Math.sqrt(Math.max(1 - beta * beta, 1e-12));
    const Llab  = L / gamma;
    const isLab = scene.userData.label === "lab";

    const { train, dad, jack, boltFront, boltBack,
            pulseFront, pulseBack, ground, staticWorld } = scene.userData;

    if (isLab) {
        // World is stationary; train moves +x at speed beta.
        staticWorld.position.x = 0;
        if (ground.userData.tex) ground.userData.tex.offset.x = 0;

        // Train is length-contracted along x by factor 1/gamma.
        train.position.set(beta * t, 0, 0);
        train.scale.set(1 / gamma, 1, 1);

        // Dad on platform (stationary).
        dad.position.set(-0.4, -1.39, -1.6);
        dad.rotation.y = -Math.PI / 16;

        // Jack rides the train -- stand him on the roof so he stays visible.
        // New roof y after the train was lowered onto the rails: -0.685 + 1.05/2 = -0.16
        jack.position.set(beta * t, -0.16, 0.0);
        jack.rotation.y = -Math.PI / 12;

        // Bolts strike at lab positions +/- L_lab/2 at t=0.
        const xF = +Llab / 2;
        const xB = -Llab / 2;
        positionBolt(boltFront, xF, 0, t, 0.0);
        positionBolt(boltBack,  xB, 0, t, 0.0);
        positionPulse(pulseFront, xF, t, 0.0);
        positionPulse(pulseBack,  xB, t, 0.0);

    } else {
        // Train sits still; the world (platform, foliage, ground texture)
        // slides past at -beta.
        const worldX = -beta * t;
        staticWorld.position.x = worldX;
        if (ground.userData.tex) {
            // Tile is 60 world-units wide (planeW=180 / repeat.x=3), so the
            // texture-space offset is worldX / 60 to keep the visual scroll
            // tied to the bushes/platform.
            ground.userData.tex.offset.x = -worldX / 60;
        }

        train.position.set(0, 0, 0);
        train.scale.set(1, 1, 1);

        // Jack on the train (stationary in this frame).
        jack.position.set(0.0, -0.16, 0);
        jack.rotation.y = 0;

        // Dad rides on the platform, which slides at -beta.
        dad.position.set(-0.4 + worldX, -1.39, -1.6);
        dad.rotation.y = -Math.PI / 16;

        // Strikes at fixed train-frame positions x' = +/- L/2.
        const xF = +L / 2, xB = -L / 2;
        const tF = -beta * L / 2;
        const tB = +beta * L / 2;
        positionBolt(boltFront, xF, 0, t, tF);
        positionBolt(boltBack,  xB, 0, t, tB);
        positionPulse(pulseFront, xF, t, tF);
        positionPulse(pulseBack,  xB, t, tB);
    }
}

function animateClocks(grp, now) {
    if (!grp) return;
    const ts = now * 0.001;
    for (const clk of grp.children) {
        const ud = clk.userData;
        clk.rotateOnAxis(ud.spinAxis, ud.spinSpeed * 0.016);
        clk.position.y = ud.bobBase + Math.sin(ts * 0.6 + ud.bobPhase) * ud.bobAmp;
    }
}

function positionBolt(grp, x, z, t, tFire) {
    grp.position.set(x, 0, z);
    const dt = t - tFire;
    if (dt < 0 || dt > FLASH_DUR) {
        grp.visible = false;
        grp.userData.flash.intensity = 0;
        return;
    }
    grp.visible = true;
    const a = 1 - dt / FLASH_DUR;
    const tube = grp.userData.tube;
    const halo = grp.userData.halo;
    tube.material.color.setRGB(1, 1, 1);
    halo.material.opacity = 0.45 * a;
    grp.userData.flash.intensity = 30.0 * a;
}

function positionPulse(mesh, x, t, tFire) {
    const dt = t - tFire;
    if (dt <= 0) {
        mesh.visible = false;
        return;
    }
    mesh.visible = true;
    mesh.position.x = x;
    // ring expands at speed c=1 in this frame's units.
    const r = dt;
    mesh.scale.set(r, r, 1);
    const fade = Math.max(0, 1 - r / PULSE_FADE);
    mesh.material.opacity = 0.65 * fade;
}

// ============ verdict text update ============
function updateVerdicts(t) {
    const beta = state.beta;
    const L    = state.L;
    const tF_train = -beta * L / 2;
    const tB_train = +beta * L / 2;

    // Lab frame: both strike at t = 0.
    if (t < 0)               verdictGEl.innerHTML = "&#9203; Calm... waiting for the storm.";
    else if (t < FLASH_DUR)  verdictGEl.innerHTML = "&#9889;&#9889; <b>BOTH bolts strike at once!</b>";
    else                      verdictGEl.innerHTML = "&#10003; Both pulses left at the same instant.";

    // Train frame
    if (t < tF_train)        verdictTEl.innerHTML = "&#9203; Calm... waiting for the storm.";
    else if (t < tF_train + FLASH_DUR)
                              verdictTEl.innerHTML = "&#9889; <b>FRONT bolt strikes first.</b>";
    else if (t < tB_train)   verdictTEl.innerHTML = "...front bolt is already racing back.";
    else if (t < tB_train + FLASH_DUR)
                              verdictTEl.innerHTML = "&#9889; <b>NOW the back bolt strikes.</b>";
    else                      verdictTEl.innerHTML = "&#10003; Two strikes, separated in time.";
}

// ============ derived display ============
function refreshDerived() {
    const b = state.beta;
    const g = 1 / Math.sqrt(Math.max(1 - b * b, 1e-12));
    const dtT = b * state.L;
    $("vGamma").textContent = g.toFixed(3);
    $("vDtT").textContent   = dtT.toFixed(2);

    if (b < 1e-3) {
        takeawayEl.textContent =
            "At rest, both observers see exactly the same thing.";
    } else {
        takeawayEl.innerHTML =
            "Dad says the bolts struck <b>simultaneously</b>. " +
            "Jack says the front bolt struck <b>" + dtT.toFixed(2) +
            " ticks</b> before the back. Both are right — simultaneity is relative.";
    }
}

// ============ UI bindings ============
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
bindRange("rBeta", "vBeta", "beta", (v) => v.toFixed(2), () => {
    refreshDerived();
});
bindRange("rL", "vL", "L", (v) => v.toFixed(1), (v) => {
    applyTrainLength(sceneLab.userData.train, v);
    applyTrainLength(sceneTrain.userData.train, v);
    refreshDerived();
});
bindRange("rTs", "vTs", "timeSpeed");
bindRange("rT",  "vT",  "t",  (v) => v.toFixed(2));

document.querySelectorAll(".presets button").forEach((btn) => {
    btn.addEventListener("click", () => {
        const v = parseFloat(btn.dataset.preset);
        if (!Number.isNaN(v)) {
            $("rBeta").value = v;
            $("rBeta").dispatchEvent(new Event("input"));
        }
    });
});

$("bPlay").textContent = "❚❚ Pause";
$("bPlay").addEventListener("click", () => {
    state.playing = !state.playing;
    $("bPlay").textContent = state.playing ? "❚❚ Pause" : "▶ Play";
});
$("bRestart").addEventListener("click", () => {
    state.t = T_RESET_LO;
    $("rT").value = state.t;
    $("rT").dispatchEvent(new Event("input"));
    state.playing = true;
    $("bPlay").textContent = "❚❚ Pause";
});

// ============ tooltips ============
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

refreshDerived();

// ============ resize ============
function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h, true);     // updateStyle=true: canvas fills viewport
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

    // LEFT: lab frame
    renderer.setViewport(0, 0, halfW, h);
    renderer.setScissor (0, 0, halfW, h);
    renderer.render(sceneLab, camera);

    // RIGHT: train frame
    renderer.setViewport(halfW, 0, w - halfW, h);
    renderer.setScissor (halfW, 0, w - halfW, h);
    renderer.render(sceneTrain, camera);

    renderer.setScissorTest(false);
}

function tick(now) {
    const dt_real = (now - prev) * 0.001;
    prev = now;

    if (state.playing) {
        state.t += dt_real * state.timeSpeed;
        if (state.t > T_RESET_HI) state.t = T_RESET_LO;
        $("rT").value = state.t.toFixed(2);
        $("vT").textContent = state.t.toFixed(2);
    }
    $("vTLab").textContent = state.t.toFixed(2);

    applyFrame(sceneLab,   state.t);
    applyFrame(sceneTrain, state.t);
    animateClocks(sceneLab.userData.clocks, now);
    animateClocks(sceneTrain.userData.clocks, now);
    updateVerdicts(state.t);

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
