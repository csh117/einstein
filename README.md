# Einstein

A 10-minute hybrid explainer of Special and General Relativity for curious 9-year-olds and the general public, combining Manim narration with interactive WebGPU/GLSL demos.

## Format

- **Hybrid**: Manim diagrams + voiceover/text + embedded interactive shader demos
- **Audience**: curious 9-year-old / general public
- **Length**: ~10 min
- **Tone**: no equations on screen (one E=mc² cameo at the end), heavy on metaphor, paper-airplane character as throughline
- **Visual ambition**: full ray-traced Schwarzschild, spectral Doppler, Terrell rotation. Each "wow" shot preceded by a stylized version that teaches the concept first.

## Repo layout

```
physics/                   Python: reference geodesic + tetrad math
                           (test_schwarzschild.py asserts the shader's
                           analytic limits; rays.json export not yet consumed)
demos/                     Three.js + GLSL/WGSL interactive shader scenes
  08_schwarzschild/        Black-hole geodesic ray-marcher (showpiece)
  ...                      Other scenes added as we go
manim/                     Narration scenes (planned — only requirements.txt so far)
assets/                    Skyboxes, blackbody spectra (planned — not in repo yet)
edit/                      Cut sheet, OBS capture list (planned — not in repo yet)
```

## Running a demo

The demos are static sites. From the repo root:

```powershell
python -m http.server 8000
```

Then open http://localhost:8000/demos/08_schwarzschild/ for the WebGL fallback build.

For the **WebGPU build** (raw `wgsl`, RTX 5090-tuned), open
http://localhost:8000/demos/08_schwarzschild/index.webgpu.html — the same scene
and UI ported to raw WGSL for lower driver overhead. Requires Chrome/Edge.

> **Capture target:** the **WebGPU build** (`main.webgpu.js`) is the canonical
> render path for the final film — it's the RTX 5090-tuned port and gets the
> physics/tuning work first. The WebGL build (`main.js`) is the secondary
> "works-everywhere" teaching build and may lag in tuning; don't assume the two
> stay pixel-identical.
>
> The WebGPU build integrates **real Kerr null geodesics** (Kerr-Schild Cartesian,
> validated against `physics/kerr.py`): the spin slider bends light with true
> frame dragging — the photon ring goes asymmetric and the shadow shifts as spin
> increases. The WebGL build's spin slider only adjusts the disk's inner edge.

## Scene plan (10:00 total)

| # | Scene | Length | Demo | Status |
|---|---|---|---|---|
| 0 | Cold open teaser | 0:20 | — | — |
| 1 | Light is stubborn (postulate) | 0:30 | — | — |
| 2 | Simultaneity (train + lightning) | 1:00 | dual-frame | ✅ shipped |
| 3 | Time dilation (light clock) | 1:00 | live light-clock | ✅ shipped |
| 4 | Length contraction + Terrell rotation | 1:00 | naive vs correct | ✅ shipped |
| 5 | Doppler + aberration flythrough | 0:45 | spectral starfield | partial (lives in scene 8) |
| 6 | Equivalence principle | 1:15 | falling elevator | — |
| 7 | Curved spacetime (geodesic deviation) | 1:00 | two marbles | ✅ shipped |
| 8 | Schwarzschild + Kerr black hole (showpiece) | 2:00 | geodesic ray-march | ✅ shipped |
| 9 | (merged into 8) | — | — | — |
| 10 | Gravitational waves | 0:30 | LIGO ring | — |
| 11 | Recap + outro | 0:45 | — | — |

## Three core insights the video lands on

1. Light has a stubborn speed → time and space stretch
2. Free fall feels like nothing → gravity isn't a force, it's the shape of space
3. Mass bends the shape; the shape steers the motion
