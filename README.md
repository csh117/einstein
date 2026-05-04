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
physics/                   Python: trajectory/metric math, validates shaders
demos/                     Three.js + GLSL/WGSL interactive shader scenes
  08_schwarzschild/        Black-hole geodesic ray-marcher (showpiece)
  ...                      Other scenes added as we go
manim/                     Narration scenes, equations, diagrams
assets/                    Skyboxes, blackbody spectra
edit/                      Cut sheet, OBS capture list
```

## Running a demo

The demos are static sites. From the repo root:

```powershell
python -m http.server 8000
```

Then open http://localhost:8000/demos/08_schwarzschild/

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
