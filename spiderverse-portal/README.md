# Spider-Verse Portal

Real-time neural comic stylization, revealed through a portal you frame with your hands.

Open the app and you see yourself in the mirrored webcam preview. Raise both hands and make a frame with your
thumbs and index fingers. Inside that frame, the live camera image is redrawn as comic-book art: neural
stylization, then ink contours, cel shading, halftone dots and a Spider-Verse-inspired palette. Everything
outside the frame stays ordinary video. Move, tilt or resize the frame and the portal follows. Drop your hands
and it fades out.

Inside the portal you are transformed into a **Miles Morales-inspired Spider-Man**: a real-time 3D character
(rigged glTF model, toon-shaded with ink outlines, web and halftones) whose mask follows your head in 3D, whose white
lenses blink, squint, widen and frown with your face, and whose black-and-red suit follows your shoulders and arms.
The portal only decides where you see it: the character is always registered to your body, wherever the frame is.
The original stylization (and the older 2D procedural costume) stays available as mode 1 for comparison. See
[The 3D character renderer](#the-3d-character-renderer) and, for the production model it is ready to load,
[docs/CHARACTER_ASSET_SPEC.md](docs/CHARACTER_ASSET_SPEC.md).

The comic image is computed from the current camera frame, every frame. There is no static artwork anywhere in
the project: the suit, emblem, web and mask are drawn procedurally each frame, anchored to your live body and
face, and shaded with the light on your real clothes. When you smile, blink or turn your head, the stylized face
does the same, because it *is* your face, redrawn. The 3D character is not an image either: a rigged mesh is posed
from your tracked head and body and rendered every frame.

## Quick start

```bash
cd spiderverse-portal
/opt/homebrew/bin/python3.12 -m venv .venv          # native arm64 Python (3.11 or 3.12)
source .venv/bin/activate
pip install -e ".[dev,tools]"                       # tools = torch + onnx, only needed for model export
python tools/setup_models.py                        # downloads, exports, pre-compiles, builds the prototype character
python -m spiderverse_portal
```

`tools/setup_models.py` ends with a one-time Neural Engine compile of the default model at 640×480 and 720p.
On an M1 each compile took about one to three minutes. macOS caches the result, and later launches load the
model in about 0.1 s. The app never waits for this: if the compiled program is not cached (first run, a new
model or resolution, or after macOS purges its cache) it starts on the GPU (ONNX Runtime + Core ML, ~70 ms per
inference) while the Neural Engine compiles in the background, then switches over. The HUD's `Model:` line
shows which engine is active.

### Camera permission

If no frames arrive, macOS is blocking the camera. Open **System Settings → Privacy & Security → Camera**, enable
the app you launch from (Cursor, Terminal, VS Code), quit it completely, reopen it, and run again.

`device: auto` picks the built-in FaceTime camera even when an iPhone is connected through Continuity Camera
(OpenCV orders devices by unique ID, which often puts the iPhone at index 0). `python -m spiderverse_portal
--list-cameras` shows the order; `--camera 1` picks one explicitly.

## Making the portal

Each hand makes a loose **L**: index finger up, thumb pointing sideways toward the other hand. The two index tips
and the two thumb tips are the portal's four corners.

```text
  left index tip ─────────────── right index tip
        │                               │
        │        comic-book universe    │
        │                               │
  left thumb tip ─────────── right thumb tip
```

Fingers don't have to be straight or the frame rectangular: trapezoids, rotated frames and uneven perspective
all work. The portal opens after about 90 ms of a valid frame, rides through brief tracking dropouts (280 ms),
and closes when you lower a hand or pinch thumb to index. With the debug HUD on (**D**), a "why not" line says
which check is failing (for example `hand2:thumb`, `too small`, `twisted corners`).

## Keys

Focus the preview window first.

| Key | Action |
|---|---|
| Q / Esc | Quit |
| D | Debug HUD with all metrics |
| H | Hand landmarks, track ids, portal corners |
| F | Face landmarks |
| P | Enable / disable the portal gesture |
| N | Neural rendering on / off |
| C | Comic post-processing on / off |
| V | Full-screen stylization preview (check the neural output without the portal) |
| X | Comparison view: camera / neural output / final composite |
| M | Mirror on / off |
| R | Reset tracking, filters and temporal state |
| S | Save current settings to `configs/user.yaml` |
| T | Cycle neural model (paprika → celeba_distill → face_paint_512_v2) |
| O | Hand occlusion on / off |
| [ / ] | Lower / raise neural inference resolution |
| 1 | ORIGINAL_STYLIZATION: neural + comic stylization (with the 2D procedural costume selected by K) |
| 2 | MILES_MASK: the tracked 3D mask (with neck) on your head, your clothes stylized |
| 3 | **MILES_FULL_SUIT** (default): mask, neck, shoulders, torso, arms, emblem |
| 4 | DEBUG_3D: full frame, camera image under the semi-transparent character, wireframe, head axes, landmarks, pose skeleton, rig joints, alignment points |
| K | Cycle the 2D procedural costume of mode 1 (off / suit / mask) |
| - / = | Head calibration: scale down / up (2 % steps) |
| 9 / 0 | Head calibration: move the mask down / up (0.5 cm) |
| ; / ' | Head calibration: pitch the mask down / up (2 degrees) |

N and C combine into four modes, always labelled in the HUD: **hybrid** (neural + comic, default), **neural**
(raw network output), **procedural** (edge-preserving filter + comic, no network), **passthrough**.

## Useful flags

```bash
python -m spiderverse_portal --debug              # HUD + landmarks from the start
python -m spiderverse_portal --full-stylized      # start in the V preview
python -m spiderverse_portal --hd                 # 1280x720 capture
python -m spiderverse_portal --model celeba_distill --inference-width 384
python -m spiderverse_portal --sync matched       # show the exact frame the stylizer used (no seam lag, lower fps)
python -m spiderverse_portal --headless --duration 20 --snapshot-dir snapshots   # metrics run, no window
python -m spiderverse_portal --character miles_mask   # original | miles_mask | miles_full_suit | debug_3d
python -m spiderverse_portal --character original --costume suit   # mode 1 with the 2D procedural costume
python -m spiderverse_portal --camera osmo        # pick a camera by name (see --list-cameras)
python -m spiderverse_portal --debug-fixed-portal # moving portal without hands (debug aid)
```

## Configuration

`configs/default.yaml` holds every parameter. It is **hot-reloaded**: save it while the app runs and the comic,
portal, occlusion and temporal settings update live, which makes art direction fast. **S** writes the runtime
state to `configs/user.yaml`, layered on top of the defaults. Unknown keys and wrong types are rejected with a
message rather than ignored. `tools/write_default_config.py` regenerates the file from the dataclass defaults.

The main artistic controls under `comic:` are `ink_strength`, `ink_scale`, `ink_threshold`, `ink_silhouette`
and `ink_ghost` (linework); `color_levels`, `cel_strength` and `cel_softness` (shading bands); `halftone_*`,
`benday_strength` and `crosshatch_strength` (print texture); `saturation`, `contrast`, `hue_snap`,
`shadow_tint`, `highlight_tint` and `tint_strength` (palette); `print_offset` (plate misregistration);
`silhouette_px` (bold outer contour around the person); and `background_offset` / `background_strength` (extra
print offset behind the person, for depth).

Under `character:` (3D renderer): `mode`, `asset` (GLB path), `head_scale` / `head_offset_cm` /
`head_rotation_offset_deg` (calibration, also on keys), `auto_refine`, `body_scale`, smoothing (`head_min_cutoff`,
`head_beta`, `rotation_beta`, `body_min_cutoff`, `body_beta`), `expression_gain`, look (`outline_px`,
`halftone_cell_px`, `halftone_strength`, `rim_strength`, `key_light`) and integration (`erase_real_head`,
`erase_margin`, `hand_cutout`, `debug_opacity`).

Under `costume:` (2D procedural costume of mode 1): `mode` (off / suit / mask), `suit_strength`, `red_balance` (how much red web over the black),
`emblem_scale`, `web_density` and `web_rings`, `line_px` (web line weight), `highlight_strength` (how much of the
real clothing's light shows through the black), `suit_color` / `highlight_color` / `accent_color`,
`face_photo_mix` (how much of your real face sits under the comic pass in suit mode), `hair_cap` and
`mask_eye_scale`.

## How it works

```text
camera thread ──> FrameRing (newest 8 frames, ids + timestamps, mirrored once at capture)
                    │                                   │
                    ▼                                   ▼
tracking thread  MediaPipe hands (+ face every 2nd)   stylizer PROCESS (shared memory, 1 frame in flight)
                 identity association                  auto-exposure
                 L-pose + quad validation              ├ perception threads: person mask, face, pose
                 One Euro corners, state machine       ├ Layer A base: AnimeGANv2 on Core ML (Neural Engine)
                    │                                   │   + motion-aware temporal stabilization
                    │                                   ├ Layer B costume (mode 1): procedural suit / mask
                    │                                   ├ 3D character (modes 2-4): rig from face matrix +
                    │                                   │   blendshapes + pose, OpenGL draw submitted here
                    │                                   └ comic pass: grade, cel, halftone, plates, ink,
                    │                                     silhouette line, background offset
                    │                                     -> erase real hair/clothes around the character
                    │                                     -> upscale -> character composited (camera res)
                    ▼
main thread:  pairs each tracking result with its exact frame, takes the newest stylized frame,
              polygon alpha mask → blend → panel border → hand occlusion → HUD → imshow
```

**Pixel alignment.** The stylized frame is a same-size, pixel-aligned transform of the camera frame. The portal
quad only shapes the alpha mask (`final = original·(1−α) + stylized·α`); nothing is warped by a homography, so
the scene behind the portal stays exactly where it is (covered by a test that checks pixel equality under a
rotated portal).

**Why the stylizer is a separate process.** Core ML's Python binding holds the GIL during prediction and during
model loading. Measured here, a stylizer *thread* blocked every other Python thread for the full ~24 ms of each
inference, and for 110 s during one cold model load. In a child process the display and tracking keep their
own pace. Frames go through shared memory with exactly one frame in flight, so a slow stylizer skips frames and
never builds a backlog. (Running the comic stage in a parent thread, to overlap it with the next inference, was
tried and measured slower on this machine: it competed with tracking for the GIL.)

**Neural Engine loading.** Measured behaviour behind the loading design: a cold Neural Engine compile of this
model takes 100–170 s on an M1 and the system compiler serves one request at a time; once compiled, a load takes
~0.1 s; and macOS sometimes drops the compiled program between runs (several times in one evening here, with no
trigger I could reproduce; its cache is likely shared with system services that also use the Neural Engine). So
the app never waits: it starts on a GPU stand-in (the ONNX export via ONNX Runtime, ~70 ms per inference), runs
the Neural Engine compile in a throwaway subprocess, and swaps engines once a load is instant (a 0.26 s pause).

**Timing honesty.** The HUD reports camera, render, tracking and stylizer rates separately, plus model inference
time, the stylizer's total per-frame time, how old the displayed frame is, and how far the portal content lags the
outside (`portal content lag`). A stylized frame older than `max_stylized_age_ms` is never composited; the portal
says the stylizer is stale instead.

**Expression check.** With the HUD on, the tracking thread runs the face landmarker on the *stylized* frame twice
a second and prints its smile / jaw / blink scores next to the camera frame's. That is a live measurement that the
stylization keeps the expression.

### The neural model

**AnimeGANv2** generator ([bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch), MIT
license): a fully convolutional encoder-decoder (~2.2 M parameters) that maps a photo to an illustrated image of
the same geometry. Three pretrained weight sets are included:

| Variant | Character | Notes |
|---|---|---|
| `paprika` (default) | Saturated, painterly, keeps identity and expression | Best expression preservation at webcam scale |
| `celeba_distill` | Flat cel look, strong dark outlines | Most "cartoon"; faces slightly caricatured |
| `face_paint_512_v2` | Soft portrait painting | Trained on 512 px aligned faces; distorts small webcam-scale faces |

Why this model: permissive license with weights on GitHub; fully convolutional and small enough for real time on
an M1 Neural Engine; converts cleanly to Core ML and ONNX; preserves pose and expression because it re-renders the
input rather than generating a new image. Alternatives considered: White-box Cartoonization and AnimeGANv3 have
unclear or non-commercial licensing and no trustworthy ONNX source; fast neural style transfer (Johnson et al.)
produces painterly textures, not comic structure; diffusion img2img is far too slow on this hardware and not
temporally stable.

A generic anime model does not produce the Spider-Verse look by itself, so the network is the first stage of a
hybrid pipeline. The comic renderer then adds what defines that style: confident ink contours (multi-scale DoG on
the dark side of edges, cleaned of specks), 4 anti-aliased cel bands, a palette snap toward red, magenta, cyan,
electric blue and violet with warm skin kept warm, violet shadow and pink highlight tints, screen-fixed halftone
dots in shadows, Ben-Day dots in midtones, hatching in deep shadows, a misregistered colour plate under an aligned
black key plate, and an offset magenta ink ghost. Eyes, brows and mouth (from the face landmarker) are kept free
of dots and heavy fills so expressions read clearly.

### Measured performance (MacBook Air M1, 8 GB, macOS 26.5.1)

Model inference only (AnimeGANv2 generator, median of 20 runs, `tools/benchmark_neural.py`):

| Engine | 256×192 | 320×240 | 384×288 |
|---|---|---|---|
| ONNX Runtime CPU | 288 ms | 504 ms | |
| ONNX Runtime + Core ML EP, MLProgram, GPU | 40 ms | 64 ms | |
| ONNX Runtime + Core ML EP, Neural Engine only | 95 ms | 171 ms | |
| ONNX Runtime + Core ML EP, NeuralNetwork format | 438 ms (44 partitions) | 1139 ms | |
| PyTorch MPS | 55 ms | 86 ms | |
| **Core ML native fp16, Neural Engine** (used) | **13–15 ms** | **21–24 ms** | **34 ms** |
| Core ML native fp16, GPU (interim while compiling) | 45 ms | 73–77 ms | 112 ms |

With the costume, measured live on the OsmoPocket (1280×720 capture, rendered at 640×360, Neural Engine active):

| Costume mode | Display | Tracking | Stylizer | Stylizer per frame | Camera → stylized |
|---|---|---|---|---|---|
| off (comic only, with silhouette and background passes) | 28.8 fps | 28.9 fps | 29.3 fps | 29 ms | 31 ms |
| suit (MILES_SUIT_FACE_VISIBLE) | 29.9 fps | 29.9 fps | 23.3 fps | 42 ms | 59 ms |
| mask (MILES_FULL_MASK) | 30.0 fps | 30.2 fps | 22.4 fps | 44 ms | 64 ms |

Costume stages per frame: person mask ~2.7 ms, face ~8 ms (overlaps the neural inference), pose ~11 ms (own
thread, off the critical path), regions ~2.5 ms, suit ~7 ms, mask ~9 ms.

Other stages: comic renderer 7 ms at 640×480 in isolation (30 ms before its passes were fused); MediaPipe
hands 16–26 ms; face 2–9 ms (every second frame); compositing with occlusion about 1 ms. Frames wider than
640 px are tracked and stylized at 640 px and scaled back up, which keeps 720p as fast as 480p.

### Verification status

Measured on the live FaceTime camera with `--headless --duration 25 --debug-fixed-portal` (a moving portal
without hands), Neural Engine engine active, hybrid mode, in a dim room (auto-exposure gain ×3–3.6):

| Capture | Render | Tracking | Neural stylizer | Inference | Stylizer total | Capture → stylized | Displayed frame age |
|---|---|---|---|---|---|---|---|
| 640×480 | 30.1 fps | 30.1 fps | 24.1 fps | 23 ms | 41 ms | 59 ms | 24 ms |
| 1280×720 | 28.8 fps | 28.1 fps | 24.4 fps | 18 ms | 37 ms | 57 ms | 30 ms |

"Displayed frame age" is measured from the moment OpenCV returns the frame, so it excludes the camera's own
sensor and driver latency, which software cannot see. A 60-second run held these rates with stable memory
(app ~130–170 MB, stylizer ~170 MB after start-up).

Verified by me on this machine: webcam capture and built-in camera selection, the full threaded pipeline, Neural
Engine inference and the GPU fallback with hot-swap, compositing, the HUD, comparison view, 720p, headless
metrics, and every automated test. Expression preservation was verified on a still portrait: the face landmarker
still finds the face on the stylized output, with a smile score of 0.87–0.89 against 0.91 on the photo.

**Not yet verified, because it needs a person in front of the camera:** the framing gesture with real hands
(activation, following, rotation, resizing, release, recovery), hand occlusion on real fingers, and live smile,
blink, mouth and head-turn behaviour. The gesture logic is covered by tests on synthetic landmarks, but its
thresholds (`portal:` and the L-pose limits) may need tuning against real MediaPipe output. See the next section.

### Hands-on test plan

```bash
python -m spiderverse_portal --debug
```

1. The window shows your mirrored camera at about 30 fps (HUD line 1). Raise your right hand: it appears on
   the right side of the image.
2. Press **V**. The full frame is now the comic version of the live scene. Smile, blink, open your mouth and turn
   your head: the drawing does the same with no visible delay beyond ~60 ms. Press V again.
3. With both hands make the L-frame in front of your face. The HUD goes `CANDIDATE` then `ACTIVE` within about
   0.1 s, and the comic version appears inside the frame with a black and magenta/cyan panel border. Your
   fingertips stay in front of the border.
4. Move, tilt and widen the frame. The portal follows; the face behind it stays in place and is never warped.
5. Pinch thumb to index or drop a hand. The portal holds briefly (`TEMPORARILY_LOST`) then fades out.
6. Cross your hands or hide one for a moment. The HUD's hand ids (`#1`, `#2`) should not swap.
7. If the portal will not open, read the `why not:` line. To make it easier to trigger, lower `enter_score`,
   `minimum_area_ratio` or `min_hand_separation` in `configs/default.yaml` and save; the change applies live.
8. The `Expression check` line compares smile, jaw and blink scores between the camera and the stylized frame
   twice a second.

## The 2D procedural costume (mode 1, Layer B)

This was the first costume approach, kept for comparison (mode 1, cycle with K). It repaints the segmented body
in 2D; the 3D character above replaces it in modes 2-4.

Two layers produce the image inside the portal:

- **Layer A, comic reality** (neural + procedural): AnimeGANv2 redraws the live frame, then the comic pass adds
  ink, cel bands, halftone, plate offset, a bold silhouette line and background separation.
- **Layer B, costume** (procedural, driven by live perception): repaints your body as a Miles-inspired suit.

| Part | Neural or procedural | Driven by |
|---|---|---|
| Illustrated base image | Neural (AnimeGANv2 on the Neural Engine) | the current camera frame |
| Person / background mask | Neural (MediaPipe selfie segmenter, 2–3 ms) | the current frame |
| Face landmarks | Neural (MediaPipe face landmarker) | the current frame |
| Shoulders, torso yaw | Neural (MediaPipe pose landmarker lite, own thread) | the newest frame, smoothed |
| Head vs suit regions | Procedural geometry | face oval + jaw line (pose fallback when no face is found) |
| Black suit material | Procedural | the live luminance of your clothes: folds, light and motion stay visible |
| Red web and spider emblem | Procedural vector graphics | a body frame from your shoulders: they move, scale, rotate and foreshorten with you |
| Mask web and comic eyes | Procedural vector graphics | face landmarks every frame; eye height follows your lids, eye top follows your brows |
| Comic look | Procedural | every layer above |

**Mode 1, MILES_SUIT_FACE_VISIBLE** (default, key 2): the person mask minus the head becomes the suit. Your
face stays visible and is part real photo (edge-preserved) and part neural, so its likeness and every expression
read clearly under the comic pass.

**Mode 2, MILES_FULL_MASK** (experimental, key 3): the head region is covered by a black mask with red web
strands radiating from the nose bridge through the face outline, so the web deforms as the head turns. The comic
eyes squint and blink with your real eyelids and lift or slant with your brows, which keeps expression readable
through the mask. Limits: the mouth is hidden, and fast head motion can make the eyes lag by about a frame.

What it is not: a generative model that invents a costume. Layer B is region-aware procedural rendering on top of
live perception. That is what keeps it real-time on this machine and keeps every pose and movement yours. The
segmenter that labels hair, face, skin and clothes separately was measured at 128 ms per frame on the M1, so the
head and suit are split geometrically instead.

## The 3D character renderer

Modes 2-4 replace the 2D costume with a rigged 3D character that is posed from tracking and rendered every frame.

```text
stylizer process, per frame (640 px wide analysis frame)
  MediaPipe Face Landmarker ── 478 landmarks, 52 blendshapes, facial transformation matrix
  MediaPipe Pose Landmarker ── shoulders, elbows, wrists, hips (2D + relative depth), own thread
  selfie segmenter ─────────── person mask
        │
        ▼
  rig (character/rig.py)
    head    = face matrix x calibration, One-Euro smoothed (rotation, x/y, depth separately),
              + 2D refinement fitted to rigid landmarks
    chest   = placed on the detected shoulders at the head's depth, scaled to the shoulder width
              (clamped to keep the character's proportions), turned by the pose depth difference
    arms    = bones keep their length, rotate toward the detected elbow / wrist; hang when not visible
    neck    = half-way between chest and head
    lenses  = blendshapes -> 12 lens morph weights (character/expressions.py)
        │
        ▼
  OpenGL 4.1 (ModernGL, Metal on Apple Silicon), offscreen 4x MSAA, camera resolution
    GPU skinning, morphs, toon bands + cool sheen, hard speculars, violet rim, per-pixel web from
    rest-pose positions, anchored halftone, inverted-hull ink outline, lens shader with red contour
        │  (the GPU renders while the CPU runs the comic pass)
        ▼
  integration (character/layer.py)
    learned background plate replaces real hair / ears / loose clothing around the character,
    real hands are cut out of the character (they stay in front), premultiplied "over"
        │
        ▼
  stylized frame (same pixel grid as the camera) ──> portal mask in the main process
```

**Registration.** The Face Landmarker's transformation matrix maps MediaPipe's canonical face (centimetres) into a
metric camera space that, projected with a 63 degree vertical-FOV camera, lands on the detected landmarks (median
error 5 px on the test portrait, versus 12 px at 60 and 10 px at 66 degrees). The character's head is built in, or
aligned to, canonical-face coordinates, so it lands on your face independent of the webcam's real intrinsics. A
small 2D scale + shift fitted to inner rigid landmarks (forehead, brow ridge, nose, eye corners, cheekbones; not the
face contour, which biased the fit by 4 % in a test) removes most of the remaining offset.

**The portal is only a mask.** The character is rendered in camera coordinates into the stylized frame. The portal
reveals the part of it behind the frame: frame your head and you see the mask, move the frame down and you see the
emblem; moving the frame never moves the character, and nothing leaks outside (both covered by tests).

**Expressions.** `eyeBlink` closes a lens to a red slit, `eyeSquint` narrows it, `browDown` slants it (angry,
focused), brows up or `jawOpen` widen it (surprise), inner brows alone droop it (sad), a smile pushes its lower edge
up (happy eyes). Each person's resting blendshape levels are learned (a slowly adapting neutral), blinks are smoothed
with a 20 ms attack, other expressions with 70 ms. Verified mapping: MediaPipe's `...Left` channels drive the lens
on the subject's left (+X), checked on a photo with one eye painted shut.

**What is prototype and what is production.** The rendering infrastructure (GLB loading with skins, sparse morphs
and embedded textures; Blender/Rigify/Mixamo/VRM joint names; GPU skinning; toon materials; the rig; expression
mapping; background erase; portal integration) is production code. The **character model is a procedural
prototype** (`assets/characters/miles_prototype.glb`, 104k triangles): the head is MediaPipe's canonical face
smoothed into a fabric mask and blended into a skull, the lenses are spline outlines projected onto it with six
morph targets each, the body is a signed-distance sculpture of ellipsoids and cones polygonised with surface nets,
and the emblem texture is drawn procedurally. It reads as a Spider-Man in the Spider-Verse style, but it is not a
film-quality Miles Morales: anatomy, mask sculpting, fabric detail and the exact ATSV lens and emblem designs need a
modelled asset. [docs/CHARACTER_ASSET_SPEC.md](docs/CHARACTER_ASSET_SPEC.md) specifies that asset (format, rig,
morphs, materials, budgets, Blender export settings, licensing), and `tools/check_character_asset.py` validates one
and renders a preview sheet.

Where the quality limits are, honestly:

| Dimension | State |
|---|---|
| Tracking | MediaPipe face matrix (6 DoF) + pose; 2D refinement; One-Euro smoothing. Single camera: body depth is estimated, not measured. |
| Animation | Head rigid from tracking; neck blends; torso from the shoulder line; arms aim at elbows / wrists (no twist, no hands). Lenses: 12 morphs. |
| **Model** | **The bottleneck.** Procedural prototype: correct face fit, but generic body anatomy and approximate Miles details. |
| Materials / shading | Toon shader with web, halftone, ink, rim; shared by head and body, so they read as one character. No cloth simulation, no crease lines yet. |
| Composite | Real hair / ears / clothing near the character replaced by a learned background (blurry where the background was never seen); hands cut out from landmarks (approximate shape, no true depth). |

### Measured (MacBook Air M1, 8 GB)

From `python tools/benchmark_character.py`: the test portrait in a 1280x720 frame, moved every frame, through the
same pipeline the stylizer process runs (Core ML engine), 60 frames per mode after warm-up:

| Median ms (1280x720, under the load described below) | Mode 1 + 2D costume | MILES_MASK | MILES_FULL_SUIT |
|---|---|---|---|
| **Total per stylized frame** | **114.4** | **106.1** | **114.9** |
| Neural inference (Neural Engine, overlaps perception) | 30.0 | 30.5 | 30.6 |
| Face landmarks + blendshapes (worker thread) | 39.4 | 39.4 | 39.9 |
| Pose (own thread, off the critical path) | 38.6 | 38.6 | 40.6 |
| Person mask | 9.2 | 8.9 | 9.0 |
| 2D costume (regions + paint) | 26.9 | - | - |
| Rig (tracking -> joints, expressions) | - | 3.3 | 3.3 |
| Draw submission | - | 1.7 | 1.7 |
| GPU wait + readback (character's box only) | - | 4.3 | 6.8 |
| Background erase + plate update | - | 8.9 | 9.9 |
| Composite at 1280x720 | - | 2.2 | 4.5 |

Renderer alone, 104k-triangle prototype, draw + resolve + full-frame readback: 4.1 ms at 640x360 and 6.3 ms at
1280x720 with 4x MSAA (3.6 / 5.7 ms without MSAA).

These were measured while another instance of this app (launched before this work, holding the camera) used about
2.5 CPU cores, and while macOS was recompiling the Neural Engine program in the background. That makes every stage
2-3x slower than the earlier measurements in this README (for example face 38 ms here versus about 8 ms before). The
meaningful comparison is between modes under the same load: the 3D modes cost about the same as mode 1 with the 2D
costume, because they replace it. Rerun `tools/benchmark_character.py` and a live `--headless --duration 30` run on
an idle machine for absolute numbers.

### Verified without a person in front of the camera

- Tests on the test portrait: the face matrix + virtual camera registers with the landmarks; the mask covers more
  than 97 % of the face oval; the face landmarker no longer finds a human face in the masked output; the real hair
  band is replaced; a hand in front of the face stays real; the portal reveals the registered character without
  moving it, and nothing leaks outside the portal; the full pipeline renders in modes 2-4 and inside the spawned
  stylizer process.
- Synthetic head turns (the portrait's face re-rendered as a textured 3D landmark mesh turned by known angles, then
  re-tracked): the rig follows the tracker within 2 degrees for yaw -35..+35 and pitch -15..+15, and keeps the mask on
  the face (coverage 0.97-1.00). The synthetic images turn less than nominal (MediaPipe's landmark depth is
  compressed), so they test following, not absolute angle.
- Rig maths (rotation, calibration, smoothing, shoulder placement, arm aiming), expression mapping (each blendshape,
  side mapping, blink speed, neutral learning), lens morphs (blink closes the white area, no outline folds), glTF
  round trip with skins / morphs / sparse accessors / textures, ROI readback.
- A live 20 s headless run on the OsmoPocket camera in full-suit mode (no person in view): no errors.

**Not yet verified, because it needs you in front of the camera:** live head turns, scale with distance, blink /
squint / brow / smile on the lenses with real blendshapes, shoulder and arm following, the hair erase on your hair,
hand cut-outs on real fingers, and the overall look in your room's light. Test plan:

```bash
python -m spiderverse_portal --character debug_3d --debug
```

1. DEBUG_3D: green dots (detected rigid landmarks) and red circles (projected 3D head) should coincide; the cyan
   wireframe should sit on your head and shoulders; the magenta joints on your shoulders/elbows. If the mask sits
   high or low, use `9` / `0`; too big or small, `-` / `=`; tilted, `;` / `'`; then `S` to save.
2. Press `2` (MILES_MASK) and `V` (full-frame preview): turn your head 30 degrees each way, nod, tilt, lean in and
   out. The mask should turn with you, scale with distance and never detach. No hair or ears should show.
3. Blink, wink, squint, raise your brows, frown, smile: the lenses should close, narrow, widen, slant and lift.
   If they react too little, raise `character.expression_gain` in `configs/default.yaml` (hot-reloaded).
4. Press `3` (MILES_FULL_SUIT): shrug, turn your shoulders, raise an arm. The suit and emblem should follow.
5. Press `V` again and frame yourself with both hands: the portal should reveal the mask over your head and the
   emblem lower down; moving the frame must not move the character; your fingers should stay in front.

## Tests

```bash
pytest                                   # 134 tests, 3-6 minutes on the M1 (1 needs the camera, skipped)
SPIDERVERSE_CAMERA_TESTS=1 pytest -k live   # also opens the physical webcam
```

Covered: corner ordering, convexity and self-intersection, quad validation, the gesture state machine
(activation timing, hysteresis, release, dropout recovery, glitch rejection, twisted frames, translation/rotation/
scale following), hand identity through crossings and label swaps, One Euro behaviour, masks and compositing
(including pixel alignment under a rotated portal), the portal fade, hand occlusion, configuration loading/
validation/saving, frame-ring no-backlog behaviour, the comic renderer's determinism and screen-fixed textures,
auto-exposure, temporal stabilization, the neural model (output geometry, spatial correspondence with the input,
determinism, Core ML vs. ONNX agreement, missing-model errors, cleanup), the expression surviving stylization
(face landmarker smile score on the hybrid output), a scan proving the package never loads image files, and the
stylizer process protocol (skips instead of queueing, bounded result age, shared memory released), 720p
frames rendered at the capped width and returned at full size, the tracking thread end to end on a real
photo (face found, render context built, expression check run against the stylized frame), and the costume layer:
person/face/shoulder detection, the head/suit split, body-frame maths, the suit changing only its region, the
emblem following the torso, mask eyes closing with the eyelid, all three modes, and the expression staying
readable in suit mode. The 3D character (`tests/test_character.py`, plus a stylizer-process test): glTF round trip
(skins, morphs, sparse accessors, textures, extras), canonical-face parsing, camera projection / unprojection /
matrix agreement, face-matrix registration on a real photo, the prototype asset against the spec (roles, morphs,
closed meshes, alignment), lenses on the mask surface and never folding, rig head rotation / calibration /
smoothing / hold-and-hide, torso on the shoulders and arms toward the elbows, every expression mapping and the
blink side, renderer registration and lenses turning with the head, blink morph, mask-mode cut, ROI readback,
push-pull fill, hand masks, face coverage / hair erase / hand cut-out on the portrait, the masked output hiding the
face from a face detector, portal independence (no movement, no leak), synthetic head turns (tracking + coverage),
config validation, the full pipeline in modes 2-4, and the character rendered inside the spawned stylizer process.

## Project layout

```text
src/spiderverse_portal/
  app.py                  main loop, keys, HUD, hot reload, adaptive quality, CLI
  camera.py               capture thread, camera selection (auto / index / name)
  camera_avf.py           native AVFoundation capture (picks the camera's format, hardware scaling)
  config.py               typed config, YAML load/validate/save
  tracking/               hands.py, face.py, gesture.py, smoothing.py, worker.py
  geometry/               quadrilateral.py, masks.py
  stylization/            base.py (interface), neural.py, procedural.py, hybrid.py (pipeline),
                          service.py (child process), temporal.py, exposure.py, onnx_session.py, precompile.py
  rendering/              comic.py, colors.py, shading.py, halftone.py, ink.py,
                          compositor.py, occlusion.py, overlay.py
  semantic/               analyzer.py (person mask, face, pose in the stylizer process)
  costume/                layer.py, regions.py, body.py, suit.py, mask.py (2D procedural costume, mode 1)
  character/              3D character: gltf.py (GLB I/O), asset.py (loader, joint roles), face_geometry.py
                          (canonical face, virtual camera), rig.py (tracking -> joints), expressions.py
                          (blendshapes -> lens morphs), renderer.py (ModernGL toon renderer), layer.py
                          (background erase, hand cut-out, composite), prototype.py + meshtools.py
                          (procedural prototype builder)
  performance/            profiler.py, frame_scheduler.py
tools/                    setup_models.py, export_animegan_onnx.py, export_coreml.py,
                          benchmark_neural.py, write_default_config.py, build_prototype_character.py,
                          check_character_asset.py, export_canonical_face.py, benchmark_character.py
assets/characters/        miles_prototype.glb (generated; any spec-conforming GLB can replace it)
docs/                     CHARACTER_ASSET_SPEC.md
tests/                    (synthetic_faces.py: textured landmark-mesh head turns for tests)
```

## Licenses of model files

- AnimeGANv2 weights and generator architecture: MIT, © 2021 Bryan Lee (`models/animegan2/LICENSE`).
- MediaPipe Hand, Face and Pose Landmarker and Selfie Segmenter models: Apache-2.0, Google. The prototype head
  is built on the canonical face mesh contained in `face_landmarker.task` (same license).
- `assets/characters/miles_prototype.glb`: generated by this project's code. Miles Morales and the Spider-Verse
  designs belong to Marvel/Sony; this is a personal fan project, see the licensing notes in
  `docs/CHARACTER_ASSET_SPEC.md`.
- Test image `tests/data/astronaut.png`: NASA portrait of Eileen Collins, public domain (via scikit-image).
