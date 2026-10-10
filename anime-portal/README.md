# Anime Portal

A live webcam effect. The preview is mirrored. When both hands form a picture frame with the thumbs and index fingers, an anime face appears inside that frame. The frame follows the hands. The anime face follows the head.

This is a camera application. It does not read a video file, and it does not export one.

## Run

From this directory, with the virtual environment that was created for Python 3.12:

```bash
source .venv/bin/activate
python -m anime_portal
```

The preview window opens immediately. The first launch spends a moment loading the MediaPipe hand and face models. Focus that window before using the keyboard.

If `.venv` is missing, create it again:

```bash
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Useful flags:

```bash
python -m anime_portal --camera 1
python -m anime_portal --width 1280 --height 720
python -m anime_portal --no-mirror
```

Capture starts at 640×480. The camera may return a different size; the window title and the startup line show what was actually opened.

### Camera permission

If no frames arrive, macOS is blocking the camera. Open **System Settings → Privacy & Security → Camera**, enable the app you launched from (Cursor or Terminal), quit that app completely, and run the command again.

## What you do with your hands

Each hand makes a loose L with the thumb and index finger. The four fingertips are the corners of the portal:

```text
left index tip  ----------------  right index tip
      |                                   |
      |            anime face             |
      |                                   |
left thumb tip  ----------------  right thumb tip
```

The fingers do not need to be perfectly straight. Move, rotate, spread, or bring the hands closer. The quad follows, including uneven perspective when one hand is nearer to the camera. Lower either hand, or pinch the thumb against the index finger, and the portal fades out.

The effect turns on after a few valid frames and stays up through brief tracking gaps, so it does not flicker.

## Controls

Focus the preview window.

| Key | Action |
| --- | --- |
| `Q` or `Esc` | Quit |
| `D` | Debug overlay |
| `M` | Toggle mirror |
| `R` | Reset calibration to `configs/default.json` |
| `P` | Toggle the anime portal |
| `C` | Save a webcam reference frame in `captures/` |
| `S` | Save calibration to `configs/calibration.json` |
| Arrow keys, or `I` `J` `K` `L` | Nudge the anime face |
| `-` / `=` | Scale the anime face |
| `[` / `]` | Rotate the anime face |

`IJKL` is there because some OpenCV builds do not report arrow keys. `J` left, `L` right, `I` up, `K` down.

Debug mode draws hand landmarks, fingertip points, hand ids, portal corners, face landmarks, gesture state, tracking age, and FPS. Normal mode does not draw those graphics. FPS is still printed in the terminal about once a second, and it is shown in the window title. When you quit, the app prints the average FPS for the session.

## Artwork

The included file is `assets/anime_face.png`, a static placeholder head. Replace it with your own PNG. Transparent pixels stay transparent, so the photo remains visible around the hair.

The face is placed with a similarity transform (translation, rotation, scale) from these normalized anchors in `configs/default.json`:

- `left_eye`
- `right_eye`
- `nose`
- `chin`

Coordinates are fractions of the PNG width and height, origin at the top left. After replacing the image, set the anchors to the eyes, nose, and chin in that file, then use the nudge keys while watching the preview. Press `S` to keep the offset.

The placeholder anchors match the drawn placeholder. They will be wrong for a different painting until you edit them.

## How the image is built

Two coordinate systems stay separate:

1. **Face.** Eye, nose, and chin landmarks place and scale the anime layer over your head.
2. **Hands.** The four fingertips define the only region where that layer is visible.

The PNG is not stretched across the quad. It is aligned to the face, then clipped with a feathered polygon mask. A light color grade inside the quad makes the window visible even where the PNG is transparent. Your real fingers are painted back on top using a landmark mask: thick strokes along the finger bones and a filled palm. That mask is approximate.

Hand and face tracking use MediaPipe Tasks in `LIVE_STREAM` mode (`detect_async`), with a monotonically increasing timestamp on every frame. Models load once. The render loop does not wait for inference; it draws the newest result that has arrived. If a tracker is still busy, later camera frames are shown immediately and are not queued, so the overlay stays on the latest pose instead of lagging behind a backlog. The FPS in the window title is the preview rate. Debug mode also shows how old the landmarks are.

## Layout

```text
anime-portal/
├── assets/anime_face.png
├── models/                  # downloaded .task files
├── configs/default.json
├── src/anime_portal/
│   ├── app.py               # camera loop, keys, debug view
│   ├── camera.py
│   ├── hand_tracking.py
│   ├── face_tracking.py
│   ├── gesture_detection.py
│   ├── portal_geometry.py
│   ├── smoothing.py
│   ├── anime_renderer.py
│   ├── hand_occlusion.py
│   ├── calibration.py
│   └── config.py
└── tests/
```

`config.py` loads `configs/default.json`. It is the only extra module beyond the requested layout.

## Tests

```bash
source .venv/bin/activate
pytest
```

The tests cover quad validation, corner identity, the One Euro Filter, gesture hysteresis, face-anchor mapping, clipping, hand occlusion, calibration, and a smoke test that the installed MediaPipe Tasks models accept an image and a live-stream callback.

## Limits

- The anime face is a still image. It does not reproduce blinks, mouth shapes, or other live expressions. Expression-aware animation would be a later model.
- Head roll, position, and scale are aligned in 2D. Nodding and turning are only approximated from the nose and chin, so a strong pitch or yaw will not look like a 3D head.
- Inside the frame, outside the painted head, you still see the camera image with a light grade. This is not a generated anime background.
- Finger occlusion follows landmarks, not a silhouette. Thin fingers, overlapping hands, and rings can leave a fringe of the effect on the skin.
- A single missed hand does not close the portal immediately. During that short hold, the finger mask can lag behind the real hand.
- Very small frames, pinched fingers, and one-handed poses are ignored on purpose.
- Tracking quality drops in dim light, when a hand leaves the frame, or when the hands overlap the face completely.

## What was checked without a camera

This session could not open the webcam. macOS reported `not authorized to capture video`, so the hand gesture was not watched on a live picture. The program exits with camera-permission instructions in that case instead of crashing. After Camera access is enabled for Cursor or Terminal, the command above is what runs the effect.

These checks did run:

- 32 automated tests for geometry, smoothing, gesture hysteresis, face alignment, clipping, occlusion, calibration, and keyboard commands
- MediaPipe 1.1 hand and face landmarkers, including a live-stream callback, on a synthetic image
- The placeholder PNG loads with a transparent background and opaque eyes

## Versions used here

- Python 3.12.15 (Homebrew), Apple Silicon
- mediapipe 1.1.0 Tasks API (`HandLandmarker`, `FaceLandmarker`, `RunningMode.LIVE_STREAM`)
- opencv-python 5.0.0
- numpy 2.5.3

The system Python on this Mac is 3.9.6, which is older than the range this project targets. Use the 3.12 virtual environment above.
