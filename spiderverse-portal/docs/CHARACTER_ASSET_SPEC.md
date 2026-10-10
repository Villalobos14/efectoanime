# Character asset specification (production Miles model)

The 3D character modes (MILES_MASK, MILES_FULL_SUIT, DEBUG_3D) render any glTF 2.0 character that follows this
document. The repository ships only a **procedural prototype** (`assets/characters/miles_prototype.glb`, built by
`tools/build_prototype_character.py`). It proves the pipeline but is not a film-quality Miles Morales: its head is the
MediaPipe canonical face smoothed into a mask, its body is a set of blended ellipsoids and cones, and its web and
emblem are procedural. Shader tuning cannot make up for missing sculpting, topology and texture work. That needs a
modelled asset.

To use a new asset:

```bash
python tools/check_character_asset.py path/to/miles.glb --preview preview.png   # validate + render a contact sheet
# then in configs/user.yaml:
character:
  asset: /absolute/or/project-relative/path/to/miles.glb
```

The validator prints errors (the asset will not work) and warnings (it will work with reduced function).

## 1. File format

| Item | Requirement |
|---|---|
| Container | glTF 2.0 binary (`.glb`), all buffers and textures embedded. `.gltf` + `.bin` also loads. |
| Units / axes | Metres, +Y up, character faces **+Z** (glTF defaults; Blender's exporter converts automatically). |
| Primitives | Triangles only (`mode` 4). Indexed. |
| Attributes | `POSITION`, `NORMAL` (smooth), `TEXCOORD_0`, `JOINTS_0`, `WEIGHTS_0` (4 influences). Tangents, colours, extra UV sets are ignored. |
| Morph targets | `POSITION` deltas; names in `mesh.extras.targetNames` (Blender writes these). Sparse accessors are fine. Normal deltas are ignored. |
| Skins | One skin preferred. At most **64 joints influence any one primitive** (split the mesh if a primitive uses more). |
| Animations | Not used (the pose comes from tracking every frame). |
| Closed meshes | Mask and suit meshes must be **watertight** (no open borders): the ink outline is an inverted hull. The validator warns otherwise. |

## 2. Coordinate convention and head alignment (the important part)

Head tracking delivers MediaPipe's *facial transformation matrix*: a rigid transform from MediaPipe's **canonical
face model** (468 points, centimetres, X right = subject's left, Y up, face looking toward +Z) into camera space.
The renderer projects with the same virtual camera MediaPipe uses (vertical FOV 63 degrees), which is what makes
the 3D head land on the real face.

So the asset must say where the canonical face sits on its head. Do it with **empties (null objects) parented to the
head bone**, placed on the mask surface at these canonical landmarks:

| Empty name | Canonical landmark | Where on the head |
|---|---|---|
| `align_eye_outer_L` | 263 | subject's left outer eye corner (under the lens) |
| `align_eye_outer_R` | 33 | subject's right outer eye corner |
| `align_nose_tip` | 4 | tip of the nose (under the fabric) |
| `align_chin` | 152 | bottom of the chin |
| `align_forehead` | 10 | top of the forehead at the hairline |

At least three are required; all five are better. The loader fits a similarity transform (scale, rotation,
translation) from the canonical points to these empties. If no empties exist, the head joint's own frame is taken
to *be* the canonical face frame divided by 100 (that is how the prototype is built).

Practical recipe in Blender: run `python tools/export_canonical_face.py` (writes `face_canonical.obj`, the
canonical face from MediaPipe's model bundle, Apache-2.0; vertex *i + 1* in the file is landmark *i*), import it,
scale it by 0.01, fit it onto the character's face, snap the five empties to the listed vertices, parent them to
the head bone, delete the guide.

Calibration that remains person-specific (scale, height of the mask on the face, pitch) is done live with the
`- =`, `9 0`, `; '` keys and saved with `S`.

## 3. Rig

Joint names are matched by role, case-insensitive. Any of the aliases works (Blender / Rigify / Mixamo / VRM).

| Role | Required | Example names | Driven by |
|---|---|---|---|
| `head` | **yes** | `head`, `mixamorig:Head`, `DEF-spine.006`, `J_Bip_C_Head` | face matrix (6 DoF) |
| `neck` | recommended | `neck`, `mixamorig:Neck` | half-way rotation between chest and head |
| `chest` | recommended | `chest`, `upper_chest`, `mixamorig:Spine2` | shoulder line from pose |
| `spine`, `root` | optional | `spine`, `mixamorig:Spine1`, `hips` | follow the chest |
| `shoulder.L/.R` | recommended | `shoulder.L`, `mixamorig:LeftShoulder`, `clavicle.L` | follow the chest |
| `upper_arm.L/.R` | recommended | `upper_arm.L`, `mixamorig:LeftArm` | pose shoulder -> elbow |
| `forearm.L/.R` | recommended | `forearm.L`, `mixamorig:LeftForeArm` | pose elbow -> wrist |
| `hand.L/.R` | optional | `hand.L`, `mixamorig:LeftHand` | follows the forearm |

`.L` is the **character's** left (+X). Joints not in the table (fingers, jaw, twist bones) follow their parents in
rest pose. Rest pose: arms down at 45-60 degrees (A-pose); bone roll/orientation is free (the rig uses joint
positions, not bone axes). Shoulder joints define the shoulder width the tracker scales to; place them at the
glenohumeral joint, not on the deltoid surface.

Skin weights: smooth falloff across neck (chest -> neck -> head) and shoulders (chest/shoulder/upper arm). The neck
is twisted half way by the rig, so a 3-bone neck blend avoids candy-wrapping at 45 degrees of head yaw.

## 4. Expressive lens morph targets

The mask hides the mouth, so expression lives in the lenses. Twelve morph targets, six per lens, on whichever mesh
holds the lenses (they may be separate lens meshes, as in the prototype, or part of the head mesh):

| Target (per side `_L` / `_R`) | Shape at weight 1 | Driven by |
|---|---|---|
| `lensBlink_*` | lens closed to a thin slit (top and bottom edges meet), red contour stays | eyeBlink |
| `lensSquint_*` | bottom edge raised ~55 %, top lowered slightly | eyeSquint |
| `lensWide_*` | taller, rounder (top up, bottom down) | eyeWide, brows up, jaw open (surprise) |
| `lensAngry_*` | inner-top edge pulled down toward the nose: hard slant | browDown |
| `lensSad_*` | inner-top raised, outer tip lowered | browInnerUp without browOuterUp, mouthFrown |
| `lensHappy_*` | bottom edge pushed up in the middle (smiling eyes) | mouthSmile, cheekSquint |

Weights combine linearly and can reach 1 together (for example angry 0.6 + squint 0.5 is a focused look), so test
pairs. Keep the lens on the mask surface in every target (sculpt on the surface, 1-2 mm lift); never let an edge
fold over itself. A blink morph is required for both sides; the others are recommended. `_L` is the subject's left
eye (+X); the side mapping from MediaPipe was verified on a photo with one eye painted shut.

## 5. Materials

The renderer is a stylised toon shader, not PBR: metallic/roughness/normal maps are ignored. Per material:

| Field | Meaning |
|---|---|
| `baseColorFactor`, `baseColorTexture` (sRGB PNG) | albedo. Black fabric should be near-black with a cool cast, about (0.024, 0.024, 0.034). |
| `extras.spiderverse.role` | `mask` (head fabric), `suit` (body fabric), `lens`, or `other`. Mask mode hides `suit` below the collar. Guessed from the material name if absent (`lens`/`eye` -> lens, `mask`/`head` -> mask). |
| `extras.spiderverse.shader` | `toon` (default), `lens` (prototype lens: needs `TEXCOORD_0.x` = 0 centre, 1 white edge, 2 outer red edge), `lens_flat` (white lens with soft shading; model the red rim as geometry or texture). |
| `extras.spiderverse.decal` | `true`: texture alpha-blends over the base colour (an emblem on black) instead of multiplying it. |
| `extras.spiderverse.web` | optional procedural web, evaluated per pixel from rest-pose positions: `center` (m), `axis`, `ref`, `radials`, `ring_step` (rad), `sag`, `width` (m), `max_angle` (rad, red region), `faint` (0..1, dark web beyond it). Use it, or paint the web into the texture. |

Texture resolution: head 2048x2048, body 2048x2048 (4096 if the web is painted), lenses 512. Web lines painted
into textures must be at least ~3 px wide at 2048 to stay clean at 720p.

## 6. Budgets (measured on the M1)

The 104k-triangle prototype renders, resolves 4x MSAA and reads back in 3.6 ms at 640x360 and 6.2 ms at 1280x720
(`tools/benchmark_character.py`). Keep a production asset at or below:

| Part | Triangles |
|---|---|
| Head / mask | <= 40,000 |
| Lenses (both) | <= 6,000 |
| Body (neck, torso, arms, hands) | <= 70,000 |

## 7. Blender export settings

File > Export > glTF 2.0:

- Format: **glTF Binary (.glb)**
- Include: Selected Objects (mesh + armature + align empties), Custom Properties **on** (writes `extras`)
- Transform: **+Y Up** on
- Data > Mesh: Apply Modifiers on, UVs on, Normals on, Tangents off, Vertex Colors off
- Data > Material: Export; Images: Automatic (PNG)
- Data > Shape Keys: **on** (Shape Key Normals optional), sparse shape keys allowed
- Data > Armature: Use Rest Position on, Export Deformation Bones Only on
- Data > Skinning: on, bone influences 4
- Animation: off

Material custom properties: add a property `spiderverse` of type *Python dictionary* (or set it via script:
`mat["spiderverse"] = {"role": "mask", "shader": "toon"}`) so it is exported as `extras.spiderverse`.

## 8. Where to get a production model, and licensing

A properly modelled, rigged Miles Morales is the real bottleneck for film-like fidelity. As of this writing:

- Fan-made rigged Miles models are sold or shared on Sketchfab, CGTrader, TurboSquid and Gumroad, usually as
  Blender/FBX with Rigify or Mixamo rigs. Some are marked *Editorial Use Only*. None can grant rights to the
  character itself: Miles Morales and the Spider-Verse designs are Marvel/Sony intellectual property. A
  Creative Commons tag on a fan model covers only the uploader's own work. Rips from games or films are not
  legally usable.
- For a private, non-commercial project, a purchased fan model (respecting its license) is a pragmatic route. For
  anything public or commercial, use an original character design or obtain a license from the rights holders.
- The cleanest route is a commissioned or self-made model in Blender that follows this spec: about 1-3 days of
  modelling, retopology, rigging and shape-key work for an experienced character artist.

Whichever route: convert to GLB with the settings above, add the align empties and the twelve lens shape keys, run
`tools/check_character_asset.py`, then point `character.asset` at it.
