"""Real-time stylised character renderer (ModernGL, OpenGL 4.1 core; on Apple Silicon it runs on Metal).

Offscreen: no window. Each frame renders the skinned, morphed character into a
4x MSAA framebuffer with a transparent background and reads back premultiplied
RGBA at the camera frame's resolution, in the same pixel grid (the projection
is MediaPipe's virtual camera, so it registers with the face landmarks).

Passes:
  1. toon   : two-band cel shading with a cool sheen band for black fabric,
              hard-edged speculars, a violet rim, procedural web lines evaluated
              per pixel from rest-pose positions (so the web is attached to the
              surface and stays crisp at any distance), decal textures (emblem),
              and halftone dots in the terminator band. The dot grid is anchored
              to the character's projected head position and scaled with its
              apparent size, so dots move with the character instead of
              swimming over it, and do not shimmer.
  2. ink    : inverted-hull silhouette outline with a constant width in pixels.
  3. lenses : white lenses with a red contour band and pixel-constant ink lines.
  4. debug  : optional wireframe and joint axes.

Why a readback: the rest of the app composites on the CPU (OpenCV display).
Measured on an M1, render + MSAA resolve + readback at 1280x720 takes ~3.5 ms,
cheaper than moving the whole display path to the GPU.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .asset import CharacterAsset, MeshPart

MAX_JOINTS = 64

VERT = """
#version 410
uniform mat4 u_proj;
uniform mat4 u_joints[64];
uniform float u_outline_px;
uniform vec2 u_viewport;
in vec3 in_pos;
in vec3 in_nrm;
in vec2 in_uv;
in vec4 in_joints;
in vec4 in_weights;
out vec3 v_nrm;
out vec3 v_cam;
out vec3 v_rest;
out vec2 v_uv;
void main() {
    mat4 skin = in_weights.x * u_joints[int(in_joints.x + 0.5)]
              + in_weights.y * u_joints[int(in_joints.y + 0.5)]
              + in_weights.z * u_joints[int(in_joints.z + 0.5)]
              + in_weights.w * u_joints[int(in_joints.w + 0.5)];
    vec4 p = skin * vec4(in_pos, 1.0);
    vec3 n = normalize(mat3(skin) * in_nrm);
    v_cam = p.xyz;
    v_nrm = n;
    v_rest = in_pos;
    v_uv = in_uv;
    vec4 clip = u_proj * p;
    if (u_outline_px > 0.0) {
        vec4 cn = u_proj * vec4(p.xyz + n * 0.5, 1.0);
        vec2 d = cn.xy / cn.w - clip.xy / clip.w;
        float len = length(d);
        if (len > 1e-6) {
            clip.xy += (d / len) * u_outline_px * 2.0 / u_viewport * clip.w;
        }
    }
    gl_Position = clip;
}
"""

COMMON_FRAG = """
#version 410
in vec3 v_nrm;
in vec3 v_cam;
in vec3 v_rest;
in vec2 v_uv;
out vec4 o_color;
uniform vec3 u_key;
uniform vec3 u_rim_dir;
uniform vec4 u_cut;        // x: enabled, y: min rest y, z: max radius from cut axis, w: feather
uniform vec2 u_cut_xz;
const float PI = 3.14159265;

float cut_alpha() {
    if (u_cut.x < 0.5) return 1.0;
    float r = length(v_rest.xz - u_cut_xz);
    float a = smoothstep(u_cut.y - u_cut.w, u_cut.y + u_cut.w, v_rest.y);
    a *= 1.0 - smoothstep(u_cut.z - u_cut.w, u_cut.z + u_cut.w, r);
    return a;
}
"""

TOON_FRAG = COMMON_FRAG + """
uniform vec3 u_base;
uniform sampler2D u_tex;
uniform int u_has_tex;
uniform int u_decal;
uniform int u_web_on;
uniform vec3 u_web_center;
uniform vec3 u_web_axis;
uniform vec3 u_web_ref;
uniform float u_web_radials;
uniform float u_web_step;
uniform float u_web_sag;
uniform float u_web_width;
uniform float u_web_max;
uniform float u_web_faint;
uniform vec3 u_web_faint_color;
uniform vec3 u_red;
uniform vec3 u_shadow_tint;
uniform vec3 u_shadow_add;
uniform vec3 u_sheen;
uniform vec3 u_highlight;
uniform vec3 u_rim_color;
uniform vec3 u_spec_color;
uniform float u_terminator;
uniform vec2 u_ht_anchor;
uniform float u_ht_cell;
uniform float u_ht_strength;
uniform float u_rim_strength;
uniform float u_spec_strength;

vec2 web_mask() {  // (red lines, faint lines beyond the red region)
    vec3 d = v_rest - u_web_center;
    float r = length(d);
    vec3 dn = d / max(r, 1e-6);
    float alpha = acos(clamp(dot(dn, u_web_axis), -1.0, 1.0));
    if (alpha > u_web_max + 0.05 && u_web_faint <= 0.0) return vec2(0.0);
    vec3 x = normalize(u_web_ref - dot(u_web_ref, u_web_axis) * u_web_axis);
    vec3 y = cross(u_web_axis, x);
    float beta = atan(dot(dn, y), dot(dn, x));
    float sector = beta * u_web_radials / (2.0 * PI);
    float fs = fract(sector);
    float radial = min(fs, 1.0 - fs) * (2.0 * PI / u_web_radials) * sin(alpha) * r;
    // Rings sag toward the web centre between two radial strands.
    float ae = alpha + u_web_sag * u_web_step * sin(PI * fs);
    float k = max(1.0, floor(ae / u_web_step + 0.5));
    float ring = abs(ae - k * u_web_step) * r;
    float dline = min(radial, ring);
    float aa = max(fwidth(dline), 1e-6);
    float m = 1.0 - smoothstep(u_web_width - aa, u_web_width + aa, dline);
    float region = 1.0 - smoothstep(u_web_max - 0.03, u_web_max, alpha);
    return vec2(m * region, m * (1.0 - region) * u_web_faint);
}

float halftone(float coverage) {
    vec2 g = (gl_FragCoord.xy - u_ht_anchor) / u_ht_cell;
    g = mat2(0.7071, -0.7071, 0.7071, 0.7071) * g;
    vec2 c = fract(g) - 0.5;
    float d = length(c);
    float rad = sqrt(clamp(coverage, 0.0, 1.0) / PI);
    float aa = fwidth(d) * 1.2;
    return 1.0 - smoothstep(rad - aa, rad + aa, d);
}

void main() {
    float ca = cut_alpha();
    if (ca <= 0.002) discard;
    vec3 N = normalize(v_nrm);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(-v_cam);
    vec3 albedo = u_base;
    if (u_has_tex == 1) {
        vec4 t = texture(u_tex, v_uv);
        albedo = (u_decal == 1) ? mix(u_base, t.rgb, t.a) : u_base * t.rgb;
    }
    vec2 web = (u_web_on == 1) ? web_mask() : vec2(0.0);
    albedo = mix(albedo, u_web_faint_color, web.y);
    albedo = mix(albedo, u_red, web.x);
    float lum = dot(albedo, vec3(0.299, 0.587, 0.114));

    float ndl = dot(N, u_key);
    float w = max(fwidth(ndl), 1e-4);
    float lit = smoothstep(u_terminator - w, u_terminator + w, ndl);
    float hi = smoothstep(0.86 - w, 0.86 + w, ndl);
    vec3 dark = albedo * u_shadow_tint + u_shadow_add;
    vec3 base_lit = albedo + u_sheen * (1.0 - lum);
    vec3 top = base_lit + u_highlight * (1.0 - 0.6 * lum);
    // Halftone dots of the shadow colour across the terminator band.
    float band = 1.0 - smoothstep(u_terminator, u_terminator + 0.45, ndl);
    float dots = halftone(band * band * 0.85) * u_ht_strength * lit;
    vec3 col = mix(dark, base_lit, lit);
    col = mix(col, top, hi);
    col = mix(col, dark, dots);
    // Hard-edged stylised specular.
    vec3 H = normalize(u_key + V);
    float s = dot(N, H);
    float sw = max(fwidth(s), 1e-4);
    col += u_spec_color * u_spec_strength * smoothstep(0.997 - sw, 0.997 + sw, s) * (0.5 + 0.5 * lit);
    // Coloured rim on the far side of the key light.
    float fres = 1.0 - max(dot(N, V), 0.0);
    float rw = max(fwidth(fres), 1e-4);
    float rim = smoothstep(0.72 - rw, 0.72 + rw, fres) * smoothstep(-0.1, 0.35, dot(N, u_rim_dir));
    col = mix(col, u_rim_color * (0.6 + 0.4 * (1.0 - lum)) + albedo * 0.5, rim * u_rim_strength);
    o_color = vec4(clamp(col, 0.0, 1.0) * ca, ca);
}
"""

INK_FRAG = COMMON_FRAG + """
uniform vec3 u_ink;
void main() {
    float ca = cut_alpha();
    if (ca <= 0.002) discard;
    o_color = vec4(u_ink * ca, ca);
}
"""

LENS_FRAG = COMMON_FRAG + """
uniform vec3 u_white;
uniform vec3 u_lens_shade;
uniform vec3 u_rim_red;
uniform vec3 u_ink;
uniform float u_ink_px;
uniform int u_flat;
void main() {
    vec3 N = normalize(v_nrm);
    float ndl = dot(N, u_key);
    float r = v_uv.x;
    float y = v_uv.y;
    float shade = smoothstep(0.1, -1.1, y) * 0.55 + (1.0 - smoothstep(-0.3, 0.4, ndl)) * 0.25;
    vec3 white = mix(u_white, u_lens_shade, clamp(shade, 0.0, 1.0));
    if (u_flat == 1) { o_color = vec4(white, 1.0); return; }
    float aw = clamp(fwidth(r), 1e-5, 0.06);  // clamped: a squashed triangle must not smear ink
    float lit = smoothstep(-0.05, 0.05, ndl);
    vec3 red = mix(u_rim_red * vec3(0.55, 0.35, 0.75), u_rim_red, 0.35 + 0.65 * lit);
    vec3 col = (r < 1.0) ? white : red;
    // Pixel-constant ink: a line on the white/red boundary and one on the outer edge.
    // Inner ink sits on the white side of the boundary so the red band stays fully visible.
    float inner = 1.0 - smoothstep(u_ink_px * aw, u_ink_px * aw + aw, abs(r - 1.0 + 0.5 * u_ink_px * aw));
    float outer = smoothstep(2.0 - 0.8 * u_ink_px * aw - aw, 2.0 - 0.8 * u_ink_px * aw, r);
    col = mix(col, u_ink, max(inner, outer));
    o_color = vec4(col, 1.0);
}
"""

FLAT_FRAG = COMMON_FRAG + """
uniform vec4 u_color;
void main() { o_color = vec4(u_color.rgb * u_color.a, u_color.a); }
"""

LINE_VERT = """
#version 410
uniform mat4 u_proj;
in vec3 in_pos;
in vec3 in_color;
out vec3 v_color;
void main() { v_color = in_color; gl_Position = u_proj * vec4(in_pos, 1.0); }
"""
LINE_FRAG = """
#version 410
in vec3 v_color;
out vec4 o_color;
void main() { o_color = vec4(v_color, 1.0); }
"""


@dataclass
class ToonStyle:
    key_dir: tuple[float, float, float] = (-0.45, 0.55, 0.70)
    rim_dir: tuple[float, float, float] = (0.75, 0.25, -0.45)
    red: tuple[float, float, float] = (0.86, 0.05, 0.10)
    shadow_tint: tuple[float, float, float] = (0.45, 0.30, 0.70)
    shadow_add: tuple[float, float, float] = (0.006, 0.005, 0.014)
    sheen: tuple[float, float, float] = (0.034, 0.040, 0.085)
    highlight: tuple[float, float, float] = (0.040, 0.050, 0.115)
    web_faint: tuple[float, float, float] = (0.11, 0.10, 0.16)
    rim_color: tuple[float, float, float] = (0.42, 0.30, 1.00)
    spec_color: tuple[float, float, float] = (0.45, 0.52, 0.85)
    ink: tuple[float, float, float] = (0.015, 0.008, 0.025)
    lens_white: tuple[float, float, float] = (0.985, 0.985, 0.975)
    lens_shade: tuple[float, float, float] = (0.70, 0.76, 0.90)
    terminator: float = 0.05
    halftone_strength: float = 0.75
    rim_strength: float = 0.65
    spec_strength: float = 0.35
    lens_ink_px: float = 1.3


@dataclass
class RenderFrame:
    size: tuple[int, int]
    proj: np.ndarray  # 4x4 camera(cm) -> clip, row-major
    joint_world: np.ndarray  # (J, 4, 4) joint world matrices: model -> camera (cm)
    morph_weights: dict[str, float] = field(default_factory=dict)
    roles: tuple[str, ...] = ("mask", "lens", "suit", "other")
    outline_px: float = 2.0
    halftone_anchor: tuple[float, float] = (0.0, 0.0)  # pixels, image coordinates
    halftone_cell: float = 7.0
    cut: tuple[float, float, float, float] | None = None  # (min rest y, radius, feather) for role 'suit'
    cut_xz: tuple[float, float] = (0.0, -0.04)
    wireframe: bool = False
    axes: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]] = field(default_factory=list)
    roi: tuple[int, int, int, int] | None = None  # (x0, y0, x1, y1) to read back; None = full frame


class _GpuPart:
    def __init__(self, ctx, part: MeshPart, programs: dict, textures: dict) -> None:
        self.part = part
        self.n = len(part.positions)
        self.base = np.ascontiguousarray(part.positions, np.float32)
        self.vbo_pos = ctx.buffer(self.base.tobytes(), dynamic=part.has_morphs)
        self.vbo_rest = ctx.buffer(self.base.tobytes())
        self.vbo_nrm = ctx.buffer(np.ascontiguousarray(part.normals, np.float32).tobytes())
        self.vbo_uv = ctx.buffer(np.ascontiguousarray(part.uv0, np.float32).tobytes())
        self.vbo_j = ctx.buffer(np.ascontiguousarray(part.joints, np.float32).tobytes())
        self.vbo_w = ctx.buffer(np.ascontiguousarray(part.weights, np.float32).tobytes())
        self.ibo = ctx.buffer(np.ascontiguousarray(part.indices, np.uint32).tobytes())
        self.vaos = {}
        for key, prog in programs.items():
            if key == "line":
                continue
            layout = [(self.vbo_pos, "3f", "in_pos"), (self.vbo_nrm, "3f", "in_nrm"), (self.vbo_uv, "2f", "in_uv"),
                      (self.vbo_j, "4f", "in_joints"), (self.vbo_w, "4f", "in_weights")]
            used = [x for x in layout if x[2] in prog]
            self.vaos[key] = ctx.vertex_array(prog, used, self.ibo, index_element_size=4)
        self.texture = textures.get(id(part.material))
        self._last_weights: tuple | None = None

    def update_morphs(self, weights: dict[str, float]) -> None:
        if not self.part.has_morphs:
            return
        w = tuple(round(float(weights.get(n, 0.0)), 4) for n in self.part.target_names)
        if w == self._last_weights:
            return
        self._last_weights = w
        pos = self.base.copy()
        for wi, delta in zip(w, self.part.targets):
            if abs(wi) > 1e-4:
                pos += wi * delta
        self.vbo_pos.write(pos.tobytes())


class CharacterRenderer:
    def __init__(self, asset: CharacterAsset, style: ToonStyle | None = None, samples: int = 4, ctx=None) -> None:
        import moderngl

        self.moderngl = moderngl
        self.asset = asset
        self.style = style or ToonStyle()
        self.ctx = ctx or moderngl.create_standalone_context(require=410)
        self.samples = int(min(samples, self.ctx.max_samples))
        self.programs = {
            "toon": self.ctx.program(vertex_shader=VERT, fragment_shader=TOON_FRAG),
            "ink": self.ctx.program(vertex_shader=VERT, fragment_shader=INK_FRAG),
            "lens": self.ctx.program(vertex_shader=VERT, fragment_shader=LENS_FRAG),
            "flat": self.ctx.program(vertex_shader=VERT, fragment_shader=FLAT_FRAG),
            "line": self.ctx.program(vertex_shader=LINE_VERT, fragment_shader=LINE_FRAG),
        }
        self.textures = {}
        for part in asset.parts:
            m = part.material
            if m.texture is not None and id(m) not in self.textures:
                t = self.ctx.texture((m.texture.shape[1], m.texture.shape[0]), 4,
                                     np.ascontiguousarray(m.texture).tobytes())
                t.build_mipmaps()
                t.repeat_x = t.repeat_y = False
                t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                t.anisotropy = 4.0
                self.textures[id(m)] = t
        self._dummy_tex = self.ctx.texture((1, 1), 4, bytes(4))  # keeps sampler unit 0 valid for untextured draws
        self.parts = [_GpuPart(self.ctx, p, self.programs, self.textures) for p in asset.parts]
        self.inverse_bind = asset.skeleton.inverse_bind
        self.joint_radius = self._joint_radii(asset)
        self._line_vbo = self.ctx.buffer(reserve=6 * 4 * 64, dynamic=True)
        self._line_vao = self.ctx.vertex_array(self.programs["line"],
                                               [(self._line_vbo, "3f 3f", "in_pos", "in_color")])
        self.size: tuple[int, int] | None = None
        self.timings: dict[str, float] = {}

    @staticmethod
    def _joint_radii(asset: CharacterAsset) -> dict[str, np.ndarray]:
        """Per role set: bounding radius (model units) of the vertices each joint dominates, for ROI culling."""
        nj = len(asset.skeleton.names)
        out = {}
        for key, roles in (("all", None), ("head", ("mask", "lens"))):
            rad = np.zeros(nj)
            for p in asset.parts:
                if roles is not None and p.material.role not in roles:
                    continue
                dom = p.joint_map[p.joints[np.arange(len(p.joints)), np.argmax(p.weights, 1)].astype(int)]
                rest = asset.skeleton.rest_world[dom][:, :3, 3]
                d = np.linalg.norm(p.positions - rest, axis=1)
                np.maximum.at(rad, dom, d)
            out[key] = rad
        return out

    def screen_roi(self, joint_world: np.ndarray, camera, size: tuple[int, int], roles: tuple[str, ...],
                   margin_px: float = 6.0) -> tuple[int, int, int, int] | None:
        """Pixel box that contains the posed character (projected bounding spheres of the joints)."""
        rad = self.joint_radius["all" if "suit" in roles else "head"]
        sel = rad > 0
        if not sel.any():
            return None
        pos = joint_world[sel, :3, 3]
        scale = np.linalg.norm(joint_world[sel, :3, 0], axis=1)
        depth = np.maximum(-pos[:, 2], 1.0)
        r_px = rad[sel] * scale * camera.focal * camera.refine_scale / depth * 1.15 + margin_px
        c = camera.project(pos)
        w, h = size
        x0 = int(max(0, np.floor((c[:, 0] - r_px).min())))
        y0 = int(max(0, np.floor((c[:, 1] - r_px).min())))
        x1 = int(min(w, np.ceil((c[:, 0] + r_px).max())))
        y1 = int(min(h, np.ceil((c[:, 1] + r_px).max())))
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    # framebuffers ----------------------------------------------------------------
    def _ensure_size(self, size: tuple[int, int]) -> None:
        if self.size == size:
            return
        w, h = size
        ctx = self.ctx
        self.msaa = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 4, samples=self.samples)],
                                    depth_attachment=ctx.depth_renderbuffer((w, h), samples=self.samples))
        self.resolved = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 4)])
        self.size = size

    # rendering --------------------------------------------------------------------
    def submit(self, frame: RenderFrame) -> None:
        """Issue all draw calls (returns before the GPU finishes; `read` waits)."""
        with self.ctx:  # current on this thread (tests and tools may hold several contexts)
            self._submit(frame)

    def _submit(self, frame: RenderFrame) -> None:
        t0 = time.perf_counter()
        ctx = self.ctx
        self._ensure_size(frame.size)
        w, h = frame.size
        st = self.style
        flip = np.diag([1.0, -1.0, 1.0, 1.0])  # render upside down so the readback is top-down
        proj = (flip @ frame.proj).T.astype("f4").tobytes()
        skin_all = np.einsum("jab,jbc->jac", frame.joint_world, self.inverse_bind)
        self.msaa.use()
        ctx.viewport = (0, 0, w, h)
        ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.enable(self.moderngl.DEPTH_TEST | self.moderngl.CULL_FACE)
        ctx.front_face = "cw"  # the flipped projection mirrors winding
        ctx.enable(self.moderngl.BLEND)
        ctx.blend_func = (self.moderngl.ONE, self.moderngl.ONE_MINUS_SRC_ALPHA)
        key = np.asarray(st.key_dir, np.float64)
        key /= np.linalg.norm(key)
        rim = np.asarray(st.rim_dir, np.float64)
        rim /= np.linalg.norm(rim)
        cut = frame.cut
        cut_vec = (1.0, cut[0], cut[1], cut[2]) if cut is not None else (0.0, 0.0, 0.0, 0.0)
        for prog in self.programs.values():
            for name, value in (("u_proj", proj), ("u_viewport", (float(w), float(h))),
                                ("u_key", tuple(key)), ("u_rim_dir", tuple(rim)), ("u_cut_xz", frame.cut_xz)):
                if name in prog:
                    if isinstance(value, bytes):
                        prog[name].write(value)
                    else:
                        prog[name].value = value
        toon = self.programs["toon"]
        for name, value in (("u_red", st.red), ("u_shadow_tint", st.shadow_tint), ("u_shadow_add", st.shadow_add),
                            ("u_sheen", st.sheen), ("u_web_faint_color", st.web_faint), ("u_highlight", st.highlight), ("u_rim_color", st.rim_color),
                            ("u_spec_color", st.spec_color), ("u_terminator", st.terminator),
                            ("u_ht_anchor", (frame.halftone_anchor[0], frame.halftone_anchor[1])),
                            ("u_ht_cell", max(2.0, frame.halftone_cell)), ("u_ht_strength", st.halftone_strength),
                            ("u_rim_strength", st.rim_strength), ("u_spec_strength", st.spec_strength)):
            if name in toon:
                toon[name].value = value
        self.programs["ink"]["u_ink"].value = st.ink
        lens = self.programs["lens"]
        for name, value in (("u_white", st.lens_white), ("u_lens_shade", st.lens_shade), ("u_ink", st.ink),
                            ("u_ink_px", st.lens_ink_px)):
            if name in lens:
                lens[name].value = value

        visible = [g for g in self.parts if g.part.material.role in frame.roles]
        for g in visible:
            g.update_morphs(frame.morph_weights)
        skins = {id(g): np.ascontiguousarray(skin_all[g.part.joint_map].transpose(0, 2, 1), np.float32).tobytes()
                 for g in visible}

        def set_common(prog, g, outline=0.0):
            prog["u_joints"].write(_pad_joints(skins[id(g)]))
            if "u_outline_px" in prog:
                prog["u_outline_px"].value = outline
            if "u_cut" in prog:
                prog["u_cut"].value = cut_vec if g.part.material.role == "suit" else (0.0, 0.0, 0.0, 0.0)

        # 1. toon colour pass
        ctx.cull_face = "back"
        for g in visible:
            m = g.part.material
            if m.shader != "toon":
                continue
            set_common(toon, g)
            toon["u_base"].value = tuple(float(c) for c in m.base_color[:3])
            has_tex = g.texture is not None
            toon["u_has_tex"].value = int(has_tex)
            toon["u_decal"].value = int(m.decal)
            (g.texture if has_tex else self._dummy_tex).use(0)
            toon["u_tex"].value = 0
            web = m.web
            toon["u_web_on"].value = int(bool(web))
            if web:
                axis = np.asarray(web.get("axis", (0, 0, 1)), np.float64)
                axis /= np.linalg.norm(axis)
                toon["u_web_center"].value = tuple(web.get("center", (0, 0, 0)))
                toon["u_web_axis"].value = tuple(axis)
                toon["u_web_ref"].value = tuple(web.get("ref", (0, 1, 0)))
                toon["u_web_radials"].value = float(web.get("radials", 16))
                toon["u_web_step"].value = float(web.get("ring_step", 0.25))
                toon["u_web_sag"].value = float(web.get("sag", 0.3))
                toon["u_web_width"].value = float(web.get("width", 0.001))
                toon["u_web_max"].value = float(web.get("max_angle", math.pi))
                toon["u_web_faint"].value = float(web.get("faint", 0.0))
            g.vaos["toon"].render()
        # 2. ink outline (inverted hull)
        if frame.outline_px > 0:
            ctx.cull_face = "front"
            ink = self.programs["ink"]
            for g in visible:
                if g.part.material.shader != "toon":
                    continue
                set_common(ink, g, frame.outline_px)
                g.vaos["ink"].render()
            ctx.cull_face = "back"
        # 3. lenses, pulled slightly toward the camera so they never z-fight the mask
        ctx.polygon_offset = (-2.0, -16.0)
        for g in visible:
            m = g.part.material
            if m.shader not in ("lens", "lens_flat"):
                continue
            set_common(lens, g)
            rim_red = m.params.get("rim_color", st.red)
            lens["u_rim_red"].value = tuple(float(c) for c in rim_red[:3])
            lens["u_flat"].value = int(m.shader == "lens_flat")
            g.vaos["lens"].render()
        ctx.polygon_offset = (0.0, 0.0)
        # 4. debug overlays
        if frame.wireframe:
            flat = self.programs["flat"]
            ctx.wireframe = True
            ctx.polygon_offset = (-1.0, -2.0)
            for g in visible:
                set_common(flat, g)
                flat["u_color"].value = (0.2, 1.0, 1.0, 0.55) if g.part.material.shader == "toon" else \
                    (1.0, 0.9, 0.2, 0.7)
                g.vaos["flat"].render()
            ctx.wireframe = False
            ctx.polygon_offset = (0.0, 0.0)
        if frame.axes:
            verts = []
            for a, b, color in frame.axes[:32]:
                verts += [*a, *color, *b, *color]
            data = np.asarray(verts, np.float32).tobytes()
            self._line_vbo.write(data)
            ctx.disable(self.moderngl.DEPTH_TEST)
            self._line_vao.render(self.moderngl.LINES, vertices=len(verts) // 6)
            ctx.enable(self.moderngl.DEPTH_TEST)
        self.timings = {"submit_ms": (time.perf_counter() - t0) * 1000}

    def read(self, roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """Resolve MSAA and read back premultiplied RGBA (H, W, 4) uint8, top row first."""
        t0 = time.perf_counter()
        w, h = self.size
        self.ctx.__enter__()
        self.ctx.copy_framebuffer(self.resolved, self.msaa)
        if roi is None:
            data = self.resolved.read(components=4, alignment=1)
            out = np.frombuffer(data, np.uint8).reshape(h, w, 4)
        else:
            x0, y0, x1, y1 = roi
            data = self.resolved.read(viewport=(x0, y0, x1 - x0, y1 - y0), components=4, alignment=1)
            out = np.zeros((h, w, 4), np.uint8)
            out[y0:y1, x0:x1] = np.frombuffer(data, np.uint8).reshape(y1 - y0, x1 - x0, 4)
        self.timings["read_ms"] = (time.perf_counter() - t0) * 1000
        return out

    def render(self, frame: RenderFrame) -> np.ndarray:
        self.submit(frame)
        return self.read(frame.roi)

    def release(self) -> None:
        try:
            self.ctx.release()
        except Exception:  # noqa: BLE001
            pass


def _pad_joints(data: bytes) -> bytes:
    full = MAX_JOINTS * 64
    return data + bytes(full - len(data)) if len(data) < full else data[:full]
