"""Synthetic head turns from a single photo, for testing head tracking without a person.

The face landmarker's 478 3D landmarks on a frontal photo, triangulated with
the canonical face topology and textured with the photo, are rotated about
the head centre and re-rendered (orthographic, OpenGL). The face landmarker
then runs on the rendered image. This gives images of the same face turned by
a known angle; only the face surface is rotated (hair and body stay as a
blurred backdrop), which is enough for the face tracker.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

VERT = """
#version 410
uniform mat4 u_m;
in vec3 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = u_m * vec4(in_pos, 1.0); }
"""
FRAG = """
#version 410
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 o;
void main() { o = texture(u_tex, v_uv); }
"""


def detect(landmarker, bgr):
    import mediapipe as mp

    h, w = bgr.shape[:2]
    r = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    if not r.face_landmarks:
        return None
    lm = np.array([[p.x * w, p.y * h, p.z * w] for p in r.face_landmarks[0]])
    shapes = {c.category_name: c.score for c in r.face_blendshapes[0]} if r.face_blendshapes else {}
    matrix = np.asarray(r.facial_transformation_matrixes[0]) if r.facial_transformation_matrixes else None
    return lm, shapes, matrix


def make_landmarker(task_path: str):
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=task_path), num_faces=1, output_face_blendshapes=True,
        output_facial_transformation_matrixes=True))


class FaceTurner:
    def __init__(self, photo: np.ndarray, landmarks3d: np.ndarray, triangles: np.ndarray) -> None:
        import moderngl

        self.photo = photo
        h, w = photo.shape[:2]
        self.size = (w, h)
        self.ctx = moderngl.create_standalone_context(require=410)
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        p = landmarks3d[:468].astype(np.float64)
        self.center = np.array([p[:, 0].mean(), p[:, 1].mean(), 0.0])
        uv = np.c_[p[:, 0] / w, p[:, 1] / h].astype("f4")
        self.pos = p
        self.vbo = self.ctx.buffer(np.zeros((468, 3), "f4").tobytes(), dynamic=True)
        self.uvb = self.ctx.buffer(uv.tobytes())
        self.ibo = self.ctx.buffer(np.asarray(triangles, np.uint32).tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "3f", "in_pos"), (self.uvb, "2f", "in_uv")],
                                         self.ibo, index_element_size=4)
        rgba = cv2.cvtColor(photo, cv2.COLOR_BGR2RGBA)
        self.tex = self.ctx.texture((w, h), 4, rgba.tobytes())
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer((w, h), 4)],
                                        depth_attachment=self.ctx.depth_renderbuffer((w, h)))
        self.backdrop = cv2.GaussianBlur(photo, (0, 0), 9)

    def render(self, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
        import moderngl

        w, h = self.size
        y, x = math.radians(yaw_deg), math.radians(pitch_deg)
        ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
        # Image coordinates (y down, z away from the camera) -> rotate about the head centre.
        q = (self.pos - self.center) @ (ry @ rx).T + self.center
        with self.ctx:  # make this context current (other GL contexts may exist in the process)
            return self._draw(q)

    def _draw(self, q: np.ndarray) -> np.ndarray:
        import moderngl

        w, h = self.size
        ndc = np.c_[q[:, 0] / w * 2 - 1, 1 - q[:, 1] / h * 2, q[:, 2] / w].astype("f4")
        self.vbo.write(ndc.tobytes())
        self.fbo.use()
        self.ctx.clear(0, 0, 0, 0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.tex.use(0)
        self.prog["u_m"].write(np.eye(4, dtype="f4").tobytes())
        self.vao.render()
        img = np.frombuffer(self.fbo.read(components=4), np.uint8).reshape(h, w, 4)[::-1]
        a = img[..., 3:4].astype(np.float32) / 255.0
        a = cv2.GaussianBlur(a, (0, 0), 1.0)[..., None]
        face = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR).astype(np.float32)
        return (face * a + self.backdrop.astype(np.float32) * (1 - a)).astype(np.uint8)

    def release(self) -> None:
        self.ctx.release()
