"""Minimal glTF 2.0 reader and writer (GLB container, plus .gltf with external or data-URI buffers).

Only what a rigged character needs: node hierarchy (TRS or matrix), meshes with
POSITION / NORMAL / TEXCOORD_n / JOINTS_0 / WEIGHTS_0 / custom `_` attributes,
morph targets (dense or sparse accessors, as Blender exports shape keys),
skins, materials with extras, and embedded PNG/JPEG textures.

No dependency beyond NumPy (and OpenCV to decode texture images).
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

GLB_MAGIC = 0x46546C67  # "glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

COMPONENT_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
DTYPE_COMPONENTS = {np.dtype(v): k for k, v in COMPONENT_DTYPES.items()}
TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_NORMALIZE_DIV = {np.uint8: 255.0, np.uint16: 65535.0, np.int8: 127.0, np.int16: 32767.0}


class GltfError(ValueError):
    pass


@dataclass
class GltfDocument:
    json: dict[str, Any]
    buffers: list[bytes]
    base_dir: Path | None = None

    # accessors ---------------------------------------------------------------
    def _view_bytes(self, view_index: int) -> tuple[bytes, int, int | None]:
        view = self.json["bufferViews"][view_index]
        buf = self.buffers[view["buffer"]]
        start = view.get("byteOffset", 0)
        return buf, start, view.get("byteStride")

    def _read_raw(self, view_index: int, offset: int, count: int, dtype, ncomp: int) -> np.ndarray:
        buf, start, stride = self._view_bytes(view_index)
        start += offset
        item = np.dtype(dtype).itemsize * ncomp
        if stride and stride != item:
            rows = np.frombuffer(buf, np.uint8, count=stride * (count - 1) + item, offset=start)
            idx = (np.arange(count)[:, None] * stride + np.arange(item)[None, :]).ravel()
            return np.frombuffer(rows[idx].tobytes(), dtype).reshape(count, ncomp)
        return np.frombuffer(buf, dtype, count=count * ncomp, offset=start).reshape(count, ncomp).copy()

    def accessor(self, index: int) -> np.ndarray:
        """Accessor as an (count, components) array (normalized integers become float32)."""
        acc = self.json["accessors"][index]
        dtype = COMPONENT_DTYPES[acc["componentType"]]
        ncomp = TYPE_SIZES[acc["type"]]
        count = acc["count"]
        if "bufferView" in acc:
            data = self._read_raw(acc["bufferView"], acc.get("byteOffset", 0), count, dtype, ncomp)
        else:
            data = np.zeros((count, ncomp), dtype)
        sparse = acc.get("sparse")
        if sparse:
            n = sparse["count"]
            si, sv = sparse["indices"], sparse["values"]
            idx = self._read_raw(si["bufferView"], si.get("byteOffset", 0), n,
                                 COMPONENT_DTYPES[si["componentType"]], 1).ravel()
            vals = self._read_raw(sv["bufferView"], sv.get("byteOffset", 0), n, dtype, ncomp)
            data = data.copy()
            data[idx.astype(np.int64)] = vals
        if acc.get("normalized") and dtype in _NORMALIZE_DIV:
            data = (data.astype(np.float32) / _NORMALIZE_DIV[dtype]).astype(np.float32)
        if acc["type"] == "MAT4":
            data = data.reshape(count, 4, 4).transpose(0, 2, 1)  # column-major -> row-major
        return data

    def image_bytes(self, image_index: int) -> bytes:
        img = self.json["images"][image_index]
        if "bufferView" in img:
            view = self.json["bufferViews"][img["bufferView"]]
            buf = self.buffers[view["buffer"]]
            start = view.get("byteOffset", 0)
            return bytes(buf[start:start + view["byteLength"]])
        return _load_uri(img["uri"], self.base_dir)

    def decode_image(self, image_index: int) -> np.ndarray:
        """Texture image as RGBA uint8 (H, W, 4)."""
        import cv2

        raw = np.frombuffer(self.image_bytes(image_index), np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise GltfError(f"cannot decode image {image_index}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return img

    # nodes ---------------------------------------------------------------------
    def node_local_matrix(self, index: int) -> np.ndarray:
        return node_matrix(self.json["nodes"][index])

    def node_parents(self) -> list[int]:
        parents = [-1] * len(self.json.get("nodes", []))
        for i, node in enumerate(self.json.get("nodes", [])):
            for c in node.get("children", []):
                parents[c] = i
        return parents

    def node_world_matrices(self) -> list[np.ndarray]:
        nodes = self.json.get("nodes", [])
        parents = self.node_parents()
        world: list[np.ndarray | None] = [None] * len(nodes)

        def resolve(i: int) -> np.ndarray:
            if world[i] is None:
                local = self.node_local_matrix(i)
                world[i] = local if parents[i] < 0 else resolve(parents[i]) @ local
            return world[i]

        return [resolve(i) for i in range(len(nodes))]


def node_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], np.float64).reshape(4, 4).T
    t = np.asarray(node.get("translation", (0, 0, 0)), np.float64)
    q = np.asarray(node.get("rotation", (0, 0, 0, 1)), np.float64)
    s = np.asarray(node.get("scale", (1, 1, 1)), np.float64)
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(q) * s[None, :]
    m[:3, 3] = t
    return m


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """glTF quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q, np.float64) / max(float(np.linalg.norm(q)), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _load_uri(uri: str, base_dir: Path | None) -> bytes:
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    if base_dir is None:
        raise GltfError(f"external URI {uri!r} without a base directory")
    return (base_dir / uri).read_bytes()


def read_gltf(path: str | Path) -> GltfDocument:
    path = Path(path)
    data = path.read_bytes()
    if len(data) >= 12 and struct.unpack_from("<I", data, 0)[0] == GLB_MAGIC:
        magic, version, length = struct.unpack_from("<III", data, 0)
        if version != 2:
            raise GltfError(f"unsupported GLB version {version}")
        offset, doc_json, bin_chunk = 12, None, b""
        while offset < min(length, len(data)):
            chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
            chunk = data[offset + 8: offset + 8 + chunk_len]
            if chunk_type == CHUNK_JSON:
                doc_json = json.loads(chunk.decode("utf-8"))
            elif chunk_type == CHUNK_BIN:
                bin_chunk = chunk
            offset += 8 + chunk_len
        if doc_json is None:
            raise GltfError("GLB without a JSON chunk")
    else:
        doc_json = json.loads(data.decode("utf-8"))
        bin_chunk = b""
    buffers = []
    for i, b in enumerate(doc_json.get("buffers", [])):
        if "uri" in b:
            buffers.append(_load_uri(b["uri"], path.parent))
        elif i == 0:
            buffers.append(bin_chunk)
        else:
            raise GltfError(f"buffer {i} has no data")
    if doc_json.get("asset", {}).get("version", "2.0").split(".")[0] != "2":
        raise GltfError("only glTF 2.x is supported")
    return GltfDocument(doc_json, buffers, path.parent)


# writing -------------------------------------------------------------------------
@dataclass
class GltfBuilder:
    """Accumulates JSON + one binary buffer and writes a .glb file."""

    json: dict[str, Any] = field(default_factory=lambda: {
        "asset": {"version": "2.0", "generator": "spiderverse-portal prototype builder"},
        "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "accessors": [],
        "bufferViews": [], "buffers": [], "materials": [],
    })
    _bin: bytearray = field(default_factory=bytearray)

    def _align(self) -> None:
        while len(self._bin) % 4:
            self._bin.append(0)

    def buffer_view(self, data: bytes, target: int | None = None) -> int:
        self._align()
        view = {"buffer": 0, "byteOffset": len(self._bin), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self._bin.extend(data)
        self.json["bufferViews"].append(view)
        return len(self.json["bufferViews"]) - 1

    def accessor(self, array: np.ndarray, kind: str, target: int | None = None, normalized: bool = False,
                 minmax: bool = False) -> int:
        arr = np.ascontiguousarray(array)
        if kind == "MAT4":
            arr = np.ascontiguousarray(arr.reshape(-1, 4, 4).transpose(0, 2, 1))  # to column-major
        count = arr.shape[0]
        view = self.buffer_view(arr.tobytes(), target)
        acc: dict[str, Any] = {"bufferView": view, "componentType": DTYPE_COMPONENTS[arr.dtype], "count": count,
                               "type": kind}
        if normalized:
            acc["normalized"] = True
        if minmax:
            flat = arr.reshape(count, -1)
            acc["min"] = [float(v) for v in flat.min(axis=0)]
            acc["max"] = [float(v) for v in flat.max(axis=0)]
        self.json["accessors"].append(acc)
        return len(self.json["accessors"]) - 1

    def add_node(self, node: dict[str, Any], parent: int | None = None) -> int:
        self.json["nodes"].append(node)
        idx = len(self.json["nodes"]) - 1
        if parent is None:
            self.json["scenes"][0]["nodes"].append(idx)
        else:
            self.json["nodes"][parent].setdefault("children", []).append(idx)
        return idx

    def add_material(self, material: dict[str, Any]) -> int:
        self.json["materials"].append(material)
        return len(self.json["materials"]) - 1

    def add_png_texture(self, png: bytes, name: str) -> int:
        view = self.buffer_view(png)
        self.json.setdefault("images", []).append({"bufferView": view, "mimeType": "image/png", "name": name})
        self.json.setdefault("samplers", [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}])
        self.json.setdefault("textures", []).append({"source": len(self.json["images"]) - 1, "sampler": 0})
        return len(self.json["textures"]) - 1

    def add_mesh(self, name: str, primitives: list[dict[str, Any]], target_names: list[str] | None = None) -> int:
        mesh: dict[str, Any] = {"name": name, "primitives": primitives}
        if target_names:
            mesh["extras"] = {"targetNames": list(target_names)}
            mesh["weights"] = [0.0] * len(target_names)
        self.json["meshes"].append(mesh)
        return len(self.json["meshes"]) - 1

    def primitive(self, positions: np.ndarray, normals: np.ndarray, indices: np.ndarray, material: int,
                  uv0: np.ndarray | None = None, joints: np.ndarray | None = None, weights: np.ndarray | None = None,
                  targets: list[np.ndarray] | None = None, extra_attributes: dict[str, np.ndarray] | None = None
                  ) -> dict[str, Any]:
        attrs = {
            "POSITION": self.accessor(positions.astype(np.float32), "VEC3", 34962, minmax=True),
            "NORMAL": self.accessor(normals.astype(np.float32), "VEC3", 34962),
        }
        if uv0 is not None:
            attrs["TEXCOORD_0"] = self.accessor(uv0.astype(np.float32), "VEC2", 34962)
        if joints is not None:
            attrs["JOINTS_0"] = self.accessor(joints.astype(np.uint16), "VEC4", 34962)
            attrs["WEIGHTS_0"] = self.accessor(weights.astype(np.float32), "VEC4", 34962)
        for key, arr in (extra_attributes or {}).items():
            kind = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}[arr.shape[1]]
            attrs[key] = self.accessor(arr.astype(np.float32), kind, 34962)
        prim: dict[str, Any] = {"attributes": attrs, "mode": 4, "material": material,
                                "indices": self.accessor(indices.astype(np.uint32).reshape(-1, 1), "SCALAR", 34963)}
        if targets:
            prim["targets"] = [{"POSITION": self.accessor(t.astype(np.float32), "VEC3", 34962, minmax=True)}
                               for t in targets]
        return prim

    def add_skin(self, joints: list[int], inverse_bind: np.ndarray, name: str, skeleton: int | None = None) -> int:
        skin: dict[str, Any] = {"name": name, "joints": list(joints),
                                "inverseBindMatrices": self.accessor(inverse_bind.astype(np.float32), "MAT4")}
        if skeleton is not None:
            skin["skeleton"] = skeleton
        self.json.setdefault("skins", []).append(skin)
        return len(self.json["skins"]) - 1

    def to_glb(self) -> bytes:
        self._align()
        doc = dict(self.json)
        doc["buffers"] = [{"byteLength": len(self._bin)}]
        for key in ("materials", "meshes", "accessors", "bufferViews"):
            if not doc.get(key):
                doc.pop(key, None)
        js = json.dumps(doc, separators=(",", ":")).encode("utf-8")
        js += b" " * ((4 - len(js) % 4) % 4)
        binary = bytes(self._bin)
        total = 12 + 8 + len(js) + 8 + len(binary)
        out = bytearray(struct.pack("<III", GLB_MAGIC, 2, total))
        out += struct.pack("<II", len(js), CHUNK_JSON) + js
        out += struct.pack("<II", len(binary), CHUNK_BIN) + binary
        return bytes(out)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(self.to_glb())
        tmp.replace(path)
        return path
