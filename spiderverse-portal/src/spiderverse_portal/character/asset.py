"""Character asset loading: any glTF 2.0 / GLB that follows docs/CHARACTER_ASSET_SPEC.md.

Everything is normalised into one representation the renderer and rig use:

  * a skeleton (joints, parents, rest world matrices) in model space,
  * skinned parts whose vertices are in that rest pose ("bind space"), with
    inverse bind = inverse(rest world). Skins whose inverse-bind matrices do
    not match the node rest pose are re-posed on load; rigid meshes parented
    under a joint become skinned 100 % to that joint,
  * named morph targets (position deltas),
  * materials: base colour / texture plus the `extras.spiderverse` block
    (shader, role, decal, procedural web parameters),
  * the head alignment: how the MediaPipe canonical face (cm) sits in the head
    joint's local frame, from `align_*` empties if present.

Joint names are matched by role (head, neck, chest, ...) with aliases for
Blender, Rigify, Mixamo and VRM naming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .face_geometry import CHIN, EYE_OUTER_L, EYE_OUTER_R, FOREHEAD, NOSE_TIP, canonical_face, default_task_path
from .gltf import GltfDocument, GltfError, read_gltf

ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "root": ("root", "hips", "pelvis", "mixamorig:hips", "def-spine", "j_bip_c_hips"),
    "spine": ("spine", "mixamorig:spine", "mixamorig:spine1", "def-spine.001", "j_bip_c_spine"),
    "chest": ("chest", "upperchest", "upper_chest", "mixamorig:spine2", "def-spine.003", "j_bip_c_upperchest",
              "j_bip_c_chest"),
    "neck": ("neck", "mixamorig:neck", "def-spine.004", "j_bip_c_neck"),
    "head": ("head", "mixamorig:head", "def-spine.006", "j_bip_c_head"),
}
for _side, _word in (("L", "left"), ("R", "right")):
    ROLE_ALIASES[f"shoulder.{_side}"] = (f"shoulder.{_side}", f"shoulder_{_side}", f"{_word}shoulder",
                                          f"mixamorig:{_word}shoulder", f"def-shoulder.{_side}",
                                          f"j_bip_{_side}_shoulder", f"clavicle.{_side}", f"clavicle_{_side}")
    ROLE_ALIASES[f"upper_arm.{_side}"] = (f"upper_arm.{_side}", f"upper_arm_{_side}", f"upperarm.{_side}",
                                           f"{_word}arm", f"{_word}upperarm", f"mixamorig:{_word}arm",
                                           f"def-upper_arm.{_side}", f"j_bip_{_side}_upperarm")
    ROLE_ALIASES[f"forearm.{_side}"] = (f"forearm.{_side}", f"forearm_{_side}", f"lowerarm.{_side}",
                                         f"{_word}forearm", f"{_word}lowerarm", f"mixamorig:{_word}forearm",
                                         f"def-forearm.{_side}", f"j_bip_{_side}_lowerarm")
    ROLE_ALIASES[f"hand.{_side}"] = (f"hand.{_side}", f"hand_{_side}", f"{_word}hand", f"mixamorig:{_word}hand",
                                      f"def-hand.{_side}", f"j_bip_{_side}_hand")
ROLES = tuple(ROLE_ALIASES)
ALIGN_LANDMARKS = {"align_eye_outer_l": EYE_OUTER_L, "align_eye_outer_r": EYE_OUTER_R, "align_chin": CHIN,
                   "align_nose_tip": NOSE_TIP, "align_forehead": FOREHEAD}


@dataclass
class MaterialDef:
    name: str
    shader: str = "toon"  # toon | lens | lens_flat
    role: str = "other"  # mask | suit | lens | other
    base_color: np.ndarray = field(default_factory=lambda: np.ones(4, np.float32))
    texture: np.ndarray | None = None  # RGBA uint8
    decal: bool = False  # texture alpha-blends over base_color instead of multiplying it
    web: dict[str, Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MeshPart:
    name: str
    positions: np.ndarray  # (N, 3) float32, bind space (model metres)
    normals: np.ndarray
    uv0: np.ndarray
    joints: np.ndarray  # (N, 4) float32 indices into `joint_map`
    weights: np.ndarray  # (N, 4) float32
    indices: np.ndarray  # (M,) uint32
    material: MaterialDef
    joint_map: np.ndarray  # local joint index -> skeleton joint index
    target_names: list[str] = field(default_factory=list)
    targets: np.ndarray | None = None  # (T, N, 3) float32

    @property
    def has_morphs(self) -> bool:
        return self.targets is not None and len(self.target_names) > 0


@dataclass
class Skeleton:
    names: list[str]
    parents: np.ndarray  # (J,) int, -1 for roots
    rest_world: np.ndarray  # (J, 4, 4) model space
    roles: dict[str, int]

    @property
    def inverse_bind(self) -> np.ndarray:
        return np.linalg.inv(self.rest_world)

    def index(self, role: str) -> int:
        return self.roles.get(role, -1)

    def rest_local(self) -> np.ndarray:
        out = self.rest_world.copy()
        for j, p in enumerate(self.parents):
            if p >= 0:
                out[j] = np.linalg.inv(self.rest_world[p]) @ self.rest_world[j]
        return out


@dataclass
class CharacterAsset:
    path: Path
    parts: list[MeshPart]
    skeleton: Skeleton
    head_from_canonical: np.ndarray  # 4x4: canonical face (cm) -> head joint local frame (asset units)
    prototype: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def morph_names(self) -> list[str]:
        names: list[str] = []
        for p in self.parts:
            names += [n for n in p.target_names if n not in names]
        return names

    def summary(self) -> str:
        nv = sum(len(p.positions) for p in self.parts)
        nt = sum(len(p.indices) // 3 for p in self.parts)
        kind = "PROTOTYPE" if self.prototype else "asset"
        return (f"{kind} {self.path.name}: {len(self.parts)} parts, {nv} verts, {nt} tris, "
                f"{len(self.skeleton.names)} joints, {len(self.morph_names)} morphs")


def _match_roles(names: list[str]) -> dict[str, int]:
    lower = [n.lower() for n in names]
    roles: dict[str, int] = {}
    for role, aliases in ROLE_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower:
                roles[role] = lower.index(alias.lower())
                break
    return roles


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Similarity (scale, rotation, translation) mapping src -> dst, as a 4x4 matrix."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    cov = b.T @ a / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2, 2] = -1
    r = u @ d @ vt
    scale = np.trace(np.diag(s) @ d) / max((a * a).sum() / len(src), 1e-12)
    m = np.eye(4)
    m[:3, :3] = scale * r
    m[:3, 3] = md - scale * r @ ms
    return m


def load_character(path: str | Path, task_path: str | Path | None = None) -> CharacterAsset:
    path = Path(path)
    doc = read_gltf(path)
    js = doc.json
    nodes = js.get("nodes", [])
    world = doc.node_world_matrices()
    parents = doc.node_parents()

    # Skeleton: the union of all skins' joints (by node index), in node order.
    joint_nodes: list[int] = []
    for skin in js.get("skins", []):
        for n in skin["joints"]:
            if n not in joint_nodes:
                joint_nodes.append(n)
    if not joint_nodes:  # unrigged asset: synthesise a single head joint at the origin
        raise GltfError("character asset has no skin; a rig with at least a 'head' joint is required")
    node_to_joint = {n: i for i, n in enumerate(joint_nodes)}
    names = [nodes[n].get("name", f"joint{i}") for i, n in enumerate(joint_nodes)]
    jparents = []
    for n in joint_nodes:
        p = parents[n]
        while p >= 0 and p not in node_to_joint:
            p = parents[p]
        jparents.append(node_to_joint.get(p, -1))
    rest_world = np.stack([world[n] for n in joint_nodes])
    skeleton = Skeleton(names, np.asarray(jparents), rest_world, _match_roles(names))
    if skeleton.index("head") < 0:
        raise GltfError(f"no head joint found (joints: {names[:12]}...)")

    materials = [_material(doc, i) for i in range(len(js.get("materials", [])))]
    default_mat = MaterialDef("default")
    parts: list[MeshPart] = []
    for ni, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        mesh = js["meshes"][node["mesh"]]
        tnames = list(mesh.get("extras", {}).get("targetNames", []))
        skin_idx = node.get("skin")
        for pi, prim in enumerate(mesh["primitives"]):
            if prim.get("mode", 4) != 4:
                continue
            part = _part(doc, prim, f"{mesh.get('name', node.get('name', ni))}#{pi}", tnames,
                         materials[prim["material"]] if "material" in prim else default_mat)
            if skin_idx is not None:
                _bind_skinned(part, js["skins"][skin_idx], doc, node_to_joint, rest_world)
            else:
                _bind_rigid(part, ni, parents, world, node_to_joint, skeleton, node.get("name", ""))
            _compact_joints(part)
            parts.append(part)
    if not parts:
        raise GltfError("character asset has no triangle meshes")

    head = skeleton.index("head")
    head_from_canonical, align_source = _head_alignment(doc, world, rest_world[head], task_path)
    extras = dict(js.get("asset", {}).get("extras", {}).get("spiderverse", {}))
    extras["head_alignment"] = align_source
    return CharacterAsset(path, parts, skeleton, head_from_canonical, bool(extras.get("prototype", False)), extras)


def _material(doc: GltfDocument, index: int) -> MaterialDef:
    m = doc.json["materials"][index]
    pbr = m.get("pbrMetallicRoughness", {})
    ex = m.get("extras", {}).get("spiderverse", {})
    name = m.get("name", f"material{index}")
    tex = None
    if "baseColorTexture" in pbr:
        t = doc.json["textures"][pbr["baseColorTexture"]["index"]]
        if "source" in t:
            tex = doc.decode_image(t["source"])
    lname = name.lower()
    guess_lens = any(k in lname for k in ("lens", "eye"))
    return MaterialDef(
        name=name,
        shader=ex.get("shader", "lens_flat" if guess_lens else "toon"),
        role=ex.get("role", "lens" if guess_lens else "mask" if "mask" in lname or "head" in lname else "suit"),
        base_color=np.asarray(pbr.get("baseColorFactor", (1, 1, 1, 1)), np.float32),
        texture=tex,
        decal=bool(ex.get("decal", False)),
        web=ex.get("web"),
        params={k: v for k, v in ex.items() if k not in ("shader", "role", "decal", "web")},
    )


def _part(doc: GltfDocument, prim: dict, name: str, tnames: list[str], material: MaterialDef) -> MeshPart:
    attrs = prim["attributes"]
    pos = doc.accessor(attrs["POSITION"]).astype(np.float32)
    n = len(pos)
    idx = doc.accessor(prim["indices"]).ravel().astype(np.uint32) if "indices" in prim else \
        np.arange(n, dtype=np.uint32)
    if "NORMAL" in attrs:
        nrm = doc.accessor(attrs["NORMAL"]).astype(np.float32)
    else:
        from .meshtools import vertex_normals

        nrm = vertex_normals(pos, idx.reshape(-1, 3)).astype(np.float32)
    uv = doc.accessor(attrs["TEXCOORD_0"]).astype(np.float32) if "TEXCOORD_0" in attrs else np.zeros((n, 2), np.float32)
    joints = doc.accessor(attrs["JOINTS_0"]).astype(np.float32) if "JOINTS_0" in attrs else np.zeros((n, 4), np.float32)
    weights = doc.accessor(attrs["WEIGHTS_0"]).astype(np.float32) if "WEIGHTS_0" in attrs else \
        np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
    targets = None
    names: list[str] = []
    if prim.get("targets"):
        tl = []
        for ti, tg in enumerate(prim["targets"]):
            if "POSITION" not in tg:
                continue
            tl.append(doc.accessor(tg["POSITION"]).astype(np.float32))
            names.append(tnames[ti] if ti < len(tnames) else f"target_{ti}")
        targets = np.stack(tl) if tl else None
    return MeshPart(name, pos, nrm, uv, joints, weights, idx, material, np.zeros(0, np.int64), names, targets)


def _bind_skinned(part: MeshPart, skin: dict, doc: GltfDocument, node_to_joint: dict[int, int],
                  rest_world: np.ndarray) -> None:
    jmap = np.array([node_to_joint[n] for n in skin["joints"]], np.int64)
    ibm = doc.accessor(skin["inverseBindMatrices"]) if "inverseBindMatrices" in skin else \
        np.tile(np.eye(4), (len(jmap), 1, 1))
    # Re-pose to the node rest pose if the skin was bound elsewhere: v' = sum w * rest * ibm * v.
    pose = np.einsum("jab,jbc->jac", rest_world[jmap], ibm)
    if not np.allclose(pose, np.eye(4)[None], atol=1e-4):
        ji = part.joints.astype(np.int64)
        m = np.einsum("nk,nkab->nab", part.weights, pose[ji])
        p = np.c_[part.positions, np.ones(len(part.positions))]
        part.positions = np.einsum("nab,nb->na", m, p)[:, :3].astype(np.float32)
        nrm = np.einsum("nab,nb->na", m[:, :3, :3], part.normals)
        part.normals = (nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        if part.targets is not None:
            part.targets = np.einsum("nab,tnb->tna", m[:, :3, :3], part.targets).astype(np.float32)
    part.joint_map = jmap
    s = part.weights.sum(1, keepdims=True)
    part.weights = (part.weights / np.where(s > 0, s, 1)).astype(np.float32)


def _bind_rigid(part: MeshPart, node: int, parents: list[int], world: list[np.ndarray], node_to_joint: dict[int, int],
                skeleton: Skeleton, node_name: str) -> None:
    m = world[node]
    p = np.c_[part.positions, np.ones(len(part.positions))]
    part.positions = (p @ m.T)[:, :3].astype(np.float32)
    nrm = part.normals @ np.linalg.inv(m[:3, :3])
    part.normals = (nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    if part.targets is not None:
        part.targets = np.einsum("ab,tnb->tna", m[:3, :3], part.targets).astype(np.float32)
    a = parents[node]
    while a >= 0 and a not in node_to_joint:
        a = parents[a]
    if a >= 0:
        joint = node_to_joint[a]
    else:
        lname = (node_name + part.name).lower()
        joint = skeleton.index("head") if any(k in lname for k in ("head", "mask", "lens", "eye")) else 0
    part.joint_map = np.array([joint], np.int64)
    part.joints = np.zeros((len(part.positions), 4), np.float32)
    part.weights = np.tile(np.array([1, 0, 0, 0], np.float32), (len(part.positions), 1))


def _compact_joints(part: MeshPart) -> None:
    """Keep only joints with weight in this part (the shader takes at most 64 per draw)."""
    ji = part.joints.astype(np.int64)
    used = np.unique(ji[part.weights > 0])
    if len(used) > 64:
        raise GltfError(f"part {part.name} uses {len(used)} joints; at most 64 per mesh primitive are supported")
    remap = -np.ones(max(int(ji.max()) + 1, 1), np.int64)
    remap[used] = np.arange(len(used))
    local = np.where(part.weights > 0, remap[ji], 0)
    part.joints = local.astype(np.float32)
    part.joint_map = part.joint_map[used]


def _head_alignment(doc: GltfDocument, world: list[np.ndarray], head_rest: np.ndarray,
                    task_path: str | Path | None) -> tuple[np.ndarray, str]:
    """canonical (cm) -> head joint local. From align_* empties when present, else head frame = canonical/100."""
    marks = {}
    for i, node in enumerate(doc.json.get("nodes", [])):
        key = node.get("name", "").lower()
        if key in ALIGN_LANDMARKS:
            marks[key] = world[i][:3, 3]
    if len(marks) >= 3:
        canon, _, _ = canonical_face(str(task_path or default_task_path()))
        src = np.array([canon[ALIGN_LANDMARKS[k]] for k in marks])
        dst = np.array([marks[k] for k in marks])
        model_from_canon = _umeyama(src, dst)
        return np.linalg.inv(head_rest) @ model_from_canon, f"align empties ({len(marks)})"
    s = np.eye(4)
    s[:3, :3] *= 0.01
    return s, "head joint frame = canonical face / 100"
