"""Build the MuJoCo scene (MJCF plus in-memory texture assets) for a scenario.

MuJoCo only rasterises geometry here: depth, segmentation and unlit surface colour (albedo), with a
pure ambient headlight of strength 1 and no other light. All lighting, water and sensor effects are
computed by optics.py from that geometry.
"""
from __future__ import annotations

import numpy as np

from .camera import axes, fovy_deg
from .rng import stream
from .scenario import Scenario, png_bytes
from .texture import periodic_noise

# MuJoCo maps mesh texture coordinate v = 0 to the first image row (no flip); checked by
# tests/test_render.py::test_labels_align_with_rendered_texture.
TEXTURE_FLIP_V = False


def _f(*v) -> str:
    return ' '.join(f'{x:.6g}' for x in v)


def _seabed_texture(scn: Scenario) -> np.ndarray:
    w = scn.cfg['water']
    rng = stream(scn.seed, 'seabed')
    n = w['seabed_texture_px']
    shade = 1.0 + sum(amp * periodic_noise(n, px, rng) for amp, px in zip(w['seabed_variation'], w['seabed_noise_px']))
    tex = np.asarray(w['seabed_rgb'], np.float32)[None, None, :] * shade[..., None]
    return (np.clip(tex, 0, 1) * 255).astype(np.uint8)


def build_mjcf(scn: Scenario) -> tuple[str, dict[str, bytes]]:
    cfg, hull = scn.cfg, scn.hull
    cams, w = cfg['cameras'], cfg['water']
    verts, uv, faces = hull.mesh(*cfg['hull']['mesh_cells'], flip_v=TEXTURE_FLIP_V)
    assets = {'hull_texture.png': png_bytes(scn.texture), 'seabed.png': png_bytes(_seabed_texture(scn))}
    a = cfg['anodes']
    anodes = []
    for it in scn.anodes():
        L, T, H = it['size_m']
        p = np.asarray(it['centroid_world']) + np.asarray(hull.normal(it['centroid_xs'][1])) * T / 2
        rgb = a['depleted_rgb'] if it['status'] == 'depleted' else a['rgb']
        anodes.append(f'<geom name="anode_{it["id"]}" type="box" pos="{_f(*p)}" size="{_f(L / 2, T / 2, H / 2)}" '
                      f'rgba="{_f(*rgb, 1)}" contype="0" conaffinity="0"/>')
    fovy = fovy_deg(cams)
    cam_xml = []
    for name, c in cams['rig'].items():
        m = axes(c['look'])
        cam_xml.append(f'<camera name="{name}" pos="{_f(*c["offset"])}" xyaxes="{_f(*m[:, 0], *m[:, 1])}" fovy="{fovy:.6g}"/>')
    zbed = -(hull.side_depth + hull.bilge_radius + w['seabed_clearance_m'])
    span = max(hull.length, hull.girth) * 4
    xml = f"""<mujoco model="hull_inspection">
  <compiler angle="radian"/>
  <statistic extent="1" center="{_f(hull.length / 2, hull.offset_y, -hull.side_depth)}"/>
  <option gravity="0 0 0"/>
  <visual>
    <global offwidth="{cams['width']}" offheight="{cams['height']}"/>
    <headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/>
    <map znear="{cams['znear_m']}" zfar="{cams['zfar_m']}"/>
    <quality shadowsize="0"/>
  </visual>
  <asset>
    <texture name="hull_tex" type="2d" file="hull_texture.png"/>
    <material name="hull_mat" texture="hull_tex" specular="0" shininess="0"/>
    <texture name="seabed_tex" type="2d" file="seabed.png"/>
    <material name="seabed_mat" texture="seabed_tex" texuniform="true" texrepeat="{_f(*[1 / w['seabed_texture_m']] * 2)}" specular="0"/>
    <mesh name="hull_mesh" inertia="shell" vertex="{_f(*verts.ravel())}" texcoord="{_f(*uv.ravel())}" face="{' '.join(map(str, faces.ravel()))}"/>
  </asset>
  <worldbody>
    <geom name="hull" type="mesh" mesh="hull_mesh" material="hull_mat" contype="0" conaffinity="0"/>
    <geom name="seabed" type="plane" pos="{_f(hull.length / 2, hull.offset_y, zbed)}" size="{_f(span, span, 0.1)}" material="seabed_mat" contype="0" conaffinity="0"/>
    {chr(10).join('    ' + g for g in anodes)}
    <body name="rig" mocap="true" pos="0 0 -1">
      {chr(10).join('      ' + c for c in cam_xml)}
    </body>
  </worldbody>
</mujoco>
"""
    return xml, assets
