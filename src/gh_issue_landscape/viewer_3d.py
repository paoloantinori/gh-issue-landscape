"""3D interactive point-cloud viewer using Three.js.

Generates a self-contained HTML file that renders UMAP 3-D coordinates
as an interactive Three.js scene with per-topic coloring, hover tooltips,
click-to-open, and a legend panel.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from gh_issue_landscape.pipeline import PipelineResult

log = logging.getLogger(__name__)

_OUTLIER_COLOR = "#999999"


def _assign_topic_colors(topic_ids: set[int]) -> dict[int, str]:
    """Map each topic id to a hex color string.

    Uses a muted cartographic palette that reads well on light backgrounds.
    """
    palette = [
        "#9b4d6a",  # rose
        "#6b70a8",  # periwinkle
        "#7aab88",  # sage
        "#b07040",  # sienna
        "#a07098",  # orchid
        "#7a8b50",  # olive
        "#4080a0",  # teal
        "#5878a0",  # steel
        "#40908a",  # cyan-teal
        "#388040",  # emerald
        "#8a8030",  # gold-olive
        "#9088b0",  # lavender
        "#a88050",  # amber
        "#b04030",  # brick
        "#7840a0",  # purple
        "#a06060",  # dusty rose
        "#4890b0",  # sky blue
        "#80a040",  # lime
        "#a04888",  # magenta
        "#6070a0",  # slate
    ]
    colors: dict[int, str] = {}
    idx = 0
    for tid in sorted(topic_ids):
        if tid == -1:
            colors[tid] = _OUTLIER_COLOR
        else:
            colors[tid] = palette[idx % len(palette)]
            idx += 1
    return colors


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Center coordinates at origin and scale to roughly [-1, 1]."""
    centered = coords - coords.mean(axis=0)
    max_abs = np.abs(centered).max()
    if max_abs > 0:
        centered = centered / max_abs
    return centered


def generate_3d_explorer(
    result: PipelineResult,
    issues: list[dict],
    output_dir: str = "./output",
) -> Path:
    """Generate a self-contained HTML file with a Three.js 3D point cloud."""
    if result.umap_3d is None:
        raise ValueError(
            "PipelineResult.umap_3d is None. "
            "Re-run the pipeline with reduce_3d=True."
        )

    out_path = Path(output_dir) / "landscape-3d.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    coords = _normalize_coords(result.umap_3d)
    topic_ids = set(result.topics)
    topic_colors = _assign_topic_colors(topic_ids)

    points: list[dict] = []
    for i, issue in enumerate(issues):
        tid = result.topics[i]
        points.append(
            {
                "x": round(float(coords[i, 0]), 5),
                "y": round(float(coords[i, 1]), 5),
                "z": round(float(coords[i, 2]), 5),
                "topic_id": int(tid),
                "number": issue["number"],
                "title": issue["title"],
                "url": issue.get("url", ""),
                "label": result.get_label(tid),
            }
        )

    legend_entries: list[dict] = []
    for tid in sorted(topic_ids):
        legend_entries.append(
            {
                "topic_id": int(tid),
                "label": result.get_label(tid),
                "color": topic_colors[tid],
            }
        )

    points_json = json.dumps(points, ensure_ascii=False)
    colors_json = json.dumps(
        {str(k): v for k, v in topic_colors.items()}, ensure_ascii=False
    )
    legend_json = json.dumps(legend_entries, ensure_ascii=False)

    html = _build_html(points_json, colors_json, legend_json)
    out_path.write_text(html, encoding="utf-8")

    log.info("Saved 3-D explorer -> %s", out_path)
    print(f"  -> {out_path}")

    return out_path.resolve()


def _build_html(
    points_json: str,
    colors_json: str,
    legend_json: str,
) -> str:
    """Return the complete HTML string with embedded data and Three.js code."""
    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Issue Landscape — 3D Explorer</title>\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&display=swap" rel="stylesheet">\n'
        '<style>\n'
        'html, body {\n'
        '  margin: 0; padding: 0; overflow: hidden;\n'
        '  width: 100%; height: 100%;\n'
        '  background: #f8f8f5;\n'
        '  font-family: "Roboto Mono", monospace;\n'
        '  color: #333;\n'
        '}\n'
        'canvas { display: block; }\n'
        '#title-banner {\n'
        '  position: fixed;\n'
        '  top: 16px; left: 16px;\n'
        '  z-index: 100;\n'
        '  font-size: 24px;\n'
        '  font-weight: 400;\n'
        '  color: #1a1a1a;\n'
        '  padding: 8px 16px;\n'
        '  background: rgba(248, 248, 245, 0.9);\n'
        '  border: 1px solid #ccc;\n'
        '  backdrop-filter: blur(6px);\n'
        '}\n'
        '#title-banner small {\n'
        '  display: block;\n'
        '  font-size: 11px;\n'
        '  color: #888;\n'
        '  margin-top: 2px;\n'
        '  letter-spacing: 1px;\n'
        '  text-transform: uppercase;\n'
        '}\n'
        '#tooltip {\n'
        '  position: fixed;\n'
        '  pointer-events: none;\n'
        '  background: rgba(255, 255, 255, 0.95);\n'
        '  color: #1a1a1a;\n'
        '  padding: 8px 14px;\n'
        '  border-radius: 6px;\n'
        '  font-size: 12px;\n'
        '  font-family: "Roboto Mono", monospace;\n'
        '  max-width: 380px;\n'
        '  line-height: 1.4;\n'
        '  display: none;\n'
        '  z-index: 100;\n'
        '  border: 1px solid #ddd;\n'
        '  box-shadow: 0 2px 8px rgba(0,0,0,0.1);\n'
        '}\n'
        '#tooltip .tt-number { font-weight: 700; color: #0969da; }\n'
        '#tooltip .tt-label  { font-size: 11px; color: #888; margin-top: 3px; }\n'
        '#legend {\n'
        '  position: fixed;\n'
        '  top: 16px; right: 16px;\n'
        '  background: rgba(248, 248, 245, 0.92);\n'
        '  border: 1px solid #ddd;\n'
        '  border-radius: 4px;\n'
        '  padding: 12px 16px;\n'
        '  max-height: calc(100vh - 48px);\n'
        '  overflow-y: auto;\n'
        '  z-index: 50;\n'
        '  min-width: 160px;\n'
        '  box-shadow: 0 1px 4px rgba(0,0,0,0.08);\n'
        '  backdrop-filter: blur(6px);\n'
        '}\n'
        '#legend h3 {\n'
        '  margin: 0 0 10px 0; font-size: 11px;\n'
        '  text-transform: uppercase; letter-spacing: 1.5px;\n'
        '  color: #999;\n'
        '  font-family: "Roboto Mono", monospace;\n'
        '}\n'
        '.legend-item {\n'
        '  display: flex; align-items: center;\n'
        '  margin-bottom: 5px; font-size: 11px;\n'
        '  font-family: "Roboto Mono", monospace;\n'
        '  color: #555;\n'
        '}\n'
        '.legend-swatch {\n'
        '  width: 10px; height: 10px;\n'
        '  border-radius: 50%; margin-right: 8px;\n'
        '  flex-shrink: 0;\n'
        '}\n'
        '#legend::-webkit-scrollbar { width: 4px; }\n'
        '#legend::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.12); border-radius: 2px; }\n'
        '#view-switcher {\n'
        '  position: fixed; bottom: 20px; right: 20px; z-index: 100;\n'
        '  display: flex; gap: 8px;\n'
        '}\n'
        '#view-switcher a {\n'
        '  display: inline-block; padding: 8px 16px;\n'
        '  background: rgba(248, 248, 245, 0.92); color: #0969da;\n'
        '  border: 1px solid #d0d7de; border-radius: 8px;\n'
        '  font-size: 13px; font-weight: 500; text-decoration: none;\n'
        '  font-family: "Roboto Mono", monospace;\n'
        '  box-shadow: 0 1px 3px rgba(0,0,0,0.08);\n'
        '  transition: background 0.15s;\n'
        '}\n'
        '#view-switcher a:hover { background: #f0f6ff; }\n'
        '</style>\n'
        '<script type="importmap">\n'
        '{"imports":{"three":"https://unpkg.com/three@0.170.0/build/three.module.js",'
        '"three/addons/":"https://unpkg.com/three@0.170.0/examples/jsm/"}}\n'
        '</script>\n'
        '</head>\n'
        '<body>\n'
        '<div id="title-banner">Issue Landscape<small>3D Explorer</small></div>\n'
        '<div id="tooltip"></div>\n'
        '<div id="legend"></div>\n'
        '<div id="view-switcher">\n'
        '  <a href="landscape-2d.html">View in 2D &#x2197;</a>\n'
        '  <a href="timeline.html">Timeline &#x2197;</a>\n'
        '</div>\n'
        '\n'
        '<script type="module">\n'
        'import * as THREE from "three";\n'
        'import { OrbitControls } from "three/addons/controls/OrbitControls.js";\n'
        '\n'
        'const DATA_POINTS = ' + points_json + ';\n'
        'const TOPIC_COLORS = ' + colors_json + ';\n'
        'const LEGEND_ENTRIES = ' + legend_json + ';\n'
        '\n'
        'const scene  = new THREE.Scene();\n'
        'scene.background = new THREE.Color(0xf8f8f5);\n'
        '\n'
        'const camera = new THREE.PerspectiveCamera(\n'
        '  60, window.innerWidth / window.innerHeight, 0.01, 100\n'
        ');\n'
        'camera.position.set(1.8, 1.2, 1.8);\n'
        '\n'
        'const renderer = new THREE.WebGLRenderer({ antialias: true });\n'
        'renderer.setPixelRatio(window.devicePixelRatio);\n'
        'renderer.setSize(window.innerWidth, window.innerHeight);\n'
        'document.body.appendChild(renderer.domElement);\n'
        '\n'
        'const controls = new OrbitControls(camera, renderer.domElement);\n'
        'controls.enableDamping  = true;\n'
        'controls.dampingFactor  = 0.08;\n'
        'controls.autoRotate     = true;\n'
        'controls.autoRotateSpeed = 0.5;\n'
        'controls.minDistance    = 0.5;\n'
        'controls.maxDistance    = 8;\n'
        '\n'
        'const count = DATA_POINTS.length;\n'
        'const positions = new Float32Array(count * 3);\n'
        'const colors    = new Float32Array(count * 3);\n'
        'const sizes     = new Float32Array(count);\n'
        'const BASE_SIZE = 0.15;\n'
        '\n'
        'for (let i = 0; i < count; i++) {\n'
        '  const p = DATA_POINTS[i];\n'
        '  positions[i * 3]     = p.x;\n'
        '  positions[i * 3 + 1] = p.y;\n'
        '  positions[i * 3 + 2] = p.z;\n'
        '  const hex = TOPIC_COLORS[String(p.topic_id)] || "#999999";\n'
        '  const c = new THREE.Color(hex);\n'
        '  colors[i * 3]     = c.r;\n'
        '  colors[i * 3 + 1] = c.g;\n'
        '  colors[i * 3 + 2] = c.b;\n'
        '  sizes[i] = BASE_SIZE;\n'
        '}\n'
        '\n'
        'const geometry = new THREE.BufferGeometry();\n'
        'geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));\n'
        'geometry.setAttribute("color",    new THREE.BufferAttribute(colors, 3));\n'
        'geometry.setAttribute("size",     new THREE.BufferAttribute(sizes, 1));\n'
        '\n'
        'const material = new THREE.ShaderMaterial({\n'
        '  vertexShader: `\n'
        '    attribute float size;\n'
        '    varying vec3 vColor;\n'
        '    void main() {\n'
        '      vColor = color;\n'
        '      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);\n'
        '      gl_PointSize = size * (200.0 / -mvPosition.z);\n'
        '      gl_Position  = projectionMatrix * mvPosition;\n'
        '    }\n'
        '  `,\n'
        '  fragmentShader: `\n'
        '    varying vec3 vColor;\n'
        '    void main() {\n'
        '      float d = length(gl_PointCoord - vec2(0.5));\n'
        '      if (d > 0.5) discard;\n'
        '      float alpha = 1.0 - smoothstep(0.35, 0.5, d);\n'
        '      gl_FragColor = vec4(vColor, alpha);\n'
        '    }\n'
        '  `,\n'
        '  vertexColors: true,\n'
        '  transparent: true,\n'
        '  depthWrite: false,\n'
        '});\n'
        '\n'
        'const pointCloud = new THREE.Points(geometry, material);\n'
        'scene.add(pointCloud);\n'
        'scene.add(new THREE.AmbientLight(0xffffff, 0.3));\n'
        '\n'
        '// Subtle grid for spatial reference\n'
        'const grid = new THREE.GridHelper(3, 12, 0xdddddd, 0xeeeeee);\n'
        'grid.position.y = -1.1;\n'
        'scene.add(grid);\n'
        '\n'
        'const raycaster = new THREE.Raycaster();\n'
        'raycaster.params.Points.threshold = 0.06;\n'
        'const mouse = new THREE.Vector2();\n'
        'const tooltipEl = document.getElementById("tooltip");\n'
        'let hoveredIndex = -1;\n'
        '\n'
        'function onPointerMove(event) {\n'
        '  mouse.x =  (event.clientX / window.innerWidth)  * 2 - 1;\n'
        '  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;\n'
        '  raycaster.setFromCamera(mouse, camera);\n'
        '  const intersects = raycaster.intersectObject(pointCloud);\n'
        '  const sizeAttr = geometry.getAttribute("size");\n'
        '  if (hoveredIndex >= 0) {\n'
        '    sizeAttr.setX(hoveredIndex, BASE_SIZE);\n'
        '    sizeAttr.needsUpdate = true;\n'
        '  }\n'
        '  if (intersects.length > 0) {\n'
        '    const idx = intersects[0].index;\n'
        '    hoveredIndex = idx;\n'
        '    sizeAttr.setX(idx, BASE_SIZE * 2.5);\n'
        '    sizeAttr.needsUpdate = true;\n'
        '    const p = DATA_POINTS[idx];\n'
        '    tooltipEl.innerHTML =\n'
        '      \'<span class="tt-number">#\' + p.number + \'</span> \' +\n'
        '      p.title.replace(/</g, "&lt;") +\n'
        '      \'<div class="tt-label">\' + p.label.replace(/</g, "&lt;") + \'</div>\';\n'
        '    tooltipEl.style.display = "block";\n'
        '    tooltipEl.style.left = (event.clientX + 14) + "px";\n'
        '    tooltipEl.style.top  = (event.clientY + 14) + "px";\n'
        '    const rect = tooltipEl.getBoundingClientRect();\n'
        '    if (rect.right > window.innerWidth)\n'
        '      tooltipEl.style.left = (event.clientX - rect.width - 14) + "px";\n'
        '    if (rect.bottom > window.innerHeight)\n'
        '      tooltipEl.style.top = (event.clientY - rect.height - 14) + "px";\n'
        '    renderer.domElement.style.cursor = "pointer";\n'
        '  } else {\n'
        '    hoveredIndex = -1;\n'
        '    tooltipEl.style.display = "none";\n'
        '    renderer.domElement.style.cursor = "default";\n'
        '  }\n'
        '}\n'
        '\n'
        'function onClick(event) {\n'
        '  if (hoveredIndex >= 0) {\n'
        '    const url = DATA_POINTS[hoveredIndex].url;\n'
        '    if (url) window.open(url, "_blank", "noopener");\n'
        '  }\n'
        '}\n'
        '\n'
        'window.addEventListener("pointermove", onPointerMove, false);\n'
        'window.addEventListener("click", onClick, false);\n'
        '\n'
        'const legendEl = document.getElementById("legend");\n'
        'let legendHTML = "<h3>Topics</h3>";\n'
        'for (const entry of LEGEND_ENTRIES) {\n'
        '  legendHTML += \'<div class="legend-item">\' +\n'
        '    \'<span class="legend-swatch" style="background:\' + entry.color + \'"></span>\' +\n'
        '    entry.label.replace(/</g, "&lt;") +\n'
        '    \'</div>\';\n'
        '}\n'
        'legendEl.innerHTML = legendHTML;\n'
        '\n'
        'window.addEventListener("resize", () => {\n'
        '  camera.aspect = window.innerWidth / window.innerHeight;\n'
        '  camera.updateProjectionMatrix();\n'
        '  renderer.setSize(window.innerWidth, window.innerHeight);\n'
        '});\n'
        '\n'
        'function animate() {\n'
        '  requestAnimationFrame(animate);\n'
        '  controls.update();\n'
        '  renderer.render(scene, camera);\n'
        '}\n'
        'animate();\n'
        '</script>\n'
        '</body>\n'
        '</html>\n'
    )
