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

# ---------------------------------------------------------------------------
# Color palette -- 30 visually distinct colors for topic clusters
# ---------------------------------------------------------------------------
_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    "#e6beff", "#1abc9c", "#ff6348", "#7bed9f", "#70a1ff",
    "#5352ed", "#ff4757", "#2ed573", "#ffa502", "#3742fa",
]

_OUTLIER_COLOR = "#555555"


def _assign_topic_colors(topic_ids: set[int]) -> dict[int, str]:
    """Map each topic id to a hex color string."""
    colors: dict[int, str] = {}
    palette_idx = 0
    for tid in sorted(topic_ids):
        if tid == -1:
            colors[tid] = _OUTLIER_COLOR
        else:
            colors[tid] = _PALETTE[palette_idx % len(_PALETTE)]
            palette_idx += 1
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
    """Generate a self-contained HTML file with a Three.js 3D point cloud.

    Parameters
    ----------
    result:
        A fitted ``PipelineResult`` that **must** contain ``umap_3d``.
    issues:
        The list of issue dicts (with ``number``, ``title``, ``url`` keys).
    output_dir:
        Directory where ``landscape-3d.html`` will be written.

    Returns
    -------
    Path
        Absolute path to the generated HTML file.

    Raises
    ------
    ValueError
        If ``result.umap_3d`` is ``None``.
    """
    if result.umap_3d is None:
        raise ValueError(
            "PipelineResult.umap_3d is None. "
            "Re-run the pipeline with reduce_3d=True."
        )

    out_path = Path(output_dir) / "landscape-3d.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    coords = _normalize_coords(result.umap_3d)

    # Collect unique topic ids and assign colors
    topic_ids = set(result.topics)
    topic_colors = _assign_topic_colors(topic_ids)

    # Build per-point data
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

    # Build legend entries (sorted by topic id, outliers last)
    legend_entries: list[dict] = []
    for tid in sorted(topic_ids):
        legend_entries.append(
            {
                "topic_id": int(tid),
                "label": result.get_label(tid),
                "color": topic_colors[tid],
            }
        )

    # Serialize data for embedding
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


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def _build_html(
    points_json: str,
    colors_json: str,
    legend_json: str,
) -> str:
    """Return the complete HTML string with embedded data and Three.js code."""
    # Note: we use {{ and }} inside the JS blocks so Python's str.format /
    # f-strings don't interfere with JS braces.  The three data blobs are
    # injected via simple string concatenation.

    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Issue Landscape — 3D Explorer</title>\n'
        '<style>\n'
        'html, body {\n'
        '  margin: 0; padding: 0; overflow: hidden;\n'
        '  width: 100%; height: 100%;\n'
        '  background: #1a1a2e;\n'
        '  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;\n'
        '  color: #e0e0e0;\n'
        '}\n'
        'canvas { display: block; }\n'
        '#tooltip {\n'
        '  position: fixed;\n'
        '  pointer-events: none;\n'
        '  background: rgba(10, 10, 30, 0.92);\n'
        '  color: #f0f0f0;\n'
        '  padding: 8px 12px;\n'
        '  border-radius: 6px;\n'
        '  font-size: 13px;\n'
        '  max-width: 380px;\n'
        '  line-height: 1.4;\n'
        '  display: none;\n'
        '  z-index: 100;\n'
        '  border: 1px solid rgba(255,255,255,0.12);\n'
        '  box-shadow: 0 4px 16px rgba(0,0,0,0.5);\n'
        '}\n'
        '#tooltip .tt-number { font-weight: 700; color: #90caf9; }\n'
        '#tooltip .tt-label  { font-size: 11px; color: #aaa; margin-top: 3px; }\n'
        '#legend {\n'
        '  position: fixed;\n'
        '  top: 16px; right: 16px;\n'
        '  background: rgba(10, 10, 30, 0.85);\n'
        '  border: 1px solid rgba(255,255,255,0.1);\n'
        '  border-radius: 8px;\n'
        '  padding: 12px 16px;\n'
        '  max-height: calc(100vh - 48px);\n'
        '  overflow-y: auto;\n'
        '  z-index: 50;\n'
        '  min-width: 160px;\n'
        '  box-shadow: 0 4px 20px rgba(0,0,0,0.4);\n'
        '}\n'
        '#legend h3 {\n'
        '  margin: 0 0 10px 0; font-size: 13px;\n'
        '  text-transform: uppercase; letter-spacing: 1px;\n'
        '  color: #999;\n'
        '}\n'
        '.legend-item {\n'
        '  display: flex; align-items: center;\n'
        '  margin-bottom: 6px; font-size: 12px;\n'
        '}\n'
        '.legend-swatch {\n'
        '  width: 12px; height: 12px;\n'
        '  border-radius: 3px; margin-right: 8px;\n'
        '  flex-shrink: 0;\n'
        '}\n'
        '#legend::-webkit-scrollbar { width: 4px; }\n'
        '#legend::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 2px; }\n'
        '</style>\n'
        '<script type="importmap">\n'
        '{"imports":{"three":"https://unpkg.com/three@0.170.0/build/three.module.js",'
        '"three/addons/":"https://unpkg.com/three@0.170.0/examples/jsm/"}}\n'
        '</script>\n'
        '</head>\n'
        '<body>\n'
        '<div id="tooltip"></div>\n'
        '<div id="legend"></div>\n'
        '\n'
        '<script type="module">\n'
        'import * as THREE from "three";\n'
        'import { OrbitControls } from "three/addons/controls/OrbitControls.js";\n'
        '\n'
        '// ── Embedded data ──────────────────────────────────────────\n'
        'const DATA_POINTS = ' + points_json + ';\n'
        'const TOPIC_COLORS = ' + colors_json + ';\n'
        'const LEGEND_ENTRIES = ' + legend_json + ';\n'
        '\n'
        '// ── Scene setup ────────────────────────────────────────────\n'
        'const scene  = new THREE.Scene();\n'
        'scene.background = new THREE.Color(0x1a1a2e);\n'
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
        '// ── Controls ───────────────────────────────────────────────\n'
        'const controls = new OrbitControls(camera, renderer.domElement);\n'
        'controls.enableDamping  = true;\n'
        'controls.dampingFactor  = 0.08;\n'
        'controls.autoRotate     = true;\n'
        'controls.autoRotateSpeed = 0.5;\n'
        'controls.minDistance    = 0.5;\n'
        'controls.maxDistance    = 8;\n'
        '\n'
        '// ── Build point cloud ──────────────────────────────────────\n'
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
        '\n'
        '  const hex = TOPIC_COLORS[String(p.topic_id)] || "#555555";\n'
        '  const c = new THREE.Color(hex);\n'
        '  colors[i * 3]     = c.r;\n'
        '  colors[i * 3 + 1] = c.g;\n'
        '  colors[i * 3 + 2] = c.b;\n'
        '\n'
        '  sizes[i] = BASE_SIZE;\n'
        '}\n'
        '\n'
        'const geometry = new THREE.BufferGeometry();\n'
        'geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));\n'
        'geometry.setAttribute("color",    new THREE.BufferAttribute(colors, 3));\n'
        'geometry.setAttribute("size",     new THREE.BufferAttribute(sizes, 1));\n'
        '\n'
        '// Custom shader so we can control per-point size\n'
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
        '\n'
        '// Subtle ambient light (not strictly needed for points but nice for feel)\n'
        'scene.add(new THREE.AmbientLight(0xffffff, 0.3));\n'
        '\n'
        '// ── Raycaster / interaction ────────────────────────────────\n'
        'const raycaster = new THREE.Raycaster();\n'
        'raycaster.params.Points.threshold = 0.06;\n'
        'const mouse = new THREE.Vector2();\n'
        'const tooltipEl = document.getElementById("tooltip");\n'
        '\n'
        'let hoveredIndex = -1;\n'
        '\n'
        'function onPointerMove(event) {\n'
        '  mouse.x =  (event.clientX / window.innerWidth)  * 2 - 1;\n'
        '  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;\n'
        '\n'
        '  raycaster.setFromCamera(mouse, camera);\n'
        '  const intersects = raycaster.intersectObject(pointCloud);\n'
        '\n'
        '  const sizeAttr = geometry.getAttribute("size");\n'
        '\n'
        '  // Reset previous highlight\n'
        '  if (hoveredIndex >= 0) {\n'
        '    sizeAttr.setX(hoveredIndex, BASE_SIZE);\n'
        '    sizeAttr.needsUpdate = true;\n'
        '  }\n'
        '\n'
        '  if (intersects.length > 0) {\n'
        '    const idx = intersects[0].index;\n'
        '    hoveredIndex = idx;\n'
        '    sizeAttr.setX(idx, BASE_SIZE * 2.5);\n'
        '    sizeAttr.needsUpdate = true;\n'
        '\n'
        '    const p = DATA_POINTS[idx];\n'
        '    tooltipEl.innerHTML =\n'
        '      \'<span class="tt-number">#\' + p.number + \'</span> \' +\n'
        '      p.title.replace(/</g, "&lt;") +\n'
        '      \'<div class="tt-label">\' + p.label.replace(/</g, "&lt;") + \'</div>\';\n'
        '    tooltipEl.style.display = "block";\n'
        '    tooltipEl.style.left = (event.clientX + 14) + "px";\n'
        '    tooltipEl.style.top  = (event.clientY + 14) + "px";\n'
        '\n'
        '    // Keep tooltip on screen\n'
        '    const rect = tooltipEl.getBoundingClientRect();\n'
        '    if (rect.right > window.innerWidth) {\n'
        '      tooltipEl.style.left = (event.clientX - rect.width - 14) + "px";\n'
        '    }\n'
        '    if (rect.bottom > window.innerHeight) {\n'
        '      tooltipEl.style.top = (event.clientY - rect.height - 14) + "px";\n'
        '    }\n'
        '\n'
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
        '// ── Legend ─────────────────────────────────────────────────\n'
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
        '// ── Resize handling ────────────────────────────────────────\n'
        'window.addEventListener("resize", () => {\n'
        '  camera.aspect = window.innerWidth / window.innerHeight;\n'
        '  camera.updateProjectionMatrix();\n'
        '  renderer.setSize(window.innerWidth, window.innerHeight);\n'
        '});\n'
        '\n'
        '// ── Render loop ────────────────────────────────────────────\n'
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
