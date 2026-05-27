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

from gh_issue_landscape.pipeline import (
    OUTLIER_COLOR,
    TOPIC_PALETTE,
    PipelineResult,
)

log = logging.getLogger(__name__)


def _assign_topic_colors(topic_ids: set[int]) -> dict[int, str]:
    """Map each topic id to a hex color from the shared palette."""
    colors: dict[int, str] = {}
    idx = 0
    for tid in sorted(topic_ids):
        if tid == -1:
            colors[tid] = OUTLIER_COLOR
        else:
            colors[tid] = TOPIC_PALETTE[idx % len(TOPIC_PALETTE)]
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

    issue_details: list[dict] = []
    for i, issue in enumerate(issues):
        body = issue.get("body") or ""
        if len(body) > 600:
            body = body[:600] + "..."
        tid = result.topics[i] if i < len(result.topics) else -1
        issue_details.append(
            {
                "number": issue["number"],
                "title": issue["title"],
                "body": body,
                "url": issue.get("url", ""),
                "labels": issue.get("labels", []),
                "created_at": issue.get("created_at", ""),
                "topic": result.get_label(tid),
            }
        )

    points_json = json.dumps(points, ensure_ascii=False)
    colors_json = json.dumps(
        {str(k): v for k, v in topic_colors.items()}, ensure_ascii=False
    )
    legend_json = json.dumps(legend_entries, ensure_ascii=False)
    details_json = json.dumps(issue_details, ensure_ascii=False)

    html = _build_html(points_json, colors_json, legend_json, details_json)
    out_path.write_text(html, encoding="utf-8")

    log.info("Saved 3-D explorer -> %s", out_path)
    print(f"  -> {out_path}")

    return out_path.resolve()


def _build_html(
    points_json: str,
    colors_json: str,
    legend_json: str,
    details_json: str,
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
        '#issue-card-backdrop {\n'
        '  display:none; position:fixed; top:0;left:0; width:100%;height:100%;\n'
        '  background:rgba(0,0,0,0.35); z-index:9998;\n'
        '}\n'
        '#issue-card {\n'
        '  display:none; position:fixed; top:50%;left:50%;\n'
        '  transform:translate(-50%,-50%); max-width:520px; width:90%;\n'
        '  max-height:80vh; overflow-y:auto; background:#fff;\n'
        '  border-radius:12px; box-shadow:0 8px 32px rgba(0,0,0,0.28);\n'
        '  z-index:9999; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;\n'
        '  color:#1a1a1a;\n'
        '}\n'
        '#card-header { display:flex; align-items:flex-start; justify-content:space-between; padding:20px 20px 12px; border-bottom:1px solid #e8e8e8; cursor:grab; }\n'
        '#card-header:active { cursor:grabbing; }\n'
        'body.card-dragging { user-select:none; -webkit-user-select:none; }\n'
        '#card-title { font-size:16px; font-weight:600; line-height:1.4; margin-right:12px; word-break:break-word; }\n'
        '#card-number { font-size:14px; color:#656d76; font-weight:400; }\n'
        '#card-close { background:none; border:none; font-size:22px; color:#656d76; cursor:pointer; padding:0 4px; line-height:1; flex-shrink:0; border-radius:4px; }\n'
        '#card-close:hover { background:#f0f0f0; color:#1a1a1a; }\n'
        '#card-topic { padding:8px 20px; font-size:12px; color:#656d76; background:#f6f8fa; }\n'
        '#card-body { padding:16px 20px; font-size:14px; line-height:1.6; color:#333; white-space:normal; word-break:break-word; }\n'
        '#card-body h1,#card-body h2,#card-body h3,#card-body h4,#card-body h5,#card-body h6 { margin:8px 0 4px 0; font-weight:600; line-height:1.3; }\n'
        '#card-body h1 { font-size:1.15em; } #card-body h2 { font-size:1.1em; }\n'
        '#card-body h3,#card-body h4,#card-body h5,#card-body h6 { font-size:1em; }\n'
        '#card-body code { background:#f0f0f0; padding:1px 5px; border-radius:3px; font-size:0.9em; font-family:"SFMono-Regular",Consolas,monospace; }\n'
        '#card-body ul,#card-body ol { margin:4px 0; padding-left:20px; }\n'
        '#card-body a { color:#0969da; text-decoration:underline; }\n'
        '#card-body p { margin:4px 0; }\n'
        '#card-labels { padding:4px 20px 12px; display:flex; flex-wrap:wrap; gap:6px; }\n'
        '.card-label-pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:500; background:#ddf4ff; color:#0969da; border:1px solid #b6e3ff; white-space:nowrap; }\n'
        '#card-footer { display:flex; align-items:center; justify-content:space-between; padding:12px 20px; border-top:1px solid #e8e8e8; font-size:13px; }\n'
        '#card-date { color:#656d76; }\n'
        '#card-link { color:#0969da; text-decoration:underline; font-weight:500; }\n'
        '#card-link:hover { color:#0550ae; }\n'
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
        '<div id="issue-card-backdrop"></div>\n'
        '<div id="issue-card" role="dialog" aria-modal="true">\n'
        '  <div id="card-header"><span id="card-title"></span><button id="card-close" aria-label="Close">&times;</button></div>\n'
        '  <div id="card-topic"></div>\n'
        '  <div id="card-body"></div>\n'
        '  <div id="card-labels"></div>\n'
        '  <div id="card-footer"><span id="card-date"></span><a id="card-link" href="#" target="_blank" rel="noopener">Open on GitHub &#8594;</a></div>\n'
        '</div>\n'
        '<div id="view-switcher">\n'
        '  <a href="landscape-2d.html">View in 2D &#x2197;</a>\n'
        '  <a href="timeline.html">Timeline &#x2197;</a>\n'
        '  <a href="method.html">Method &#x2197;</a>\n'
        '</div>\n'
        '\n'
        '<script type="module">\n'
        'import * as THREE from "three";\n'
        'import { OrbitControls } from "three/addons/controls/OrbitControls.js";\n'
        '\n'
        'const DATA_POINTS = ' + points_json + ';\n'
        'const TOPIC_COLORS = ' + colors_json + ';\n'
        'const LEGEND_ENTRIES = ' + legend_json + ';\n'
        'const ISSUE_DETAILS = ' + details_json + ';\n'
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
        '// ── Detail card ────────────────────────────────────────────\n'
        'const card = document.getElementById("issue-card");\n'
        'const backdrop = document.getElementById("issue-card-backdrop");\n'
        'const cardTitle = document.getElementById("card-title");\n'
        'const cardTopic = document.getElementById("card-topic");\n'
        'const cardBody = document.getElementById("card-body");\n'
        'const cardLabels = document.getElementById("card-labels");\n'
        'const cardDate = document.getElementById("card-date");\n'
        'const cardLink = document.getElementById("card-link");\n'
        'const cardClose = document.getElementById("card-close");\n'
        'const cardHeader = document.getElementById("card-header");\n'
        '\n'
        '// --- Drag support ---\n'
        'let isDragging = false, dragOffsetX = 0, dragOffsetY = 0;\n'
        'cardHeader.addEventListener("mousedown", function(e) {\n'
        '  if (e.button !== 0 || e.target === cardClose) return;\n'
        '  isDragging = true;\n'
        '  document.body.classList.add("card-dragging");\n'
        '  const rect = card.getBoundingClientRect();\n'
        '  card.style.transform = "none";\n'
        '  card.style.top = rect.top + "px";\n'
        '  card.style.left = rect.left + "px";\n'
        '  dragOffsetX = e.clientX - rect.left;\n'
        '  dragOffsetY = e.clientY - rect.top;\n'
        '  e.preventDefault();\n'
        '});\n'
        'document.addEventListener("mousemove", function(e) {\n'
        '  if (!isDragging) return;\n'
        '  card.style.left = (e.clientX - dragOffsetX) + "px";\n'
        '  card.style.top = (e.clientY - dragOffsetY) + "px";\n'
        '});\n'
        'document.addEventListener("mouseup", function() {\n'
        '  if (isDragging) { isDragging = false; document.body.classList.remove("card-dragging"); }\n'
        '});\n'
        '\n'
        '// --- Lightweight markdown renderer ---\n'
        'function renderMarkdown(text) {\n'
        '  if (!text) return "(no description)";\n'
        '  let h = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");\n'
        '  h = h.replace(/^######\\s+(.+)$/gm, "<h6>$1</h6>");\n'
        '  h = h.replace(/^#####\\s+(.+)$/gm, "<h5>$1</h5>");\n'
        '  h = h.replace(/^####\\s+(.+)$/gm, "<h4>$1</h4>");\n'
        '  h = h.replace(/^###\\s+(.+)$/gm, "<h3>$1</h3>");\n'
        '  h = h.replace(/^##\\s+(.+)$/gm, "<h2>$1</h2>");\n'
        '  h = h.replace(/^#\\s+(.+)$/gm, "<h1>$1</h1>");\n'
        '  h = h.replace(/\\*\\*(.+?)\\*\\*/g, "<strong>$1</strong>");\n'
        '  h = h.replace(/__(.+?)__/g, "<strong>$1</strong>");\n'
        '  h = h.replace(/\\*(.+?)\\*/g, "<em>$1</em>");\n'
        '  h = h.replace(/`([^`]+)`/g, "<code>$1</code>");\n'
        '  h = h.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, \'<a href="$2" target="_blank" rel="noopener">$1</a>\');\n'
        '  h = h.replace(/^[\\-\\*]\\s+(.+)$/gm, "<li>$1</li>");\n'
        '  h = h.replace(/((?:<li>.*<\\/li>\\n?)+)/g, "<ul>$1</ul>");\n'
        '  h = h.replace(/\\n\\n+/g, "</p><p>");\n'
        '  h = h.replace(/\\n/g, "<br>");\n'
        '  h = h.replace(/<\\/li><br>/g, "<\\/li>");\n'
        '  h = h.replace(/<br><li>/g, "<li>");\n'
        '  h = h.replace(/<br>(<h[1-6]>)/g, "$1");\n'
        '  h = h.replace(/<\\/h[1-6]><br>/g, function(m) { return m.replace("<br>", ""); });\n'
        '  return "<p>" + h.replace(/<p>\\s*<\\/p>/g, "") + "</p>";\n'
        '}\n'
        '\n'
        'function showCard(issue) {\n'
        '  cardTitle.innerHTML = \'<span id="card-number">#\' + issue.number + \'</span> \' +\n'
        '    issue.title.replace(/</g, "&lt;").replace(/>/g, "&gt;");\n'
        '  cardTopic.textContent = issue.topic || "";\n'
        '  cardBody.innerHTML = renderMarkdown(issue.body);\n'
        '  cardLabels.innerHTML = "";\n'
        '  (issue.labels || []).forEach(function(label) {\n'
        '    const pill = document.createElement("span");\n'
        '    pill.className = "card-label-pill";\n'
        '    pill.textContent = label;\n'
        '    cardLabels.appendChild(pill);\n'
        '  });\n'
        '  cardDate.textContent = issue.created_at ? issue.created_at.substring(0, 10) : "";\n'
        '  cardLink.href = issue.url || "#";\n'
        '  backdrop.style.display = "block";\n'
        '  card.style.display = "block";\n'
        '}\n'
        '\n'
        'function hideCard() {\n'
        '  card.style.display = "none";\n'
        '  backdrop.style.display = "none";\n'
        '  card.style.top = "50%";\n'
        '  card.style.left = "50%";\n'
        '  card.style.transform = "translate(-50%,-50%)";\n'
        '}\n'
        '\n'
        'cardClose.addEventListener("click", hideCard);\n'
        'backdrop.addEventListener("click", hideCard);\n'
        'document.addEventListener("keydown", function(e) { if (e.key === "Escape") hideCard(); });\n'
        '\n'
        'function onClick(event) {\n'
        '  if (hoveredIndex >= 0 && ISSUE_DETAILS[hoveredIndex]) {\n'
        '    showCard(ISSUE_DETAILS[hoveredIndex]);\n'
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
