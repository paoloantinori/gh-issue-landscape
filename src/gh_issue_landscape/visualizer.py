"""2D interactive landscape visualization and CLI summary report."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from gh_issue_landscape.pipeline import PipelineResult, _get_keywords

log = logging.getLogger(__name__)


def _build_label_array(result: PipelineResult) -> np.ndarray:
    """Map each document to its topic label string."""
    return np.array(
        [result.get_label(tid) for tid in result.topics],
        dtype=object,
    )


def _build_hover_texts(issues: list[dict]) -> list[str]:
    return [f"#{issue['number']}: {issue['title']}" for issue in issues]


def _build_topic_summaries(
    result: PipelineResult,
    issues: list[dict],
) -> list[dict]:
    """Build sorted per-topic summaries (excluding outliers)."""
    topic_to_docs = result.topic_to_doc_indices()

    summaries: list[dict] = []
    for tid in sorted(topic_to_docs.keys()):
        if tid == -1:
            continue

        doc_indices = topic_to_docs.get(tid, [])
        keywords = _get_keywords(result.topic_model, tid, limit=5) if result.topic_model else []
        summaries.append(
            {
                "topic_id": tid,
                "label": result.get_label(tid),
                "count": len(doc_indices),
                "keywords": keywords,
                "top_issues": [
                    {"number": issues[i]["number"], "title": issues[i]["title"]}
                    for i in doc_indices[:3]
                    if i < len(issues)
                ],
            }
        )

    summaries.sort(key=lambda s: s["count"], reverse=True)
    return summaries


def _build_issue_data_json(
    result: PipelineResult,
    issues: list[dict],
) -> str:
    """Build a JSON array of issue metadata for the detail card overlay."""
    records = []
    for i, issue in enumerate(issues):
        body = issue.get("body") or ""
        if len(body) > 600:
            body = body[:600] + "..."
        tid = result.topics[i] if i < len(result.topics) else -1
        records.append(
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
    return json.dumps(records, ensure_ascii=False)


_DETAIL_CARD_CSS = """\
<style id="issue-detail-card-styles">
  #view-switcher {
    position: fixed;
    bottom: 20px; right: 20px;
    z-index: 9000;
    display: flex; gap: 8px;
  }
  #view-switcher a {
    display: inline-block;
    padding: 8px 16px;
    background: #ffffff;
    color: #0969da;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
    font-weight: 500;
    text-decoration: none;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    transition: background 0.15s, box-shadow 0.15s;
  }
  #view-switcher a:hover {
    background: #f0f6ff;
    box-shadow: 0 2px 6px rgba(0,0,0,0.18);
  }
  #issue-card-backdrop {
    display: none;
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(0, 0, 0, 0.35);
    z-index: 9998;
  }
  #issue-card {
    display: none;
    position: fixed;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    max-width: 520px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.28);
    z-index: 9999;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1a1a1a;
    padding: 0;
  }
  #card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    padding: 20px 20px 12px 20px;
    border-bottom: 1px solid #e8e8e8;
    cursor: grab;
  }
  #card-header:active { cursor: grabbing; }
  body.card-dragging { user-select: none; -webkit-user-select: none; }
  #card-title {
    font-size: 16px;
    font-weight: 600;
    line-height: 1.4;
    margin-right: 12px;
    word-break: break-word;
  }
  #card-number {
    font-size: 14px;
    color: #656d76;
    font-weight: 400;
  }
  #card-close {
    background: none;
    border: none;
    font-size: 22px;
    color: #656d76;
    cursor: pointer;
    padding: 0 4px;
    line-height: 1;
    flex-shrink: 0;
    border-radius: 4px;
    transition: background 0.15s, color 0.15s;
  }
  #card-close:hover {
    background: #f0f0f0;
    color: #1a1a1a;
  }
  #card-topic {
    padding: 8px 20px;
    font-size: 12px;
    color: #656d76;
    background: #f6f8fa;
  }
  #card-body {
    padding: 16px 20px;
    font-size: 14px;
    line-height: 1.6;
    color: #333;
    white-space: normal;
    word-break: break-word;
  }
  #card-body h1, #card-body h2, #card-body h3,
  #card-body h4, #card-body h5, #card-body h6 {
    margin: 8px 0 4px 0; font-weight: 600; line-height: 1.3;
  }
  #card-body h1 { font-size: 1.15em; }
  #card-body h2 { font-size: 1.1em; }
  #card-body h3, #card-body h4, #card-body h5, #card-body h6 { font-size: 1em; }
  #card-body code {
    background: #f0f0f0; padding: 1px 5px; border-radius: 3px;
    font-size: 0.9em; font-family: "SFMono-Regular", Consolas, monospace;
  }
  #card-body ul, #card-body ol { margin: 4px 0; padding-left: 20px; }
  #card-body a { color: #0969da; text-decoration: underline; }
  #card-body p { margin: 4px 0; }
  #card-labels {
    padding: 4px 20px 12px 20px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .card-label-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 500;
    background: #ddf4ff;
    color: #0969da;
    border: 1px solid #b6e3ff;
    white-space: nowrap;
  }
  #card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    border-top: 1px solid #e8e8e8;
    font-size: 13px;
  }
  #card-date {
    color: #656d76;
  }
  #card-link {
    color: #0969da;
    text-decoration: underline;
    font-weight: 500;
  }
  #card-link:hover {
    color: #0550ae;
  }
</style>
"""

_DETAIL_CARD_HTML = """\
<div id="view-switcher">
  <a href="landscape-3d.html">View in 3D &#x2197;</a>
  <a href="timeline.html">Timeline &#x2197;</a>
  <a href="method.html">Method &#x2197;</a>
</div>
<div id="issue-card-backdrop"></div>
<div id="issue-card" role="dialog" aria-modal="true">
  <div id="card-header">
    <span id="card-title"></span>
    <button id="card-close" aria-label="Close">&times;</button>
  </div>
  <div id="card-topic"></div>
  <div id="card-body"></div>
  <div id="card-labels"></div>
  <div id="card-footer">
    <span id="card-date"></span>
    <a id="card-link" href="#" target="_blank" rel="noopener">Open on GitHub &#8594;</a>
  </div>
</div>
"""

_DETAIL_CARD_JS_TEMPLATE = """\
<script id="issue-detail-card-script">
(function() {
  var ISSUE_DATA = %s;
  var issueByIndex = {};
  ISSUE_DATA.forEach(function(d, i) { issueByIndex[i] = d; });

  var card = document.getElementById('issue-card');
  var backdrop = document.getElementById('issue-card-backdrop');
  var cardTitle = document.getElementById('card-title');
  var cardTopic = document.getElementById('card-topic');
  var cardBody = document.getElementById('card-body');
  var cardLabels = document.getElementById('card-labels');
  var cardDate = document.getElementById('card-date');
  var cardLink = document.getElementById('card-link');
  var cardClose = document.getElementById('card-close');
  var cardHeader = document.getElementById('card-header');

  // --- Drag support ---
  var isDragging = false, dragOffsetX = 0, dragOffsetY = 0;

  cardHeader.addEventListener('mousedown', function(e) {
    if (e.button !== 0 || e.target === cardClose) return;
    isDragging = true;
    document.body.classList.add('card-dragging');
    var rect = card.getBoundingClientRect();
    card.style.transform = 'none';
    card.style.top = rect.top + 'px';
    card.style.left = rect.left + 'px';
    dragOffsetX = e.clientX - rect.left;
    dragOffsetY = e.clientY - rect.top;
    e.preventDefault();
  });

  document.addEventListener('mousemove', function(e) {
    if (!isDragging) return;
    card.style.left = (e.clientX - dragOffsetX) + 'px';
    card.style.top = (e.clientY - dragOffsetY) + 'px';
  });

  document.addEventListener('mouseup', function() {
    if (isDragging) {
      isDragging = false;
      document.body.classList.remove('card-dragging');
    }
  });

  // --- Lightweight markdown renderer ---
  function renderMarkdown(text) {
    if (!text) return '(no description)';
    var h = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    h = h.replace(/^######\\s+(.+)$/gm, '<h6>$1</h6>');
    h = h.replace(/^#####\\s+(.+)$/gm, '<h5>$1</h5>');
    h = h.replace(/^####\\s+(.+)$/gm, '<h4>$1</h4>');
    h = h.replace(/^###\\s+(.+)$/gm, '<h3>$1</h3>');
    h = h.replace(/^##\\s+(.+)$/gm, '<h2>$1</h2>');
    h = h.replace(/^#\\s+(.+)$/gm, '<h1>$1</h1>');
    h = h.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
    h = h.replace(/__(.+?)__/g, '<strong>$1</strong>');
    h = h.replace(/\\*(.+?)\\*/g, '<em>$1</em>');
    h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
    h = h.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    h = h.replace(/^[\\-\\*]\\s+(.+)$/gm, '<li>$1</li>');
    h = h.replace(/((?:<li>.*<\\/li>\\n?)+)/g, '<ul>$1</ul>');
    h = h.replace(/\\n\\n+/g, '</p><p>');
    h = h.replace(/\\n/g, '<br>');
    h = h.replace(/<\\/li><br>/g, '<\\/li>');
    h = h.replace(/<br><li>/g, '<li>');
    h = h.replace(/<br>(<h[1-6]>)/g, '$1');
    h = h.replace(/<\\/h[1-6]><br>/g, function(m) { return m.replace('<br>', ''); });
    return '<p>' + h.replace(/<p>\\s*<\\/p>/g, '') + '</p>';
  }

  function showCard(issue) {
    cardTitle.innerHTML = '<span id="card-number">#' + issue.number + '</span> ' +
      issue.title.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    cardTopic.textContent = issue.topic || '';
    cardBody.innerHTML = renderMarkdown(issue.body);
    cardLabels.innerHTML = '';
    (issue.labels || []).forEach(function(label) {
      var pill = document.createElement('span');
      pill.className = 'card-label-pill';
      pill.textContent = label;
      cardLabels.appendChild(pill);
    });
    cardDate.textContent = issue.created_at ? issue.created_at.substring(0, 10) : '';
    cardLink.href = issue.url || '#';
    backdrop.style.display = 'block';
    card.style.display = 'block';
  }

  function hideCard() {
    card.style.display = 'none';
    backdrop.style.display = 'none';
    card.style.top = '50%%';
    card.style.left = '50%%';
    card.style.transform = 'translate(-50%%, -50%%)';
  }

  cardClose.addEventListener('click', hideCard);
  backdrop.addEventListener('click', hideCard);
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') hideCard();
  });

  // Hook into deck.gl's onClick via the DataMap instance.
  // DataMapPlot creates a global `datamap` variable with a `.deckgl` DeckGL instance.
  function hookDeckClick() {
    if (window.datamap && window.datamap.deckgl) {
      window.datamap.deckgl.setProps({
        controller: {scrollZoom: {speed: 0.05, smooth: true}},
        onClick: function(info) {
          if (info && info.picked && info.index != null) {
            var issue = issueByIndex[info.index];
            if (issue) { showCard(issue); return true; }
          }
        }
      });
      // Remove CSS transition on canvas that causes sluggish repaints
      var canvas = document.querySelector('canvas');
      if (canvas) canvas.style.transition = 'none';
    }
  }

  // Retry until DataMap is initialized (data loads asynchronously)
  var attempts = 0;
  var timer = setInterval(function() {
    attempts++;
    if (window.datamap && window.datamap.deckgl) {
      clearInterval(timer);
      hookDeckClick();
    } else if (attempts > 30) {
      clearInterval(timer);
    }
  }, 500);
})();
</script>
"""


def _inject_detail_card(
    html_path: Path,
    result: PipelineResult,
    issues: list[dict],
) -> None:
    """Post-process DataMapPlot HTML to inject a click-to-detail card overlay."""
    html = html_path.read_text(encoding="utf-8")

    issue_json = _build_issue_data_json(result, issues)
    js_block = _DETAIL_CARD_JS_TEMPLATE % issue_json

    injection = _DETAIL_CARD_CSS + _DETAIL_CARD_HTML + js_block

    # Inject before the closing </html> tag
    if "</html>" in html:
        html = html.replace("</html>", injection + "\n</html>")
    else:
        html += injection

    html_path.write_text(html, encoding="utf-8")
    log.info("Injected detail-card overlay into %s", html_path)


def generate_2d_map(
    result: PipelineResult,
    issues: list[dict],
    output_dir: str = "./output",
) -> Path:
    """Generate a self-contained interactive HTML visualization."""
    import datamapplot

    out_path = Path(output_dir) / "landscape-2d.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = _build_label_array(result)
    hover_texts = _build_hover_texts(issues)

    fig = datamapplot.create_interactive_plot(
        result.umap_2d,
        labels,
        hover_text=hover_texts,
        noise_label="Uncategorized",
        noise_color="#999999",
        label_color_map=result.label_color_map(),
        inline_data=True,
        font_family="Roboto Mono",
        title="Issue Landscape",
        darkmode=False,
        color_label_text=True,
    )

    fig.save(str(out_path))
    log.info("Saved interactive 2-D map → %s", out_path)

    # Post-process: inject click-to-detail card
    _inject_detail_card(out_path, result, issues)

    print(f"  -> {out_path}")

    return out_path.resolve()


def print_cli_summary(
    result: PipelineResult,
    issues: list[dict],
) -> None:
    """Print a formatted topic landscape summary to stdout."""
    total_issues = len(issues)
    outlier_count = sum(1 for t in result.topics if t == -1)
    summaries = _build_topic_summaries(result, issues)

    print()
    print("=" * 60)
    print("  Issue Landscape Summary")
    print("=" * 60)
    print()
    print(f"  Total issues:  {total_issues}")
    print(f"  Topics found:  {len(summaries)}")
    print(f"  Outliers:      {outlier_count}")
    print()

    for i, s in enumerate(summaries, start=1):
        print("-" * 60)
        print(f"  [{i}] {s['label']}  ({s['count']} issues)")
        print()

        if s["keywords"]:
            print(f"      Keywords:  {', '.join(s['keywords'])}")

        if s["top_issues"]:
            print("      Top issues:")
            for issue in s["top_issues"]:
                title = issue["title"]
                if len(title) > 70:
                    title = title[:67] + "..."
                print(f"        - #{issue['number']}: {title}")

        print()

    print("-" * 60)
    print(f"  Uncategorized (outliers): {outlier_count} issues")
    print("=" * 60)
    print()


_TIMELINE_INJECTION = """\
<style>
  #view-switcher {
    position: fixed; bottom: 20px; right: 20px; z-index: 9000;
    display: flex; gap: 8px;
  }
  #view-switcher a {
    display: inline-block; padding: 8px 16px;
    background: #ffffff; color: #0969da;
    border: 1px solid #d0d7de; border-radius: 8px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px; font-weight: 500; text-decoration: none;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    transition: background 0.15s, box-shadow 0.15s;
  }
  #view-switcher a:hover {
    background: #f0f6ff; box-shadow: 0 2px 6px rgba(0,0,0,0.18);
  }
  #timeline-tips {
    position: fixed; top: 16px; right: 16px; z-index: 9000;
    background: rgba(255,255,255,0.92); border: 1px solid #d0d7de;
    border-radius: 10px; padding: 14px 18px; max-width: 220px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 12px; color: #1a1a1a; line-height: 1.5;
    box-shadow: 0 2px 8px rgba(0,0,0,0.10);
  }
  #timeline-tips h4 { margin: 0 0 8px 0; font-size: 13px; }
  #timeline-tips ul { margin: 0; padding-left: 16px; }
  #timeline-tips li { margin-bottom: 4px; }
</style>
<div id="view-switcher">
  <a href="landscape-2d.html">View in 2D &#x2197;</a>
  <a href="landscape-3d.html">View in 3D &#x2197;</a>
  <a href="method.html">Method &#x2197;</a>
</div>
<div id="timeline-tips">
  <h4>How to use</h4>
  <ul>
    <li><b>Zoom:</b> drag to select a date range</li>
    <li><b>Reset:</b> double-click the chart</li>
    <li><b>Filter:</b> click legend items to show/hide topics</li>
    <li><b>Details:</b> hover over a data point</li>
  </ul>
</div>
"""


def generate_timeline(
    result: PipelineResult,
    issues: list[dict],
    output_dir: str = "./output",
) -> Path:
    """Generate a self-contained interactive timeline of topic evolution.

    Groups issues by calendar month and topic, producing frequency curves
    that show when each topic was most active.

    The resulting Plotly line chart is saved as a standalone HTML file at
    ``{output_dir}/timeline.html``.
    """
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio

    out_path = Path(output_dir) / "timeline.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()

    records: list[dict] = []
    for issue, tid in zip(issues, result.topics):
        if tid == -1:
            continue
        created = pd.Timestamp(issue["created_at"])
        records.append(
            {
                "month": created.to_period("M").to_timestamp(),
                "topic": tid,
            }
        )

    if records:
        df = pd.DataFrame(records)
        counts = (
            df.groupby(["month", "topic"])
            .size()
            .reset_index(name="count")
        )

        for tid in sorted(counts["topic"].unique()):
            subset = counts[counts["topic"] == tid].sort_values("month")
            label = result.get_label(int(tid))
            fig.add_trace(
                go.Scatter(
                    x=subset["month"],
                    y=subset["count"],
                    mode="lines+markers",
                    name=label,
                    hovertemplate=(
                        "<b>%{meta}</b><br>"
                        "Date: %{x|%Y-%m}<br>"
                        "Count: %{y}<extra></extra>"
                    ),
                    meta=[label] * len(subset),
                )
            )

    fig.update_layout(
        title="Topic Evolution Over Time",
        xaxis_title="Date",
        yaxis_title="Issues per Month",
        hovermode="x unified",
        legend_title="Topic",
        template="plotly_white",
    )

    html = pio.to_html(fig, full_html=True, include_plotlyjs=True)

    if "</body>" in html:
        html = html.replace("</body>", _TIMELINE_INJECTION + "\n</body>")
    else:
        html += _TIMELINE_INJECTION

    out_path.write_text(html, encoding="utf-8")
    log.info("Saved topic timeline → %s", out_path)
    print(f"  -> {out_path}")

    return out_path.resolve()


def generate_method_page(output_dir: str = "./output") -> Path:
    """Generate a static HTML page explaining the analysis methodology."""
    out_path = Path(output_dir) / "method.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Issue Landscape — Method</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1a1a1a; background: #f8f8f5;
    line-height: 1.7; padding: 0 20px 100px;
  }
  .container { max-width: 740px; margin: 0 auto; padding-top: 48px; }
  h1 { font-size: 28px; margin-bottom: 8px; }
  .subtitle { color: #656d76; font-size: 15px; margin-bottom: 36px; }
  h2 {
    font-size: 18px; margin: 32px 0 12px 0;
    padding-bottom: 6px; border-bottom: 1px solid #e0e0e0;
  }
  p { margin-bottom: 14px; font-size: 15px; }
  .step {
    display: flex; gap: 16px; margin-bottom: 20px;
    padding: 16px; background: #fff; border-radius: 10px;
    border: 1px solid #e8e8e8;
  }
  .step-num {
    flex-shrink: 0; width: 32px; height: 32px;
    background: #0969da; color: #fff; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 14px;
  }
  .step-body h3 { font-size: 15px; margin-bottom: 4px; }
  .step-body p { margin-bottom: 0; font-size: 14px; color: #333; }
  code {
    background: #f0f0f0; padding: 1px 5px; border-radius: 3px;
    font-size: 0.9em; font-family: "SFMono-Regular", Consolas, monospace;
  }
  .refs { list-style: none; padding: 0; }
  .refs li {
    padding: 8px 0; border-bottom: 1px solid #f0f0f0; font-size: 14px;
  }
  .refs a { color: #0969da; text-decoration: none; }
  .refs a:hover { text-decoration: underline; }
  .refs .desc { color: #656d76; font-size: 13px; }
  #view-switcher {
    position: fixed; bottom: 20px; right: 20px; z-index: 9000;
    display: flex; gap: 8px;
  }
  #view-switcher a {
    display: inline-block; padding: 8px 16px;
    background: #ffffff; color: #0969da;
    border: 1px solid #d0d7de; border-radius: 8px;
    font-size: 13px; font-weight: 500; text-decoration: none;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    transition: background 0.15s, box-shadow 0.15s;
  }
  #view-switcher a:hover {
    background: #f0f6ff; box-shadow: 0 2px 6px rgba(0,0,0,0.18);
  }
</style>
</head>
<body>
<div class="container">
  <h1>How the Issue Landscape Works</h1>
  <p class="subtitle">
    From raw GitHub issues to an interactive semantic map — a six-stage pipeline
    combining NLP embeddings, manifold learning, and density-based clustering.
  </p>

  <h2>Pipeline</h2>

  <div class="step">
    <div class="step-num">1</div>
    <div class="step-body">
      <h3>Data Collection</h3>
      <p>Open issues are fetched from the GitHub API via the <code>gh</code> CLI.
         Each issue's title and body are concatenated into a single document for
         embedding.</p>
    </div>
  </div>

  <div class="step">
    <div class="step-num">2</div>
    <div class="step-body">
      <h3>Text Embedding</h3>
      <p>Each document is converted into a 384-dimensional vector using
         <b>all-MiniLM-L6-v2</b>, a sentence-transformer model trained on
         over 1 billion sentence pairs. These vectors encode semantic meaning:
         issues about similar topics land near each other in vector space.
         Alternative backends (OpenAI, local API servers) are also supported.</p>
    </div>
  </div>

  <div class="step">
    <div class="step-num">3</div>
    <div class="step-body">
      <h3>Dimensionality Reduction — UMAP</h3>
      <p><b>Uniform Manifold Approximation and Projection</b> reduces the
         384-dim embeddings down to 2D (or 3D) coordinates for visualization.
         UMAP preserves both local neighborhoods and global structure better
         than alternatives like t-SNE or PCA.
         Configuration: <code>n_neighbors=15</code>, <code>min_dist=0.1</code>,
         <code>metric=cosine</code>.</p>
    </div>
  </div>

  <div class="step">
    <div class="step-num">4</div>
    <div class="step-body">
      <h3>Clustering — HDBSCAN</h3>
      <p><b>Hierarchical Density-Based Spatial Clustering of Applications with
         Noise</b> discovers topic clusters without requiring a pre-defined
         number of topics. It identifies groups of varying density and shape,
         and naturally marks isolated points as outliers ("Uncategorized").
         Configuration: <code>min_cluster_size=5</code>,
         <code>min_samples=3</code>.</p>
    </div>
  </div>

  <div class="step">
    <div class="step-num">5</div>
    <div class="step-body">
      <h3>Topic Labeling — c-TF-IDF</h3>
      <p><b>Class-based Term Frequency–Inverse Document Frequency</b>
         extracts the most distinctive keywords for each cluster. Terms that
         are frequent within a topic but rare across others are ranked highest.
         The top 3 keywords (after stopword filtering) become the topic label,
         e.g. "Schema / Type / Compatibility".</p>
    </div>
  </div>

  <div class="step">
    <div class="step-num">6</div>
    <div class="step-body">
      <h3>Visualization</h3>
      <p>The final output is rendered in three complementary views:
         the <b>2D Semantic Map</b> (DataMapPlot with deck.gl) for spatial
         exploration, the <b>3D Explorer</b> (Three.js) for immersive
         navigation, and the <b>Timeline</b> (Plotly) showing topic evolution
         over time.</p>
    </div>
  </div>

  <h2>Orchestration — BERTopic</h2>
  <p>Steps 3–5 are orchestrated by <b>BERTopic</b>, a modular topic modeling
     framework that integrates UMAP, HDBSCAN, and c-TF-IDF into a unified
     pipeline. Pre-computed embeddings from step 2 are passed in, so the
     embedding model is independent of the topic modeling process.</p>

  <h2>References</h2>
  <ul class="refs">
    <li>
      <a href="https://github.com/MaartenGr/BERTopic" target="_blank">BERTopic</a>
      <span class="desc"> — Maarten Grootendorst. Modular topic modeling with BERT embeddings.</span>
    </li>
    <li>
      <a href="https://arxiv.org/abs/1802.03426" target="_blank">UMAP paper</a>
      <span class="desc"> — McInnes, Healy, Melville. Uniform Manifold Approximation and Projection (2018).</span>
    </li>
    <li>
      <a href="https://github.com/scikit-learn-contrib/hdbscan" target="_blank">HDBSCAN</a>
      <span class="desc"> — Campello, Moulavi, Sander. Density-based clustering.</span>
    </li>
    <li>
      <a href="https://www.sbert.net/" target="_blank">Sentence-Transformers</a>
      <span class="desc"> — Reimers & Gurevych. Sentence embeddings using Siamese BERT networks.</span>
    </li>
    <li>
      <a href="https://github.com/TutteInstitute/datamapplot" target="_blank">DataMapPlot</a>
      <span class="desc"> — Leland McInnes. Interactive data cartography visualization.</span>
    </li>
    <li>
      <a href="https://github.com/stevenfazzio/semantic-github-map" target="_blank">Semantic Map of GitHub</a>
      <span class="desc"> — Steven Fazzio. The project that inspired this approach.</span>
    </li>
  </ul>
</div>

<div id="view-switcher">
  <a href="landscape-2d.html">View in 2D &#x2197;</a>
  <a href="landscape-3d.html">View in 3D &#x2197;</a>
  <a href="timeline.html">Timeline &#x2197;</a>
</div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    log.info("Saved method page → %s", out_path)
    print(f"  -> {out_path}")

    return out_path.resolve()
