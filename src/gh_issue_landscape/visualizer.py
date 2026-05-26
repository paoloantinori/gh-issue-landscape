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
    for _, row in result.topic_info.iterrows():
        from gh_issue_landscape.pipeline import _to_int
        tid = _to_int(row["Topic"])
        if tid == -1:
            continue

        doc_indices = topic_to_docs.get(tid, [])[:3]
        summaries.append(
            {
                "topic_id": tid,
                "label": result.get_label(tid),
                "count": _to_int(row.get("Count", len(topic_to_docs.get(tid, [])))),
                "keywords": _get_keywords(result.topic_model, tid, limit=5),
                "top_issues": [
                    {"number": issues[i]["number"], "title": issues[i]["title"]}
                    for i in doc_indices
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
        if len(body) > 200:
            body = body[:200] + "..."
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
  }
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
    white-space: pre-wrap;
    word-break: break-word;
  }
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

  function showCard(issue) {
    cardTitle.innerHTML = '<span id="card-number">#' + issue.number + '</span> ' +
      issue.title.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    cardTopic.textContent = issue.topic || '';
    cardBody.textContent = issue.body || '(no description)';
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
        controller: {scrollZoom: {speed: 0.3, smooth: false}},
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


def generate_timeline(
    result: PipelineResult,
    issues: list[dict],
    output_dir: str = "./output",
) -> Path:
    """Generate a self-contained interactive timeline of topic evolution.

    Attempts to use BERTopic's ``topics_over_time()`` for the primary
    computation.  When that fails (common with small datasets) or returns
    an empty frame, falls back to a manual month-level aggregation.

    The resulting Plotly line chart is saved as a standalone HTML file at
    ``{output_dir}/timeline.html``.
    """
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio

    out_path = Path(output_dir) / "timeline.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    docs = [f"{issue['title']}\n{issue.get('body') or ''}" for issue in issues]
    timestamps = [issue["created_at"] for issue in issues]

    # ------------------------------------------------------------------
    # Try the native BERTopic method first
    # ------------------------------------------------------------------
    topics_over_time: pd.DataFrame | None = None
    try:
        tot = result.topic_model.topics_over_time(
            docs=docs,
            timestamps=timestamps,
        )
        if tot is not None and not tot.empty:
            topics_over_time = tot
            log.info("topics_over_time() returned %d rows.", len(tot))
    except Exception as exc:
        log.warning("topics_over_time() failed (%s); falling back to manual aggregation.", exc)

    # ------------------------------------------------------------------
    # Build the Plotly figure
    # ------------------------------------------------------------------
    fig = go.Figure()

    if topics_over_time is not None:
        # Filter out outlier topic (-1)
        df = topics_over_time[topics_over_time["Topic"] != -1].copy()

        for tid in sorted(df["Topic"].unique()):
            subset = df[df["Topic"] == tid].sort_values("Timestamp")
            label = result.get_label(int(tid))
            fig.add_trace(
                go.Scatter(
                    x=subset["Timestamp"],
                    y=subset["Frequency"],
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
    else:
        # ------------------------------------------------------------------
        # Manual fallback: group by month and topic
        # ------------------------------------------------------------------
        log.info("Using manual month-level aggregation for timeline.")

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
            df_manual = pd.DataFrame(records)
            counts = (
                df_manual.groupby(["month", "topic"])
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
        yaxis_title="Issue Count",
        hovermode="x unified",
        legend_title="Topic",
        template="plotly_white",
    )

    html = pio.to_html(fig, full_html=True, include_plotlyjs=True)
    out_path.write_text(html, encoding="utf-8")
    log.info("Saved topic timeline → %s", out_path)
    print(f"  -> {out_path}")

    return out_path.resolve()
