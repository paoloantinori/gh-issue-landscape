"""2D interactive landscape visualization and CLI summary report."""

from __future__ import annotations

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
