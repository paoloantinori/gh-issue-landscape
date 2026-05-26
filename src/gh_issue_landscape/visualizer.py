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
