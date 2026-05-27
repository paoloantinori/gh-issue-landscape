"""Command-line interface for gh-issue-landscape."""

from __future__ import annotations

import click


@click.command()
@click.argument("repo")
@click.option(
    "--embedding",
    type=click.Choice(["local", "api", "openai"], case_sensitive=False),
    default="local",
    show_default=True,
    help="Embedding strategy: local model, generic OpenAI-compatible API, or OpenAI.",
)
@click.option(
    "--api-url",
    default=None,
    help="Base URL for an OpenAI-compatible embedding API endpoint.",
)
@click.option(
    "--api-key",
    default=None,
    envvar="OPENAI_API_KEY",
    help="API key for the embedding endpoint (also reads OPENAI_API_KEY).",
)
@click.option(
    "--state",
    type=click.Choice(["open", "closed", "all"], case_sensitive=False),
    default="open",
    show_default=True,
    help="Filter issues by state.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="./output",
    show_default=True,
    help="Directory to write output artifacts.",
)
@click.option("--3d", "three_d", is_flag=True, default=False, help="Generate 3D Three.js interactive explorer.")
@click.option("--timeline", is_flag=True, default=False, help="Generate topic timeline chart.")
@click.option("--viz-only", is_flag=True, default=False, help="Re-render visualizations from cached data (skip collection, embedding, and clustering).")
def main(
    repo: str,
    embedding: str,
    api_url: str | None,
    api_key: str | None,
    state: str,
    output: str,
    three_d: bool,
    timeline: bool,
    viz_only: bool,
) -> None:
    """Generate a semantic landscape visualization of GitHub issues.

    REPO is the target repository in owner/repo format (e.g. octocat/Hello-World).
    """
    try:
        if viz_only:
            _run_viz_only(repo, output, three_d, timeline)
        else:
            _run_full_pipeline(repo, embedding, api_url, api_key, state, output, three_d, timeline)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _run_full_pipeline(
    repo: str,
    embedding: str,
    api_url: str | None,
    api_key: str | None,
    state: str,
    output: str,
    three_d: bool,
    timeline: bool,
) -> None:
    click.echo(f"Repository : {repo}")
    click.echo(f"Embedding  : {embedding}")
    click.echo(f"State      : {state}")
    click.echo(f"Output dir : {output}")
    click.echo()

    click.echo("═══ Collecting issues ═══")
    from gh_issue_landscape.collector import collect_issues

    issues = collect_issues(repo, state=state, output_dir=output)
    click.echo(f"Fetched {len(issues)} issues.")
    click.echo()

    texts = [f"{issue['title']}\n{issue.get('body') or ''}" for issue in issues]

    click.echo("═══ Computing embeddings ═══")
    from gh_issue_landscape.embedder import embed

    embeddings = embed(
        texts, backend=embedding, output_dir=output, api_url=api_url, api_key=api_key,
    )
    click.echo(f"Embeddings: {embeddings.shape}")
    click.echo()

    click.echo("═══ Running topic model ═══")
    from gh_issue_landscape.pipeline import run_pipeline

    result = run_pipeline(texts, embeddings, output_dir=output, reduce_3d=three_d)
    n_topics = len([t for t in set(result.topics) if t != -1])
    n_outliers = result.topics.count(-1)
    click.echo(f"Found {n_topics} topic(s), {n_outliers} outlier(s) out of {len(texts)} documents.")
    click.echo()

    _render_outputs(result, issues, output, three_d, timeline)


def _run_viz_only(repo: str, output: str, three_d: bool, timeline: bool) -> None:
    import json
    from pathlib import Path

    import numpy as np

    data_dir = Path(output) / "data"

    for name in ("issues.json", "topics.json", "umap-2d.npy"):
        if not (data_dir / name).exists():
            raise ValueError(
                f"Missing {data_dir / name}. Run without --viz-only first to generate cached data."
            )

    click.echo(f"═══ Rebuilding visualizations from {data_dir} ═══")

    raw_issues = json.loads((data_dir / "issues.json").read_text())
    from gh_issue_landscape.collector import _clean_body
    issues = [
        {
            "number": i["number"],
            "title": i["title"],
            "body": _clean_body(i.get("body")),
            "url": i["url"],
            "labels": [l["name"] for l in i.get("labels", []) if isinstance(l, dict) and "name" in l],
            "created_at": i["createdAt"],
        }
        for i in raw_issues
    ]
    click.echo(f"Loaded {len(issues)} issues from cache.")

    topics_data = json.loads((data_dir / "topics.json").read_text())
    umap_2d = np.load(data_dir / "umap-2d.npy")

    umap_3d = None
    if three_d and (data_dir / "umap-3d.npy").exists():
        umap_3d = np.load(data_dir / "umap-3d.npy")
    elif three_d:
        raise ValueError("Missing umap-3d.npy. Run with --3d without --viz-only first.")

    # Reconstruct a minimal PipelineResult from cached data
    from gh_issue_landscape.pipeline import PipelineResult

    topic_labels: dict[int, str] = {}
    all_topics: list[int] = []
    doc_to_topic: dict[int, int] = {}
    for t in topics_data:
        tid = t["topic_id"]
        topic_labels[tid] = t["label"]
        for doc_idx in t.get("document_indices", []):
            doc_to_topic[doc_idx] = tid
    for i in range(len(issues)):
        all_topics.append(doc_to_topic.get(i, -1))

    result = PipelineResult(
        topic_model=None,  # type: ignore[arg-type]
        topics=all_topics,
        topic_info=None,  # type: ignore[arg-type]
        umap_2d=umap_2d,
        topic_labels=topic_labels,
        umap_3d=umap_3d,
    )

    _render_outputs(result, issues, output, three_d, timeline)


def _render_outputs(result, issues, output, three_d, timeline):
    click.echo("═══ Generating outputs ═══")
    from gh_issue_landscape.visualizer import generate_2d_map, generate_method_page, print_cli_summary

    map_path = generate_2d_map(result, issues, output_dir=output)
    print_cli_summary(result, issues)
    click.echo()
    click.echo(f"Landscape map: {map_path}")

    if three_d:
        from gh_issue_landscape.viewer_3d import generate_3d_explorer
        explorer_path = generate_3d_explorer(result, issues, output_dir=output)
        click.echo(f"3D explorer: {explorer_path}")

    if timeline:
        from gh_issue_landscape.visualizer import generate_timeline
        timeline_path = generate_timeline(result, issues, output_dir=output)
        click.echo(f"Timeline: {timeline_path}")

    method_path = generate_method_page(output_dir=output)
    click.echo(f"Method: {method_path}")
