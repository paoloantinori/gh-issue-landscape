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
def main(
    repo: str,
    embedding: str,
    api_url: str | None,
    api_key: str | None,
    state: str,
    output: str,
) -> None:
    """Generate a semantic landscape visualization of GitHub issues.

    REPO is the target repository in owner/repo format (e.g. octocat/Hello-World).
    """
    click.echo(f"Repository : {repo}")
    click.echo(f"Embedding  : {embedding}")
    click.echo(f"API URL    : {api_url or '(not set)'}")
    click.echo(f"API Key    : {'***' if api_key else '(not set)'}")
    click.echo(f"State      : {state}")
    click.echo(f"Output dir : {output}")
    click.echo()

    try:
        click.echo("═══ Collecting issues ═══")
        from gh_issue_landscape.collector import collect_issues

        issues = collect_issues(repo, state=state, output_dir=output)
        click.echo(f"Fetched {len(issues)} issues.")
        click.echo()

        texts = [
            f"{issue['title']}\n{issue.get('body') or ''}" for issue in issues
        ]

        click.echo("═══ Computing embeddings ═══")
        from gh_issue_landscape.embedder import embed

        embeddings = embed(
            texts,
            backend=embedding,
            output_dir=output,
            api_url=api_url,
            api_key=api_key,
        )
        click.echo(f"Embeddings: {embeddings.shape}")
        click.echo()

        click.echo("═══ Running topic model ═══")
        from gh_issue_landscape.pipeline import run_pipeline

        result = run_pipeline(texts, embeddings, output_dir=output)
        n_topics = len([t for t in set(result.topics) if t != -1])
        n_outliers = result.topics.count(-1)
        click.echo(
            f"Found {n_topics} topic(s), "
            f"{n_outliers} outlier(s) out of {len(texts)} documents."
        )
        click.echo()

        click.echo("═══ Generating outputs ═══")
        from gh_issue_landscape.visualizer import generate_2d_map, print_cli_summary

        map_path = generate_2d_map(result, issues, output_dir=output)
        print_cli_summary(result, issues)
        click.echo()
        click.echo(f"Landscape map: {map_path}")

    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
