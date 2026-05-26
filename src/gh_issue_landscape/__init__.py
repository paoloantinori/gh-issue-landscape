"""gh-issue-landscape: Semantic 2D/3D visualizations of GitHub issues."""

try:
    from importlib.metadata import version

    __version__ = version("gh-issue-landscape")
except Exception:
    __version__ = "0.1.0.dev0"
