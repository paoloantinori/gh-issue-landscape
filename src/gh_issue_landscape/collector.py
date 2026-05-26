"""GitHub issue collection via the ``gh`` CLI."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


def _fetch_issues(repo: str, state: str, limit: int) -> list[dict]:
    """Run ``gh issue list`` and return parsed JSON."""
    cmd = [
        "gh", "issue", "list",
        "-R", repo,
        "--state", state,
        "--json", "number,title,body,labels,createdAt,url",
        "--limit", str(limit),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, check=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        raise ValueError(
            "The GitHub CLI ('gh') is not installed. "
            "Install it from https://cli.github.com/ and authenticate with 'gh auth login'."
        )
    except subprocess.TimeoutExpired:
        raise ValueError(
            f"Timed out fetching issues from '{repo}'. "
            "The repository may have too many issues or the network is slow."
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        if "Could not resolve to a Repository" in stderr:
            raise ValueError(
                f"Repository '{repo}' not found. "
                "Check the owner/repo format and your access permissions."
            )
        if "authentication" in stderr.lower() or "gh auth" in stderr.lower():
            raise ValueError(
                f"Authentication error for '{repo}'. "
                "Run 'gh auth login' to authenticate the GitHub CLI."
            )
        first_line = stderr.split("\n", 1)[0]
        raise ValueError(f"Failed to fetch issues from '{repo}': {first_line}")

    try:
        issues = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse gh output as JSON: {exc}")

    if not isinstance(issues, list):
        raise ValueError(
            f"Expected a JSON array from gh, got {type(issues).__name__}."
        )

    return issues


_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_MULTI_SPACE = re.compile(r"[^\S\n]{2,}")


def _clean_body(text: str | None) -> str:
    """Strip HTML tags, markdown images, and excessive whitespace."""
    if not text:
        return ""
    text = _RE_MD_IMAGE.sub("", text)
    text = _RE_HTML_TAG.sub("", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    return text.strip()


def collect_issues(
    repo: str,
    state: str = "open",
    output_dir: str = "./output",
    *,
    limit: int = 5000,
) -> list[dict]:
    """Fetch GitHub issues and return cleaned issue dictionaries.

    Each dict has: number, title, body (cleaned), url, labels (list[str]), created_at.
    Raw JSON is saved to ``{output_dir}/data/issues.json``.
    """
    raw_issues = _fetch_issues(repo, state, limit)

    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "issues.json").write_text(
        json.dumps(raw_issues, ensure_ascii=False) + "\n",
    )

    return [
        {
            "number": issue["number"],
            "title": issue["title"],
            "body": _clean_body(issue.get("body")),
            "url": issue["url"],
            "labels": [
                label["name"]
                for label in issue.get("labels", [])
                if isinstance(label, dict) and "name" in label
            ],
            "created_at": issue["createdAt"],
        }
        for issue in raw_issues
    ]
