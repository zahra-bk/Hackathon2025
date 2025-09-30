#!/usr/bin/env python3
"""
Evaluate a single PR review comment with two-commit, intent-aware scoring.

- commit1: base → original_commit  (state at comment time)
- commit2: original_commit → head  (state after changes)

Prints a JSON summary to stdout.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import requests

from core_scoring import (
    evaluate_commits,
    full_diff_text_to_file_patch,
)

GITHUB_API = "https://api.github.com"


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _gh_headers(accept: str = "application/vnd.github+json") -> Dict[str, str]:
    h = {"Accept": accept}
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _gh_get_json(url: str) -> Any:
    r = requests.get(url, headers=_gh_headers())
    r.raise_for_status()
    return r.json()


def _gh_get_text(url: str, accept: str) -> str:
    r = requests.get(url, headers=_gh_headers(accept))
    r.raise_for_status()
    return r.text


def _parse_repo(repo: str) -> tuple[str, str]:
    if "/" not in repo:
        raise ValueError("repo must be owner/repo")
    return repo.split("/", 1)


def _get_review_comment(owner: str, repo: str, comment_id: int) -> Dict[str, Any]:
    return _gh_get_json(f"{GITHUB_API}/repos/{owner}/{repo}/pulls/comments/{comment_id}")


def _get_pr(owner: str, repo: str, pr: int) -> Dict[str, Any]:
    return _gh_get_json(f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr}")


def _get_compare_json(owner: str, repo: str, base: str, head: str) -> Dict[str, Any]:
    return _gh_get_json(f"{GITHUB_API}/repos/{owner}/{repo}/compare/{base}...{head}")


def _get_compare_diff(owner: str, repo: str, base: str, head: str) -> str:
    return _gh_get_text(
        f"{GITHUB_API}/repos/{owner}/{repo}/compare/{base}...{head}",
        accept="application/vnd.github.v3.diff",
    )


def run_check(
    repo: str,
    comment_id: int,
    pr: Optional[int],
    second_commit: Optional[str],
    verbosity: int,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run evaluator and return summary dict."""
    _setup_logging(verbosity)
    owner, name = _parse_repo(repo)
    comment = _get_review_comment(owner, name, comment_id)

    path = comment["path"]
    original_commit = comment.get("original_commit_id")
    head_commit = second_commit or comment.get("commit_id")
    original_line = comment.get("original_line")
    comment_body = comment.get("body", "")

    if not original_commit or original_line is None:
        raise RuntimeError("Comment missing original_commit_id or original_line")

    base_sha: Optional[str] = None
    if pr:
        pr_meta = _get_pr(owner, name, pr)
        base_sha = pr_meta.get("base", {}).get("sha")

    # commit1 patch: base → original_commit
    file_patch_c1 = ""
    if base_sha:
        cmp1 = _get_compare_json(owner, name, base_sha, original_commit)
        cmp1_file = next((f for f in cmp1.get("files", []) if f.get("filename") == path), None)
        if cmp1_file and cmp1_file.get("patch"):
            file_patch_c1 = f"--- a/{path}\n+++ b/{path}\n{cmp1_file['patch']}"
        else:
            raw1 = _get_compare_diff(owner, name, base_sha, original_commit)
            file_patch_c1 = full_diff_text_to_file_patch(raw1, path) or ""

    # commit2 patch: original_commit → head_commit
    cmp2 = _get_compare_json(owner, name, original_commit, head_commit)
    cmp2_file = next((f for f in cmp2.get("files", []) if f.get("filename") == path), None)
    if cmp2_file and cmp2_file.get("patch"):
        file_patch_c2 = f"--- a/{path}\n+++ b/{path}\n{cmp2_file['patch']}"
    else:
        raw2 = _get_compare_diff(owner, name, original_commit, head_commit)
        file_patch_c2 = full_diff_text_to_file_patch(raw2, path) or ""

    # Score
    scores = evaluate_commits(comment_body, file_patch_c1, file_patch_c2, original_line, cfg)

    # Verdict
    pass_thr = float(cfg.get("combined_pass_threshold", 0.53))
    unclear_thr = float(cfg.get("combined_unclear_threshold", max(0.0, pass_thr - 0.03)))
    combined = scores["combined_score"]
    if combined >= pass_thr:
        verdict = "addressed"
    elif combined >= unclear_thr:
        verdict = "unclear"
    else:
        verdict = "not_addressed"

    return {
        "repo": f"{owner}/{name}",
        "path": path,
        "comment_id": comment_id,
        "pr": pr,
        "base_commit": base_sha,
        "original_commit": original_commit,
        "second_commit": head_commit,
        "original_line": original_line,
        "comment_body": comment_body,
        "verdict": verdict,
        "combined_score": scores["combined_score"],
        "locality_score": scores["locality_score"],
        "semantic_score": scores["semantic_score"],
        "commit1_hunk": scores.get("commit1_hunk", ""),
        "commit2_hunk": scores.get("commit2_hunk", ""),
        "metrics": {
            "sim_commit1": scores.get("sim_commit1", 0.0),
            "sim_commit2": scores.get("sim_commit2", 0.0),
            "focus_delta": scores.get("focus_delta", 0.0),
            "focus_component": scores.get("focus_component", 0.0),
            "intent_removal": scores.get("intent_removal", 0.0),
            "presence_absence_bonus": scores.get("presence_absence_bonus", 0.0),
            "min_line_distance": scores.get("min_line_distance"),
            "dist_to_hunk_span": scores.get("dist_to_hunk_span"),
            "weights": scores.get("weights", {}),
            "radii": scores.get("radii", {}),
        },
    }


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    try:
        cfg_path = os.path.join(os.getcwd(), "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError("Missing config.json")
        cfg = _load_config(cfg_path)

        summary = run_check(
            repo=cfg.get("repo", ""),
            comment_id=int(cfg.get("comment", 0)),
            pr=cfg.get("pr"),
            second_commit=cfg.get("second_commit") or None,
            verbosity=int(cfg.get("verbosity", 0)),
            cfg=cfg,
        )
        print(json.dumps(summary, indent=2))
        if summary["verdict"] != "addressed":
            sys.exit(2)
    except requests.HTTPError as e:
        logging.error("HTTP error: %s", e)
        if e.response is not None:
            logging.error("Response snippet: %s", e.response.text[:600])
        sys.exit(1)
    except Exception as e:
        logging.exception("Unhandled error: %s", e)
        sys.exit(1)