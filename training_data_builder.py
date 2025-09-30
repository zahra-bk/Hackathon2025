#!/usr/bin/env python3
"""
Build JSONL datasets from a PR's review comments.

- Two-commit, intent-aware scoring (see core_scoring.evaluate_commits)
- classification.jsonl: all comments with label 1/0
- generation.jsonl: positives only (input = commit1/commit2 hunks; target = comment)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import requests

from core_scoring import (
    evaluate_commits,
    full_diff_text_to_file_patch,
)

GITHUB_API = "https://api.github.com"

DEFAULT_PROBLEM_TERMS = [
    "fix", "wrong", "incorrect", "should", "must", "bug",
    "security", "vuln", "avoid", "remove", "delete", "drop",
    "deny", "error", "fail", "unsafe", "leak", "broken", "not",
]
DEFAULT_PRAISE_TERMS = ["lgtm", "nice", "great", "thanks", "nit", "nitpick", "typo", "style"]


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


def _list_pr_review_comments(owner: str, repo: str, pr: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    page = 1
    while True:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr}/comments?per_page=100&page={page}"
        batch = _gh_get_json(url)
        if not batch:
            break
        out.extend(batch)
        page += 1
    return out


def _get_pr(owner: str, repo: str, pr: int) -> Dict[str, Any]:
    return _gh_get_json(f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr}")


def _get_compare_json(owner: str, repo: str, base: str, head: str) -> Dict[str, Any]:
    return _gh_get_json(f"{GITHUB_API}/repos/{owner}/{repo}/compare/{base}...{head}")


def _get_compare_diff(owner: str, repo: str, base: str, head: str) -> str:
    return _gh_get_text(
        f"{GITHUB_API}/repos/{owner}/{repo}/compare/{base}...{head}",
        accept="application/vnd.github.v3.diff",
    )


def _flags_real_problem(body: str, problem_terms: List[str], praise_terms: List[str]) -> bool:
    text = (body or "").lower()
    if any(t in text for t in praise_terms):
        return False
    return any(v in text for v in problem_terms)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run() -> None:
    """Main entry: build classification.jsonl and generation.jsonl for one PR."""
    cfg = _load_config(os.path.join(os.getcwd(), "config.json"))
    _setup_logging(int(cfg.get("verbosity", 0)))

    repo_full = cfg.get("repo", "")
    if "/" not in repo_full:
        raise RuntimeError("config.repo must be 'owner/repo'")
    owner, repo = repo_full.split("/", 1)

    pr_num = int(cfg.get("pr", 0))
    if pr_num <= 0:
        raise RuntimeError("config.pr (pull request number) is required")

    pr = _get_pr(owner, repo, pr_num)
    base_sha = cfg.get("base_commit") or pr.get("base", {}).get("sha")
    head_sha = cfg.get("head_commit") or pr.get("head", {}).get("sha")
    if not base_sha or not head_sha:
        raise RuntimeError("Unable to resolve base/head SHAs")

    comments = _list_pr_review_comments(owner, repo, pr_num)
    max_comments = cfg.get("max_comments")
    if isinstance(max_comments, int) and max_comments > 0:
        comments = comments[:max_comments]

    logging.info(
        "Building dataset from %d comments (base=%s..head=%s)",
        len(comments),
        (base_sha or "")[:7],
        (head_sha or "")[:7],
    )

    examples: List[Dict[str, Any]] = []
    for c in comments:
        try:
            path = c.get("path")
            original_line = c.get("original_line")
            comment_body = c.get("body") or ""
            original_commit = c.get("original_commit_id")
            if not path or original_line is None or not original_commit:
                continue

            # commit1: base → original_commit
            file_patch_c1 = ""
            cmp1 = _get_compare_json(owner, repo, base_sha, original_commit)
            cmp1_file = next((f for f in cmp1.get("files", []) if f.get("filename") == path), None)
            if cmp1_file and cmp1_file.get("patch"):
                file_patch_c1 = f"--- a/{path}\n+++ b/{path}\n{cmp1_file['patch']}"
            else:
                raw1 = _get_compare_diff(owner, repo, base_sha, original_commit)
                file_patch_c1 = full_diff_text_to_file_patch(raw1, path) or ""

            # commit2: original_commit → head
            cmp2 = _get_compare_json(owner, repo, original_commit, head_sha)
            cmp2_file = next((f for f in cmp2.get("files", []) if f.get("filename") == path), None)
            if cmp2_file and cmp2_file.get("patch"):
                file_patch_c2 = f"--- a/{path}\n+++ b/{path}\n{cmp2_file['patch']}"
            else:
                raw2 = _get_compare_diff(owner, repo, original_commit, head_sha)
                file_patch_c2 = full_diff_text_to_file_patch(raw2, path) or ""

            scores = evaluate_commits(comment_body, file_patch_c1, file_patch_c2, original_line, cfg)

            combined = scores["combined_score"]
            pass_thr = float(cfg.get("combined_pass_threshold", 0.53))
            relevant_change = combined >= pass_thr

            problem_terms = cfg.get("problem_terms") or DEFAULT_PROBLEM_TERMS
            praise_terms = cfg.get("praise_terms") or DEFAULT_PRAISE_TERMS
            real_problem = _flags_real_problem(comment_body, problem_terms, praise_terms)

            label = 1 if (real_problem and relevant_change) else 0

            examples.append(
                {
                    "repo": f"{owner}/{repo}",
                    "pr": str(pr_num),
                    "comment_id": c.get("id"),
                    "comment_body": comment_body,
                    "original_line": original_line,
                    "base_sha": base_sha,
                    "head_sha": head_sha,
                    "label": label,
                    "scores": {
                        "locality": scores["locality_score"],
                        "semantic": scores["semantic_score"],
                        "combined": combined,
                        "sim_commit1": scores.get("sim_commit1", 0.0),
                        "sim_commit2": scores.get("sim_commit2", 0.0),
                        "focus_delta": scores.get("focus_delta", 0.0),
                        "focus_component": scores.get("focus_component", 0.0),
                        "intent_removal": scores.get("intent_removal", 0.0),
                        "presence_absence_bonus": scores.get("presence_absence_bonus", 0.0),
                    },
                    "context": {
                        "file": path,
                        "commit1_hunk": scores.get("commit1_hunk", ""),
                        "commit2_hunk": scores.get("commit2_hunk", ""),
                        "commit1_local_block": scores.get("commit1_local_block", ""),
                        "commit2_local_block": scores.get("commit2_local_block", ""),
                        "min_line_distance": scores.get("min_line_distance"),
                    },
                }
            )

        except requests.HTTPError as e:
            logging.warning("Skip comment %s due to HTTP error: %s", c.get("id"), e)
        except Exception as e:
            logging.warning("Skip comment %s due to error: %s", c.get("id"), e)

    out_dir = cfg.get("out_dir", "./data")
    os.makedirs(out_dir, exist_ok=True)

    # classification.jsonl
    cls_rows: List[Dict[str, Any]] = []
    for ex in examples:
        cls_rows.append(
            {
                "repo": ex["repo"],
                "pr": ex["pr"],
                "comment_id": ex["comment_id"],
                "label": ex["label"],
                "input": {
                    "file": ex["context"]["file"],
                    "commit1_hunk": ex["context"]["commit1_hunk"],
                    "commit2_hunk": ex["context"]["commit2_hunk"],
                    "comment": ex["comment_body"],
                },
                "meta": {
                    "scores": ex["scores"],
                    "base_sha": ex["base_sha"],
                    "head_sha": ex["head_sha"],
                    "original_line": ex["original_line"],
                    "commit1_local_block": ex["context"]["commit1_local_block"],
                    "commit2_local_block": ex["context"]["commit2_local_block"],
                },
            }
        )
    _write_jsonl(os.path.join(out_dir, "classification.jsonl"), cls_rows)

    # generation.jsonl
    if bool(cfg.get("write_generation", True)):
        gen_rows: List[Dict[str, Any]] = []
        for ex in examples:
            if ex["label"] == 1 and ex["comment_body"].strip():
                gen_rows.append(
                    {
                        "repo": ex["repo"],
                        "pr": ex["pr"],
                        "comment_id": ex["comment_id"],
                        "input": {
                            "file": ex["context"]["file"],
                            "commit1_hunk": ex["context"]["commit1_hunk"],
                            "commit2_hunk": ex["context"]["commit2_hunk"],
                        },
                        "target": ex["comment_body"],
                        "meta": {
                            "scores": ex["scores"],
                            "base_sha": ex["base_sha"],
                            "head_sha": ex["head_sha"],
                            "original_line": ex["original_line"],
                            "commit1_local_block": ex["context"]["commit1_local_block"],
                            "commit2_local_block": ex["context"]["commit2_local_block"],
                        },
                    }
                )
        _write_jsonl(os.path.join(out_dir, "generation.jsonl"), gen_rows)

    logging.info(
        "Wrote %d classification examples and %d generation examples to %s",
        len(cls_rows),
        sum(1 for ex in examples if ex["label"] == 1 and ex["comment_body"].strip()),
        out_dir,
    )


if __name__ == "__main__":
    try:
        run()
    except requests.HTTPError as e:
        logging.error("HTTP error: %s", e)
        if e.response is not None:
            logging.error("Response snippet: %s", e.response.text[:600])
        sys.exit(1)
    except Exception as e:
        logging.exception("Unhandled error: %s", e)
        sys.exit(1)