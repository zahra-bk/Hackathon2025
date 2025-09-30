#!/usr/bin/env python3
"""
Core scoring for PR review follow-up.

- Two-commit scoring around the commented line:
  commit1: base → original_commit (state at comment time)
  commit2: original_commit → head (state after changes)

- Intent-aware semantics:
  If the comment asks to remove something, lower similarity after the change is good.
  Otherwise, higher similarity after the change is good.

- Similarity backends (weighted):
  pair_similarity = w_ce*CrossEncoder + w_emb*cosine(embeddings) + w_lex*lexical_jaccard
  Defaults: w_ce=0.70, w_emb=0.30, w_lex=0.00 (weights renormalized over available backends).

Returns locality_score, semantic_score, combined_score, and diagnostics including commit1/commit2 hunks.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from unidiff import PatchSet

# ----------------------------- Tokenization & lexical -----------------------------

_TOKEN_RE = re.compile(r"[A-Za-z_]\w+|\d+")
_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those",
    "to", "for", "of", "on", "in", "into", "with", "by",
    "and", "or", "but", "if", "then", "else", "than",
    "be", "is", "are", "was", "were", "been", "being",
    "do", "does", "did", "done", "doing",
    "use", "using", "used", "via", "from", "as", "at",
    "should", "must", "need", "needs", "required", "require",
    "please", "kindly", "maybe", "perhaps",
    "we", "you", "i", "me", "my", "our", "your",
    "can", "could", "would", "will", "won't", "cannot", "cant", "can't",
    "no", "not", "without", "avoid",
}

REMOVAL_KEYWORDS = {
    "remove", "removal", "delete", "drop", "eliminate", "ban",
    "disallow", "forbid", "forbidden", "prohibit", "prohibited",
    "avoid", "disable", "turn off", "do not", "don't", "stop using",
    "no ", "without ", "exclude",
}


def tokenize(text: str) -> List[str]:
    """Split into identifier-like tokens and numbers."""
    return _TOKEN_RE.findall(text or "")


def lexical_similarity(a: str, b: str) -> float:
    """Jaccard over token sets (case-insensitive)."""
    ta = set(t.lower() for t in tokenize(a))
    tb = set(t.lower() for t in tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def content_tokens(text: str) -> List[str]:
    """Content-bearing tokens (stopwords removed)."""
    toks = [t.lower() for t in tokenize(text)]
    return [t for t in toks if t not in _STOPWORDS]


# ----------------------------- Embeddings and Cross-Encoder -----------------------------

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None  # type: ignore
    ST_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder  # type: ignore
    CE_AVAILABLE = True
except Exception:
    CrossEncoder = None  # type: ignore
    CE_AVAILABLE = False

_EMBED_MODEL = None
_CE_MODEL = None
_LOGGED_EMB = False
_LOGGED_CE = False


def load_embed_model(name: str):
    """Load and cache a SentenceTransformer bi-encoder by name, or return None."""
    global _EMBED_MODEL, _LOGGED_EMB
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    if not ST_AVAILABLE:
        if not _LOGGED_EMB:
            logging.info("Embeddings unavailable; skipping embedding cosine.")
            _LOGGED_EMB = True
        return None
    try:
        if not _LOGGED_EMB:
            logging.info("Load SentenceTransformer: %s", name)
            _LOGGED_EMB = True
        _EMBED_MODEL = SentenceTransformer(name)
    except Exception as e:
        logging.warning("Embedding model load failed: %s", e)
        _EMBED_MODEL = None
    return _EMBED_MODEL


def load_cross_encoder(name: str):
    """Load and cache a SentenceTransformers CrossEncoder by name, or return None."""
    global _CE_MODEL, _LOGGED_CE
    if _CE_MODEL is not None:
        return _CE_MODEL
    if not CE_AVAILABLE:
        if not _LOGGED_CE:
            logging.info("Cross-Encoder unavailable; skipping CE similarity.")
            _LOGGED_CE = True
        return None
    try:
        if not _LOGGED_CE:
            logging.info("Load Cross-Encoder: %s", name)
            _LOGGED_CE = True
        _CE_MODEL = CrossEncoder(name)
    except Exception as e:
        logging.warning("Cross-Encoder load failed: %s", e)
        _CE_MODEL = None
    return _CE_MODEL


def _sigmoid(x: float) -> float:
    """Numerically safe sigmoid."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _cross_encoder_score(a: str, b: str, ce_model) -> float:
    """Cross-Encoder similarity clamped to [0,1]."""
    if not ce_model or not a.strip() or not b.strip():
        return 0.0
    try:
        out = ce_model.predict([(a, b)])
        s = float(out[0]) if isinstance(out, (list, tuple)) else float(out)
        return _sigmoid(s) if s < 0.0 or s > 1.0 else max(0.0, min(1.0, s))
    except Exception as e:
        logging.warning("Cross-Encoder predict failed: %s", e)
        return 0.0


def _embed_texts(texts: List[str], model) -> List[List[float]]:
    """Encode texts with bi-encoder; returns list of vectors or zero-vectors."""
    try:
        return model.encode(texts, convert_to_numpy=True).tolist()
    except Exception as e:
        logging.warning("Embedding encode failed: %s", e)
        return [[0.0] * 384 for _ in texts]


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity in [0,1]."""
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        m = min(len(a), len(b))
        a, b = a[:m], b[:m]
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return max(0.0, min(1.0, num / (da * db)))


def pair_similarity(a: str, b: str, model_emb, model_ce, cfg: Dict[str, Any]) -> float:
    """
    Unified similarity with configurable weights.
    pair_similarity = w_ce*CE(a,b) + w_emb*cosine(emb(a),emb(b)) + w_lex*Jaccard
    """
    if not a.strip() or not b.strip():
        return 0.0

    w_ce = float(cfg.get("cross_encoder_weight", 0.70))
    w_emb = float(cfg.get("embed_weight", 0.30))
    w_lex = float(cfg.get("lexical_weight", 0.00))

    use_ce = bool(cfg.get("use_cross_encoder", True))
    use_emb = bool(cfg.get("use_embeddings", True))

    parts: List[Tuple[float, float]] = []

    if use_ce and model_ce and w_ce > 0.0:
        parts.append((w_ce, _cross_encoder_score(a, b, model_ce)))
    if use_emb and model_emb and w_emb > 0.0:
        v = _embed_texts([a, b], model_emb)
        parts.append((w_emb, _cosine(v[0], v[1])))
    if w_lex > 0.0:
        parts.append((w_lex, lexical_similarity(a, b)))

    if not parts:
        return lexical_similarity(a, b)

    s = sum(w for w, _ in parts)
    if s <= 0.0:
        return sum(score for _, score in parts) / len(parts)
    return sum((w / s) * score for w, score in parts)


# ----------------------------- Diff helpers -----------------------------

def full_diff_text_to_file_patch(diff_text: str, path: str) -> Optional[str]:
    """Extract one file's diff chunk from a multi-file diff."""
    marker = f"diff --git a/{path} b/{path}"
    lines = diff_text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if line.startswith(marker):
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("diff --git a/"):
            return "".join(lines[start:j])
    return "".join(lines[start:])


def _hunk_source_span(hunk) -> Tuple[int, int]:
    start = hunk.source_start
    end = start + hunk.source_length - 1
    return start, end


def _contains_line(hunk, old_line: int) -> bool:
    s, e = _hunk_source_span(hunk)
    return s <= old_line <= e


def _distance_to_span(hunk, old_line: int) -> int:
    s, e = _hunk_source_span(hunk)
    if old_line < s:
        return s - old_line
    if old_line > e:
        return old_line - e
    return 0


def _nearest_hunk_and_distance(patchset: PatchSet, old_line: int):
    nearest = None
    best: Optional[int] = None
    for fp in patchset:
        for h in fp:
            d = _distance_to_span(h, old_line)
            if best is None or d < best:
                best = d
                nearest = h
    return nearest, (best if best is not None else math.inf)


def _min_distance_to_changed_lines(hunk, original_line: int) -> Optional[int]:
    cur_old = hunk.source_start
    min_d: Optional[int] = None
    for line in hunk:
        if line.is_removed:
            d = abs(cur_old - original_line)
            min_d = d if min_d is None else min(min_d, d)
            cur_old += 1
        elif line.is_added:
            d = abs(cur_old - original_line)
            min_d = d if min_d is None else min(min_d, d)
        else:
            cur_old += 1
    return min_d


def extract_hunk_blocks(file_patch_text: str, original_line: int) -> Dict[str, Any]:
    """
    From a single-file diff, return the nearest hunk's blocks:
    - removed_block (old lines), added_block (new lines), local_block, distances.
    """
    patchset = PatchSet.from_string(file_patch_text or "")
    target = None
    dist_to_span: Optional[int] = None

    for fp in patchset:
        for h in fp:
            if _contains_line(h, original_line):
                target = h
                dist_to_span = 0
                break
        if target:
            break

    if target is None:
        target, dist_to_span = _nearest_hunk_and_distance(patchset, original_line)

    if target is None:
        return {
            "removed_block": "",
            "added_block": "",
            "local_block": "",
            "min_line_distance": None,
            "dist_to_hunk_span": None,
        }

    removed_lines: List[str] = []
    added_lines: List[str] = []
    cur_old = target.source_start
    for line in target:
        if line.is_removed:
            removed_lines.append(line.value.rstrip("\n"))
            cur_old += 1
        elif line.is_added:
            added_lines.append(line.value.rstrip("\n"))
        else:
            cur_old += 1

    removed_block = "\n".join(removed_lines)
    added_block = "\n".join(added_lines)
    local_block = (removed_block + "\n" + added_block).strip()
    min_line_distance = _min_distance_to_changed_lines(target, original_line)

    return {
        "removed_block": removed_block,
        "added_block": added_block,
        "local_block": local_block,
        "min_line_distance": min_line_distance if min_line_distance is not None else dist_to_span,
        "dist_to_hunk_span": dist_to_span,
    }


# ----------------------------- Locality and semantics -----------------------------

def locality_score_from_distance(dist: int, full_radius: int, zero_radius: int) -> float:
    """Map line distance to [0,1] with linear decay beyond full_radius."""
    if dist <= full_radius:
        return 1.0
    if dist >= zero_radius:
        return 0.0
    return max(0.0, 1.0 - (dist - full_radius) / max(1.0, (zero_radius - full_radius)))


def _detect_removal_intent(comment_body: str) -> bool:
    """Heuristic detection of removal intent in the comment."""
    text = (comment_body or "").lower()
    for kw in REMOVAL_KEYWORDS:
        if kw in text:
            return True
    if "do not " in text or "don't " in text or "should not " in text or "must not " in text:
        return True
    return False


def _presence_absence_bonus(comment: str, before_text: str, after_text: str, max_bonus: float) -> float:
    """Small bonus when tokens from the comment are present before but absent after (removal only)."""
    if max_bonus <= 0:
        return 0.0
    c = set(content_tokens(comment))
    if not c:
        return 0.0
    b = set(content_tokens(before_text))
    a = set(content_tokens(after_text))
    hits_before = c & b
    if not hits_before:
        return 0.0
    removed = hits_before - a
    if not removed:
        return 0.0
    frac = len(removed) / max(1, len(hits_before))
    return max(0.0, min(max_bonus, frac * max_bonus))


def _scale_focus(focus_raw: float, cfg: Dict[str, Any]) -> float:
    """Scale positive focus deltas into [0,1] using a small window."""
    if focus_raw <= 0:
        return 0.0
    d = float(cfg.get("semantic_focus_delta_threshold", 0.05))
    denom = max(1e-9, 2.0 * d)
    return max(0.0, min(1.0, focus_raw / denom))


def _semantic_intent_aware(
    comment_body: str,
    commit1_hunk: str,
    commit2_hunk: str,
    model_emb,
    model_ce,
    cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Compute semantic score with removal-aware inversion."""
    sim_c1 = pair_similarity(comment_body, commit1_hunk, model_emb, model_ce, cfg) if commit1_hunk else 0.0
    sim_c2 = pair_similarity(comment_body, commit2_hunk, model_emb, model_ce, cfg) if commit2_hunk else 0.0

    removal = _detect_removal_intent(comment_body)
    if removal:
        focus_raw = sim_c1 - sim_c2
        focus = _scale_focus(focus_raw, cfg)
        bonus = _presence_absence_bonus(
            comment_body, commit1_hunk, commit2_hunk, float(cfg.get("presence_absence_bonus_max", 0.08))
        )
        semantic = 0.8 * (1.0 - sim_c2) + 0.2 * focus + bonus
    else:
        focus_raw = sim_c2 - sim_c1
        focus = _scale_focus(focus_raw, cfg)
        bonus = 0.0
        semantic = 0.8 * sim_c2 + 0.2 * focus

    return semantic, {
        "sim_commit1": round(sim_c1, 4),
        "sim_commit2": round(sim_c2, 4),
        "focus_delta": round(focus_raw, 4),
        "focus_component": round(focus, 4),
        "intent_removal": 1.0 if removal else 0.0,
        "presence_absence_bonus": round(bonus, 4),
    }


# ----------------------------- Public API -----------------------------

def evaluate_commits(
    comment_body: str,
    commit1_file_patch_text: str,
    commit2_file_patch_text: str,
    original_line: int,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Score a review comment using two compare ranges.

    Args:
        comment_body: The review comment text.
        commit1_file_patch_text: Single-file diff for base..original_commit.
        commit2_file_patch_text: Single-file diff for original_commit..head.
        original_line: Old-file line number from the comment.
        cfg: Scoring/config dictionary (weights, model names, thresholds).

    Returns:
        Dict with locality_score, semantic_score, combined_score, hunks, and metrics.
    """
    # Extract nearest-hunk blocks for each step
    c1 = extract_hunk_blocks(commit1_file_patch_text, original_line) if commit1_file_patch_text else {
        "added_block": "", "local_block": "", "min_line_distance": None, "dist_to_hunk_span": None
    }
    c2 = extract_hunk_blocks(commit2_file_patch_text, original_line) if commit2_file_patch_text else {
        "added_block": "", "local_block": "", "min_line_distance": None, "dist_to_hunk_span": None
    }

    # After-state text per step
    commit1_hunk = c1.get("added_block", "") or ""
    commit2_hunk = c2.get("added_block", "") or ""

    # Locality from commit2 (fallback to commit1)
    min_line_distance = c2.get("min_line_distance")
    if min_line_distance is None:
        min_line_distance = c1.get("min_line_distance")
    full_r = int(cfg.get("proximity_full_radius", 8))
    zero_r = int(cfg.get("proximity_zero_radius", 400))
    locality = locality_score_from_distance(int(min_line_distance or 10**9), full_r, zero_r)

    # Models
    model_emb = load_embed_model(cfg.get("embed_model_name", "sentence-transformers/all-MiniLM-L6-v2")) \
        if bool(cfg.get("use_embeddings", True)) else None
    model_ce = load_cross_encoder(cfg.get("cross_encoder_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")) \
        if bool(cfg.get("use_cross_encoder", True)) else None

    # Semantic (intent-aware)
    semantic, sem_metrics = _semantic_intent_aware(
        comment_body, commit1_hunk, commit2_hunk, model_emb, model_ce, cfg
    )

    # Combine
    w_loc = float(cfg.get("locality_weight", 0.40))
    w_sem = float(cfg.get("semantic_weight", 0.60))
    s = w_loc + w_sem
    if s <= 0:
        w_loc, w_sem = 0.40, 0.60
        s = 1.0
    w_loc, w_sem = w_loc / s, w_sem / s
    combined = w_loc * locality + w_sem * semantic

    return {
        "locality_score": round(locality, 4),
        "semantic_score": round(semantic, 4),
        "combined_score": round(combined, 4),
        "min_line_distance": c2.get("min_line_distance"),
        "dist_to_hunk_span": c2.get("dist_to_hunk_span"),
        "commit1_hunk": commit1_hunk,
        "commit2_hunk": commit2_hunk,
        "commit1_local_block": c1.get("local_block", ""),
        "commit2_local_block": c2.get("local_block", ""),
        **sem_metrics,
        "weights": {"locality": round(w_loc, 3), "semantic": round(w_sem, 3)},
        "radii": {"full": full_r, "zero": zero_r},
    }