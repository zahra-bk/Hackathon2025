# PR Review Dataset Builder

Turns PR review comments and follow-up commits into:
- Verdicts (did the change address the comment?)
- JSONL datasets for training and evaluation.

## How it works
- Two-commit view around the comment line:
  - commit1: base → original_commit (state at comment time)
  - commit2: original_commit → head (state after changes)
- Locality score: distance from the comment’s original line to the nearest change.
- Intent-aware semantic score:
  - If the comment asks to remove something, lower similarity after the change is good.
  - Otherwise, higher similarity after the change is good.
- Similarity backend (renormalized over available parts):
  - Cross-Encoder (default primary, 70%)
  - Embedding cosine (30%)
  - Lexical Jaccard (0% by default)

## Output
- data/classification.jsonl (all comments)
  - input: file, commit1_hunk, commit2_hunk, comment
  - label: 1/0
  - meta: scores, SHAs, line, local blocks
- data/generation.jsonl (positives only)
  - input: file, commit1_hunk, commit2_hunk
  - target: original comment

## Configure (config.json)
```json
{
  "repo": "owner/repo",
  "pr": 123,
  "comment": 123456789,
  "verbosity": 1,

  "use_cross_encoder": true,
  "cross_encoder_model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "use_embeddings": true,
  "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",

  "cross_encoder_weight": 0.70,
  "embed_weight": 0.30,
  "lexical_weight": 0.00,

  "locality_weight": 0.40,
  "semantic_weight": 0.60,
  "proximity_full_radius": 8,
  "proximity_zero_radius": 400,
  "semantic_focus_delta_threshold": 0.05,
  "presence_absence_bonus_max": 0.08,

  "combined_pass_threshold": 0.53,
  "combined_unclear_threshold": 0.50,

  "out_dir": "./data",
  "write_generation": true
}
```

## Run
- One comment (JSON to stdout): python evaluation_algorithm.py
- Build datasets for a PR: python training_data_builder.py

Notes
- commit1_hunk = after-text near the comment line at commit1 (base→original_commit).
- commit2_hunk = after-text near the comment line at commit2 (original_commit→head).
- Default similarity: Cross-Encoder primary (70%), cosine optional (30%).