# Semantic scoring and embeddings

This system answers two questions for a pull request (PR):
1) Did the code change near the comment?
2) Did the code change move in the direction the comment asked for?

We compute:
- locality_score: closeness in lines to the commented location (1.0 near, decays to 0.0)
- semantic_score: intent alignment between the comment and the edit

Final decision:
- combined = 0.40*locality + 0.60*semantic

## Components of the semantic score

We compare the review comment against three text blocks built from the local diff hunk:
- removed_block: lines that were deleted
- added_block: lines that were inserted
- local_block: removed_block + "\n" + added_block (old+new together)

We then compute three similarities:
- sim_removed = similarity(comment, removed_block)
- sim_added   = similarity(comment, added_block)
- sim_local   = similarity(comment, local_block)  ← see below

And a progress bonus:
- focus_raw = sim_added − sim_removed
- focus = 0 if focus_raw ≤ 0
- focus = min(1, focus_raw / (2*semantic_focus_delta_threshold)) if focus_raw > 0

Semantic score:
- semantic = 0.5*sim_added + 0.3*sim_local + 0.2*focus

### What exactly is sim_local?

- Definition: sim_local is the similarity between the comment and the concatenation of old and new lines in the hunk: local_block = removed_block + "\n" + added_block
- Purpose: capture “change-of-state” intent that only becomes clear when you see what was there before and what replaced it.

Why it matters:
- Replace/rename intent: “rename foo to bar,” “use AES-GCM instead of AES-ECB,” “replace print with logger.” Seeing X→Y in one text improves alignment.
- Constraint tightening/loosening: “Avoid any-to-any on WAN” is reflected by removing broad rules and adding restricted ones. The contrast of old vs new is the signal.
- Multi-line reorganizations: When the signal is split across old/new blocks, combining both stabilizes similarity.

Edge cases:
- Only added lines → sim_local == sim_added.
- Only removed lines → sim_local == sim_removed.
- No changes in the hunk → sim_local = 0.
- Large hunks can dilute sim_local; we mitigate by selecting the hunk nearest the comment line and using locality to downweight distant changes.

### Similarity method

- Hybrid: 85% sentence-embedding cosine + 15% lexical Jaccard to stabilize short or identifier-heavy text.
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
  - Family: Sentence-Transformers (bi-encoder)
  - Size: 6 Transformer layers (MiniLM), 384-dim embeddings
  - Purpose: fast, general-purpose semantic similarity
  - Runs locally on CPU (PyTorch). No calls to OpenAI/LLMs.
- If "use_embeddings": false in config, we fall back to lexical-only similarity.

### Example (from a real run)

Given the comment:
- “Scope HTTP to trusted sources; avoid any-to-any on WAN.”

And the local diff around the commented line, we observed:
- sim_added   = 0.2745
- sim_removed = 0.3388
- sim_local   = 0.3142
- focus_raw   = -0.0643 → focus = 0

Semantic:
- 0.5*0.2745 + 0.3*0.3142 + 0.2*0 = 0.2315

With locality=1.0:
- combined = 0.40*1.0 + 0.60*0.2315 = 0.5389 → passes a 0.53 threshold

## How to inspect these values

- Evaluator output (evaluation_algorithm.py) prints:
  - sim_removed, sim_added, sim_local, focus_delta, locality_score, semantic_score, combined_score.
- Dataset files (training_data_builder.py) include:
  - input.diff_context (local_block), input.added_block, input.removed_block
  - meta.scores: locality, semantic, combined, sim_* metrics

## Tunable knobs

- combined_pass_threshold: stricter or looser relevance cutoff.
- semantic_focus_delta_threshold: how quickly the focus bonus ramps up.
- locality_weight / semantic_weight: tradeoff between proximity and meaning.
- embed_model_name: switch to a different embedding model (first run caches it).
