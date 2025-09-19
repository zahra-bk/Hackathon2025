import argparse
import json
from typing import List, Tuple
from .loader import load_from_dir
from .similarity import build_tfidf


def aggregate_comments(comments: List[str], max_items: int = 10) -> List[str]:
    # naive aggregation: split by lines, deduplicate while keeping order
    seen = set()
    agg: List[str] = []
    for c in comments:
        for line in (c or "").splitlines():
            line = line.strip()
            if not line:
                continue
            if line not in seen:
                seen.add(line)
                agg.append(line)
            if len(agg) >= max_items:
                return agg
    return agg


def review(triplets_dir: str, new_before_text: str, top_k: int = 5) -> Tuple[List[dict], List[str]]:
    triplets = load_from_dir(triplets_dir)
    if not triplets:
        raise SystemExit(f"No triplets found in {triplets_dir}")

    corpus = [t.before for t in triplets]
    vocab, idf, doc_vectors, encode, cosine = build_tfidf(corpus)
    qvec = encode(new_before_text)

    scored = []
    for idx, t in enumerate(triplets):
        score = cosine(qvec, doc_vectors[idx])
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)

    top = scored[: max(1, top_k)]
    matches = [
        {
            "id": t.id,
            "title": t.title,
            "score": round(score, 4),
            "comment": t.comment,
            "meta": t.meta,
        }
        for score, t in top
    ]

    suggestions = aggregate_comments([t.comment for _, t in top], max_items=12)
    return matches, suggestions


def main():
    p = argparse.ArgumentParser(description="Suggest PR review comments from historical triplets.")
    p.add_argument("--triplets-dir", required=True, help="Directory containing XML triplets")
    p.add_argument("--input-file", required=True, help="File path with new BEFORE text")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", default="suggestions.md", help="Path to write markdown suggestions")
    args = p.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        new_before = f.read()

    matches, suggestions = review(args.triplets_dir, new_before, args.top_k)

    # Console summary
    print("Top matches:")
    for m in matches:
        print(f"- {m['title']} (score={m['score']}) -> {m['id']}")

    print("\nSuggested comments:")
    for s in suggestions:
        print(f"- {s}")

    # Write artifacts
    md_lines = ["# Suggested Review Comments", ""]
    for s in suggestions:
        md_lines.append(f"- {s}")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    with open(args.out.replace(".md", ".json"), "w", encoding="utf-8") as jf:
        json.dump({"matches": matches, "suggestions": suggestions}, jf, indent=2)

if __name__ == "__main__":
    main()