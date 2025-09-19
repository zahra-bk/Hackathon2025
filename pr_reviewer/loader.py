import os
import xml.etree.ElementTree as ET
from typing import List
from .models import Triplet

def _text_or_empty(elem):
    return (elem.text or "").strip() if elem is not None else ""

def load_from_file(path: str) -> List[Triplet]:
    """
    Supports two XML schemas:
    1) prReviewTriplets/triplet (multiple triplets)
    2) network_review_triplet (single triplet)
    """
    tree = ET.parse(path)
    root = tree.getroot()
    triplets: List[Triplet] = []

    if root.tag == "prReviewTriplets":
        # Many <triplet> children
        for i, t in enumerate(root.findall("triplet")):
            title = _text_or_empty(t.find("title")) or f"Triplet {i+1}"
            before = _text_or_empty(t.find("before"))
            comment = _text_or_empty(t.find("comment"))
            after = _text_or_empty(t.find("after"))
            repo = _text_or_empty(t.find("repo"))
            file_path = _text_or_empty(t.find("filePath"))
            commenter = _text_or_empty(t.find("commenter"))
            meta = {}
            if repo: meta["repo"] = repo
            if file_path: meta["filePath"] = file_path
            if commenter: meta["commenter"] = commenter
            triplets.append(
                Triplet(
                    id=f"{os.path.basename(path)}#triplet-{i+1}",
                    title=title,
                    before=before,
                    comment=comment,
                    after=after,
                    meta=meta,
                )
            )
        return triplets

    if root.tag == "network_review_triplet":
        # Single triplet file with before_config, review_comment, after_config
        title = _text_or_empty(root.find("description")) or os.path.basename(path)
        before = _text_or_empty(root.find("before_config"))
        comment = _text_or_empty(root.find("review_comment"))
        after = _text_or_empty(root.find("after_config"))
        meta = {}
        md = root.find("metadata")
        if md is not None:
            for child in md:
                # simple leaf nodes only
                if len(child) == 0 and child.text:
                    meta[child.tag] = child.text.strip()
        return [
            Triplet(
                id=os.path.basename(path),
                title=title,
                before=before,
                comment=comment,
                after=after,
                meta=meta,
            )
        ]

    raise ValueError(f"Unsupported XML root tag: {root.tag} in {path}")

def load_from_dir(dir_path: str) -> List[Triplet]:
    acc: List[Triplet] = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(".xml"):
                try:
                    acc.extend(load_from_file(os.path.join(root, f)))
                except Exception as e:
                    # Skip bad files but continue
                    print(f"[WARN] Skipping {f}: {e}")
    return acc