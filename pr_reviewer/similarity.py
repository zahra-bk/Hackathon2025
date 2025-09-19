import math
import re
from collections import Counter
from typing import Dict, List

TOKEN_RE = re.compile(r"[A-Za-z0-9_.:/\\-]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]

def build_tfidf(corpus_texts: List[str]):
    """
    Returns tuple: (vocab_index, idf, doc_vectors)
    - vocab_index: token -> index
    - idf: list[float] for each token index
    - doc_vectors: list[Dict[int, float]] sparse tf-idf for each doc
    """
    tokenized = [tokenize(t) for t in corpus_texts]
    # document frequency
    df = Counter()
    for toks in tokenized:
        df.update(set(toks))
    N = len(tokenized)
    vocab = {tok: i for i, tok in enumerate(sorted(df.keys()))}
    idf = [0.0] * len(vocab)
    for tok, i in vocab.items():
        idf[i] = math.log((N + 1) / (df[tok] + 1)) + 1.0

    def tfidf_vec(toks: List[str]) -> Dict[int, float]:
        tf = Counter(toks)
        vec: Dict[int, float] = {}
        for tok, freq in tf.items():
            if tok in vocab:
                i = vocab[tok]
                vec[i] = (freq / max(1, len(toks))) * idf[i]
        return vec

    doc_vectors = [tfidf_vec(toks) for toks in tokenized]

    def encode(text: str) -> Dict[int, float]:
        return tfidf_vec(tokenize(text))

    def cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
        dot = 0.0
        na = 0.0
        nb = 0.0
        for i, va in a.items():
            dot += va * b.get(i, 0.0)
            na += va * va
        for vb in b.values():
            nb += vb * vb
        if na == 0 or nb == 0:
            return 0.0
        return dot / ((na ** 0.5) * (nb ** 0.5))

    return vocab, idf, doc_vectors, encode, cosine