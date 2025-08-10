# Compact RAG (ideas-first, not production-ready)

import os, json, hashlib, re, math
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
import faiss

# --- Models (stubs) ---
from sentence_transformers import SentenceTransformer, CrossEncoder

# pip: rank_bm25, tiktoken, pysbd/spacy
from rank_bm25 import BM25Okapi
import tiktoken
import pysbd


# -----------------------------
# Config
# -----------------------------
@dataclass
class Cfg:
    model_dense: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_rerank: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    fetch_k: int = 200  # ANN candidates
    hybrid_k: int = 200  # BM25 candidates
    final_k: int = 8  # final chunks in context
    per_doc_cap: int = 2  # diversify sources
    mmr_lambda: float = 0.7  # relevance vs diversity
    neighbor_window: int = 1  # pull +/- neighbors
    min_dense_score: float = 0.2  # cosine threshold fail-safe
    pack_tokens: int = 1200  # token budget for context
    ann_hnsw_m: int = 32  # HNSW graph degree
    ann_ef_search: int = 128
    chunk_size: int = 800
    chunk_overlap: int = 120


# -----------------------------
# Minimal data structures
# -----------------------------
@dataclass
class Chunk:
    id: str  # 16-hex
    doc: str  # path or logical id
    order: int  # position within doc for neighbor stitching
    text: str


# -----------------------------
# Utilities
# -----------------------------
enc = tiktoken.get_encoding("cl100k_base")
seg = pysbd.Segmenter(language="en", clean=True)


def hash16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def to_faiss_id(cid_hex: str) -> np.int64:
    # full 64-bit id, preserve sign via two's complement
    return np.int64(int(cid_hex, 16))


def id_to_hex(fid: int) -> str:
    return format(np.uint64(fid).item(), "016x")


def tokenize_for_bm25(txt: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", txt.lower())


def mmr_select(qv: np.ndarray, cvecs: np.ndarray, k: int, lam: float) -> List[int]:
    # classic MMR over cosine sims
    sims = cvecs @ qv
    chosen, cand = [], list(range(len(cvecs)))
    while cand and len(chosen) < k:
        if not chosen:
            i = int(np.argmax(sims[cand]))
            chosen.append(cand.pop(i))
            continue
        div = np.max((cvecs[cand] @ cvecs[chosen].T), axis=1)
        score = lam * sims[cand] - (1 - lam) * div
        i = int(np.argmax(score))
        chosen.append(cand.pop(i))
    return chosen


def pack_blocks(blocks: List[str], max_tokens: int) -> str:
    out, used = [], 0
    for b in blocks:
        n = len(enc.encode(b))
        if out and used + n > max_tokens:
            break
        out.append(b)
        used += n
    return "\n\n---\n\n".join(out)


# -----------------------------
# Chunking (sentence-aware + overlap)
# -----------------------------
def chunk_text(doc_id: str, text: str, size: int, overlap: int) -> List[Chunk]:
    sents = seg.segment(text.strip()) or [text]
    chunks, cur, order = [], "", 0
    for s in sents:
        if len(cur) + len(s) + 1 <= size:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                cid = hash16(doc_id + "||" + str(order) + "||" + cur)
                chunks.append(Chunk(cid, doc_id, order, cur))
                order += 1
            tail = cur[-overlap:] if overlap > 0 else ""
            cur = (tail + " " + s).strip()
    if cur:
        cid = hash16(doc_id + "||" + str(order) + "||" + cur)
        chunks.append(Chunk(cid, doc_id, order, cur))
    return chunks


# -----------------------------
# Index & Meta (compact)
# -----------------------------
class Store:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.meta: Dict[str, Chunk] = {}  # cid -> Chunk
        self.by_doc: Dict[str, List[str]] = defaultdict(list)
        self.embed = SentenceTransformer(cfg.model_dense)
        self.rerank = CrossEncoder(cfg.model_rerank)
        # HNSW cosine (inner product on normalized vectors)
        dim = self.embed.get_sentence_embedding_dimension()
        base = faiss.IndexHNSWFlat(dim, cfg.ann_hnsw_m, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efSearch = cfg.ann_ef_search
        self.index = faiss.IndexIDMap2(base)
        self.bm25 = None  # will build from tokens
        self.doc_tokens: Dict[str, List[str]] = {}  # cid -> tokens
        self.model_name = cfg.model_dense

    def upsert_docs(self, docs: Dict[str, str]):
        # chunk, embed new, delete stale
        new_chunks: List[Chunk] = []
        current_by_doc = {}
        for doc_id, text in docs.items():
            chs = chunk_text(doc_id, text, self.cfg.chunk_size, self.cfg.chunk_overlap)
            current_by_doc[doc_id] = [c.id for c in chs]
            for c in chs:
                if c.id not in self.meta:
                    new_chunks.append(c)
        # deletes (stale)
        stale = []
        for d, ids in self.by_doc.items():
            keep = set(current_by_doc.get(d, []))
            stale.extend([cid for cid in ids if cid not in keep])
        if stale:
            sel = faiss.IDSelectorArray(
                len(stale), np.array([to_faiss_id(x) for x in stale], dtype=np.int64)
            )
            self.index.remove_ids(sel)
            for cid in stale:
                self.meta.pop(cid, None)
                self.doc_tokens.pop(cid, None)
        # insert new
        if new_chunks:
            texts = [c.text for c in new_chunks]
            vecs = self.embed.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            ).astype(np.float32)
            ids = np.array([to_faiss_id(c.id) for c in new_chunks], dtype=np.int64)
            self.index.add_with_ids(vecs, ids)
            for c in new_chunks:
                self.meta[c.id] = c
                self.by_doc[c.doc].append(c.id)
                self.doc_tokens[c.id] = tokenize_for_bm25(c.text)
        # rebuild BM25 corpus
        corpus = [self.doc_tokens[cid] for cid in self.meta.keys()]
        self.bm25 = BM25Okapi(corpus)

    # quick neighbor fetch based on order
    def neighbors(self, cid: str, w: int) -> List[str]:
        ch = self.meta[cid]
        ids = self.by_doc[ch.doc]
        i = ch.order
        neigh = []
        for di in range(1, w + 1):
            if i - di >= 0:
                neigh.append(ids[i - di])
            if i + di < len(ids):
                neigh.append(ids[i + di])
        return neigh


# -----------------------------
# Retrieval pipeline
# -----------------------------
def retrieve(store: Store, q: str, cfg: Cfg) -> List[Dict]:
    # 1) Dense ANN
    qv = store.embed.encode(
        [q], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)[0]
    D, I = store.index.search(
        qv.reshape(1, -1), min(cfg.fetch_k, max(1, store.index.ntotal))
    )
    dense = []
    for score, fid in zip(D[0].tolist(), I[0].tolist()):
        if fid == -1:
            continue
        cid = id_to_hex(fid)
        if cid in store.meta:
            dense.append((cid, float(score)))
    # 2) BM25
    q_tokens = tokenize_for_bm25(q)
    bm25_scores = store.bm25.get_scores(q_tokens)
    all_cids = list(store.meta.keys())
    top_bm25_idx = np.argsort(bm25_scores)[::-1][: cfg.hybrid_k]
    bm25 = [(all_cids[i], float(bm25_scores[i])) for i in top_bm25_idx]

    # 3) RRF fuse
    def rrf(r):
        return 1.0 / (60.0 + r)

    ranks_dense = {
        cid: r for r, (cid, _) in enumerate(sorted(dense, key=lambda x: -x[1]))
    }
    ranks_bm25 = {
        cid: r for r, (cid, _) in enumerate(sorted(bm25, key=lambda x: -x[1]))
    }
    fused = defaultdict(float)
    for cid in set(list(ranks_dense.keys()) + list(ranks_bm25.keys())):
        if cid in ranks_dense:
            fused[cid] += rrf(ranks_dense[cid])
        if cid in ranks_bm25:
            fused[cid] += rrf(ranks_bm25[cid])
    cand = sorted(fused.items(), key=lambda x: -x[1])[: max(cfg.final_k * 10, 50)]
    cids = [cid for cid, _ in cand]

    # 4) MMR diversify on dense vectors
    cvecs = store.embed.encode(
        [store.meta[c].text for c in cids],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    sel_idx = mmr_select(qv, cvecs, k=min(len(cids), cfg.final_k), lam=cfg.mmr_lambda)
    mmr_cids = [cids[i] for i in sel_idx]

    # 5) Neighbor stitching (expand context)
    stitched: List[str] = []
    seen = set()
    for cid in mmr_cids:
        for x in [cid] + store.neighbors(cid, cfg.neighbor_window):
            if x in seen:
                continue
            seen.add(x)
            stitched.append(x)

    # 6) Per-doc cap
    per_doc_count = defaultdict(int)
    capped = []
    for cid in stitched:
        d = store.meta[cid].doc
        if per_doc_count[d] < cfg.per_doc_cap:
            capped.append(cid)
            per_doc_count[d] += 1

    # 7) Cross-encoder rerank (quality)
    pairs = [(q, store.meta[c].text) for c in capped]
    ce_scores = store.rerank.predict(pairs)
    reranked = [
        c for _, c in sorted(zip(ce_scores, capped), key=lambda x: x[0], reverse=True)
    ]

    # 8) Thresholding (fallback honesty)
    dense_map = {cid: s for cid, s in dense}
    filtered = [c for c in reranked if dense_map.get(c, 0.0) >= cfg.min_dense_score]
    final_cids = filtered[: cfg.final_k] or reranked[: min(cfg.final_k, len(reranked))]
    out = [
        {
            "id": c,
            "doc": store.meta[c].doc,
            "order": store.meta[c].order,
            "text": store.meta[c].text,
            "dense_score": float(dense_map.get(c, 0.0)),
        }
        for c in final_cids
    ]
    return out


# -----------------------------
# Answering
# -----------------------------
def build_context(hits: List[Dict], cfg: Cfg) -> str:
    blocks = [
        f"[{h['id']} | {h['doc']} | {h['dense_score']:.3f}]\n{h['text'].strip()}"
        for h in hits
    ]
    return pack_blocks(blocks, cfg.pack_tokens)


def call_llm(prompt: str) -> str:
    # placeholder; wire to your provider
    return "[LLM OUTPUT PLACEHOLDER]\n" + prompt[:500]


def answer(store: Store, q: str, cfg: Cfg) -> str:
    hits = retrieve(store, q, cfg)
    if not hits:
        return json.dumps(
            {"answer": "I don't know based on the corpus.", "citations": []},
            ensure_ascii=False,
        )
    ctx = build_context(hits, cfg)
    prompt = (
        "You are a grounded system. Use only the CONTEXT to answer. "
        "Cite chunk IDs in brackets like [chunk_id]. If unsure, say you don't know.\n\n"
        f"QUESTION: {q}\n\nCONTEXT:\n{ctx}\n"
    )
    ans = call_llm(prompt)
    cites = [
        {"id": h["id"], "doc": h["doc"], "order": h["order"], "score": h["dense_score"]}
        for h in hits
    ]
    return json.dumps({"answer": ans, "citations": cites}, ensure_ascii=False, indent=2)


# -----------------------------
# Example wiring (conceptual)
# -----------------------------
def main():
    cfg = Cfg()
    store = Store(cfg)
    # Upsert corpus (dict: doc_id -> raw text). In real code, load files, PDF->text, etc.
    store.upsert_docs(
        {
            "docA": "Your parsed text here...",
            "docB": "Another document ...",
        }
    )
    print(answer(store, "What does the doc say about X?", cfg))


if __name__ == "__main__":
    main()
