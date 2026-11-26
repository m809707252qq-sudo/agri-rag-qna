# -*- coding: utf-8 -*-
"""Top-k 检索曲线：用已建好的向量库，逐 k 计算文档级 Recall 与 MRR。"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get("BGE_MODEL_DIR", "")

rows = list(csv.DictReader(open(BASE / "results" / "eval_scored.csv", encoding="utf-8-sig")))
testset = {r["id"]: r for r in csv.DictReader(open(BASE / "data" / "testset.csv", encoding="utf-8"))}
cache = json.load(open(OUT / "rewrites.json", encoding="utf-8"))
qs = [r["question"] for r in rows]
rq = [cache[q] for q in qs]
src = [str(testset[r["id"]]["source_doc"]) for r in rows]

col = chromadb.PersistentClient(path=str(BASE / "storage" / "chroma")).get_collection("agri_wheat")
got = col.get(include=["embeddings", "documents", "metadatas"])
V = np.asarray(got["embeddings"], dtype=np.float32)
V /= np.linalg.norm(V, axis=1, keepdims=True)
names = [str(m.get("file_name", "")).replace(".md", "") for m in got["metadatas"]]

model = SentenceTransformer(MODEL_DIR if MODEL_DIR else "BAAI/bge-small-zh-v1.5")
Q = model.encode(qs, normalize_embeddings=True)
R = model.encode(rq, normalize_embeddings=True)


def curve(S):
    out = {}
    for k in range(1, 7):
        hits, rr = 0, []
        for i, s in enumerate(src):
            order = np.argsort(-S[i])[:k]
            hit = [j for j in order if names[j] == s]
            hits += 1 if hit else 0
            rank = next((p + 1 for p, j in enumerate(order) if names[j] == s), None)
            rr.append(1.0 / rank if rank else 0.0)
        out[k] = {"recall": hits / len(src), "mrr": float(np.mean(rr))}
    return out


def l2_scores(q):
    diff = q[:, None, :] - V[None, :, :]
    return -np.sum(diff * diff, axis=2)

res = {"orig": curve(l2_scores(Q)), "rewrite": curve(l2_scores(R))}
json.dump(res, open(OUT / "topk_curve.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for tag in res:
    print(tag, {k: (round(v["recall"], 3), round(v["mrr"], 3)) for k, v in res[tag].items()})
