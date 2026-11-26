# -*- coding: utf-8 -*-
"""逐题明细：检索排序位置 + 自动得分，供分层表与误差分析使用"""
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

col = chromadb.PersistentClient(path=str(BASE / "storage" / "chroma")).get_collection("agri_wheat")
got = col.get(include=["embeddings", "documents", "metadatas"])
V = np.asarray(got["embeddings"], dtype=np.float32)
V /= np.linalg.norm(V, axis=1, keepdims=True)
names = [str(m.get("file_name", "")).replace(".md", "") for m in got["metadatas"]]

model = SentenceTransformer(MODEL_DIR if MODEL_DIR else "BAAI/bge-small-zh-v1.5")
qs = [r["question"] for r in rows]


def l2_scores(q):
    diff = q[:, None, :] - V[None, :, :]
    return -np.sum(diff * diff, axis=2)

Q = l2_scores(model.encode(qs, normalize_embeddings=True))
R = l2_scores(model.encode([cache[q] for q in qs], normalize_embeddings=True))

out = []
for i, r in enumerate(rows):
    src = str(testset[r["id"]]["source_doc"])
    def rank(S):
        order = np.argsort(-S)
        pos = [p + 1 for p, j in enumerate(order) if names[j] == src]
        return pos[0] if pos else None, names[order[0]], names[order[1]], names[order[2]]
    p0, t1o, t2o, t3o = rank(Q[i])
    p2, t1r, t2r, t3r = rank(R[i])
    out.append({
        "id": r["id"], "category": testset[r["id"]]["category"], "source": src,
        "q": r["question"], "rewritten": cache[r["question"]],
        "rank_orig": p0, "rank_rewrite": p2,
        "top1_orig": t1o, "top1_rewrite": t1r,
        "score_A": r["baseline_score"], "score_B": r["naive_score"], "score_C": r["optimized_score"],
        "time_A": r["baseline_time"], "time_B": r["naive_time"], "time_C": r["optimized_time"],
    })

with open(OUT / "per_item.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

# 汇总：按源文档、按是否口语
def agg(key):
    d = {}
    for r in out:
        k = r[key]
        v = d.setdefault(k, {"n": 0, "r1_o": 0, "r1_r": 0, "r3_o": 0, "r3_r": 0,
                             "A": 0, "B": 0, "C": 0})
        v["n"] += 1
        v["r1_o"] += 1 if (r["rank_orig"] and r["rank_orig"] <= 1) else 0
        v["r1_r"] += 1 if (r["rank_rewrite"] and r["rank_rewrite"] <= 1) else 0
        v["r3_o"] += 1 if (r["rank_orig"] and r["rank_orig"] <= 3) else 0
        v["r3_r"] += 1 if (r["rank_rewrite"] and r["rank_rewrite"] <= 3) else 0
        v["A"] += int(r["score_A"]); v["B"] += int(r["score_B"]); v["C"] += int(r["score_C"])
    return d

for key in ["source"]:
    d = agg(key)
    print(f"=== by {key} ===")
    for k, v in d.items():
        print(f"{k}: n={v['n']} R@1(原/改)={v['r1_o']/v['n']:.2f}/{v['r1_r']/v['n']:.2f} "
              f"R@3={v['r3_o']/v['n']:.2f}/{v['r3_r']/v['n']:.2f} 得分 A/B/C="
              f"{v['A']/v['n']:.2f}/{v['B']/v['n']:.2f}/{v['C']/v['n']:.2f}")

print("\n=== 检索失败的题 ===")
for r in out:
    if not r["rank_orig"] or r["rank_orig"] > 3 or not r["rank_rewrite"]:
        print(r["id"], r["category"], "| 原排位", r["rank_orig"], "| 改写排位", r["rank_rewrite"], "|", r["q"][:34])

print("\n=== 改写改变 Top1 的题 ===")
for r in out:
    if r["top1_orig"] != r["top1_rewrite"]:
        print(r["id"], r["category"], r["top1_orig"], "->", r["top1_rewrite"], "|", r["q"][:30])

print("\n=== 得分差异题（B vs C） ===")
for r in out:
    if int(r["score_B"]) != int(r["score_C"]):
        print(r["id"], r["category"], "B", r["score_B"], "C", r["score_C"], "|", r["q"][:34])
