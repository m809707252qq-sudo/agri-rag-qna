# -*- coding: utf-8 -*-
"""快速自检：不加载生成模型，只校验数据与已建好的索引是否完好。

运行：python scripts/smoke_test.py
"""
import csv
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def check_chunks():
    import chromadb
    col = chromadb.PersistentClient(path=str(BASE / "storage" / "chroma")).get_collection("agri_wheat")
    got = col.get(include=["embeddings", "documents"])
    n = len(got["ids"])
    dim = len(got["embeddings"][0])
    assert n == 10, f"文本块数量应为 10，实际 {n}"
    assert dim == 512, f"向量维度应为 512，实际 {dim}"
    print(f"[ok] 向量库：{n} 个文本块，{dim} 维")


def check_testset():
    rows = list(csv.DictReader(open(BASE / "data" / "testset.csv", encoding="utf-8")))
    assert len(rows) == 30, f"测试集应为 30 题，实际 {len(rows)}"
    assert all(r["source_doc"] for r in rows), "每题都应标注 source_doc"
    print("[ok] 测试集：30 题，均标注来源文档")


def check_scores():
    rep = json.load(open(BASE / "analysis" / "eval_summary.json", encoding="utf-8"))
    chk = rep["score_check"]
    assert all(v["match"] for v in chk.values()), f"自动得分与 csv 不一致：{chk}"
    print("[ok] 自动得分：与 eval_scored.csv 完全一致")


def check_ood():
    f = BASE / "results" / "ood_eval.csv"
    if f.exists():
        rows = list(csv.DictReader(open(f, encoding="utf-8-sig")))
        assert len(rows) == 10, f"域外问题应为 10 条，实际 {len(rows)}"
        print("[ok] 域外评测：10 条")


if __name__ == "__main__":
    check_chunks()
    check_testset()
    check_scores()
    check_ood()
    print("全部通过")
