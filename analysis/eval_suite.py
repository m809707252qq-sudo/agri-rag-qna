# -*- coding: utf-8 -*-
"""检索指标和答案指标的计算脚本，结果写入 analysis/eval_summary.json。"""
import csv
import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

MODEL_DIR = os.environ.get("BGE_MODEL_DIR", "")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

KEY_TERMS = [
    "条锈", "叶脉", "成行", "虚线", "鲜黄", "三唑酮", "丙环唑", "烯唑醇", "戊唑醇",
    "发病中心", "病叶率", "100克", "拌种", "立克秀",
    "赤霉", "粉红", "霉层", "扬花", "见花就打", "氰烯菌酯", "戊唑", "咪鲜胺", "多菌灵",
    "抗药性", "呕吐毒素", "DON", "穗部", "麦穗", "补喷", "100毫升", "45", "60",
    "蚜", "蜜虫", "长管蚜", "二叉蚜", "缢管蚜", "黄矮", "吡虫啉", "啶虫脒", "抗蚜威",
    "瓢虫", "草蛉", "食蚜蝇", "蚜茧蜂", "天敌", "蜜蜂", "蜜露", "500头", "20克",
    "纹枯", "云纹", "花秆", "茎基", "拔节", "井冈霉素", "咯菌腈", "适乐时", "倒伏",
    "过密", "氮肥",
    "尿素", "磷酸二铵", "氯化钾", "有机肥", "基肥", "拔节", "追肥", "磷酸二氢钾",
    "干热风", "粒重", "播期", "9月15日", "10月5日", "基本苗", "氮肥后移", "脱肥",
    "硫酸锌", "花白苗", "缺磷", "分蘖",
]
FALLBACK = ["无法回答", "没有相关", "现有资料未", "未能提供", "不知道"]


def keywords(ref: str):
    kws = set()
    for m in re.findall(r"\d+(?:[–\-~]\d+)?(?:%|克|毫升|公斤|头|万|天|倍|公斤/亩)?", ref):
        if m.strip():
            kws.add(m.strip())
    for t in KEY_TERMS:
        if t in ref:
            kws.add(t)
    return kws


def hit_rate(ans: str, kws):
    if not ans or ans.startswith("[ERROR]") or not kws:
        return 0.0
    return sum(1 for k in kws if k in ans) / len(kws)


def bucket(rate: float) -> int:
    return 5 if rate >= .8 else 4 if rate >= .6 else 3 if rate >= .4 else 2 if rate >= .2 else 1


def obs_score(ans, ref):
    if not ans or ans.startswith("[ERROR]"):
        return 1, 0.0
    rate = hit_rate(ans, keywords(ref))
    s = bucket(rate)
    if any(p in ans for p in FALLBACK):
        s = min(s, 2)
    return s, rate


NUM = re.compile(r"\d+(?:[–\-~]\d+)?(?:\.\d+)?(?:%|克|毫升|公斤|头|万|天|倍|片|袋|次)?")
SOURCE_TAG = re.compile(r"【来源[^】]*】[：:]?[^\n]*")


def nums(text):
    # 先去掉【来源：01_小麦条锈病.md】这类标注，避免把文档编号 01/02 当成剂量数值
    text = SOURCE_TAG.sub("", text)
    return set(x for x in NUM.findall(text) if len(x) > 1)


def rewrite(prompt_text):
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/generate", json={
            "model": OLLAMA_MODEL, "prompt": prompt_text, "stream": False,
            "options": {"temperature": 0.1}}, timeout=180)
        return r.json().get("response", "").strip().strip('"')
    except Exception as e:
        return "[REWRITE_FAIL] %s" % e


def main():
    rows = list(csv.DictReader(open(BASE / "results" / "eval_scored.csv", encoding="utf-8-sig")))
    testset = {r["id"]: r for r in csv.DictReader(open(BASE / "data" / "testset.csv", encoding="utf-8"))}

    # ---------------- 检索层 ----------------
    col = chromadb.PersistentClient(path=str(BASE / "storage" / "chroma")).get_collection("agri_wheat")
    got = col.get(include=["embeddings", "documents", "metadatas"])
    V = np.asarray(got["embeddings"], dtype=np.float32)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    docs = got["documents"]
    metas = got["metadatas"]
    names = [str(m.get("file_name", "")).replace(".md", "") for m in metas]

    model = SentenceTransformer(MODEL_DIR if MODEL_DIR else "BAAI/bge-small-zh-v1.5")
    qs = [r["question"] for r in rows]

    # Query 改写（缓存，避免重跑）
    cache_f = OUT / "rewrites.json"
    cache = json.load(open(cache_f, encoding="utf-8")) if cache_f.exists() else {}
    for q in qs:
        if q not in cache or cache[q].startswith("[REWRITE_FAIL"):
            cache[q] = rewrite(
                "把下面这句农民的口语问题改写成更适合检索农业技术文档的书面化查询，"
                "保留关键病害/操作术语，不要回答问题，只输出改写后的查询：\n"
                f"原问题：{q}\n改写后：")
            json.dump(cache, open(cache_f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    qvec = model.encode(qs, normalize_embeddings=True)
    rvec = model.encode([cache[q] for q in qs], normalize_embeddings=True)

    def build(qv):
        # 平方欧氏距离；取负值后按降序排序，与 Chroma 的 l2 度量一致
        diff = qv[:, None, :] - V[None, :, :]
        return -np.sum(diff * diff, axis=2)

    S_orig, S_rew = build(qvec), build(rvec)
    C_orig, C_rew = qvec @ V.T, rvec @ V.T  # 单位向量内积，仅用于 top1_sim 诊断

    def metrics(S, src_list, ks=(1, 3, 6)):
        out = {}
        for k in ks:
            hits, rr = 0, []
            for i, src in enumerate(src_list):
                order = np.argsort(-S[i])
                top = order[:k]
                rel = [j for j in top if names[j] == src]
                hits += 1 if rel else 0
                if k == 6:
                    rank = next((r + 1 for r, j in enumerate(order[:6]) if names[j] == src), None)
                    rr.append(1.0 / rank if rank else 0.0)
            out["recall@%d" % k] = hits / len(src_list)
            if k == 6:
                out["mrr@6"] = float(np.mean(rr))
        return out

    src_list = [str(testset[r["id"]]["source_doc"]) for r in rows]
    cfg = {
        "A0_原问题Top3": metrics(S_orig[:, :], src_list, ks=(1, 3)),  # 与 Top6→截断Top3 等价
        "A1_原问题Top6": metrics(S_orig, src_list, ks=(6,)),
        "A2_改写Top3": metrics(S_rew, src_list, ks=(1, 3)),
        "A3_改写Top6": metrics(S_rew, src_list, ks=(6,)),
    }
    # 顶端 Top3 内容级指标：Top1 相似度均值
    cfg["A0_原问题Top3"]["top1_sim"] = float(np.mean(np.max(C_orig, axis=1)))
    cfg["A2_改写Top3"]["top1_sim"] = float(np.mean(np.max(C_rew, axis=1)))

    # 分层：按题型
    cats = [testset[r["id"]]["category"] for r in rows]
    strat = {}
    for c in sorted(set(cats)):
        idx = [i for i, x in enumerate(cats) if x == c]
        strat[c] = {
            "n": len(idx),
            "A0_recall@3": float(np.mean([metrics(S_orig[idx], [src_list[i] for i in idx])["recall@3"]])),
            "A2_recall@3": float(np.mean([metrics(S_rew[idx], [src_list[i] for i in idx])["recall@3"]])),
            "A0_recall@1": float(np.mean([metrics(S_orig[idx], [src_list[i] for i in idx])["recall@1"]])),
            "A2_recall@1": float(np.mean([metrics(S_rew[idx], [src_list[i] for i in idx])["recall@1"]])),
        }

    # ---------------- 答案层 ----------------
    sch = ["baseline", "naive", "optimized"]
    ans_stats = {}
    per_item = {s: [] for s in sch}
    for s in sch:
        src_rate, dose_cov, dose_bad, fallback, lens, times = [], [], [], [], [], []
        for r in rows:
            a = r[f"{s}_answer"]
            ref = r["reference"]
            lens.append(len(a))
            times.append(float(r[f"{s}_time"]) if r[f"{s}_time"] else np.nan)
            # 只统计“【来源...】”标注是否出现，不校验来源是否对应知识库文档
            src_rate.append(1 if "【来源" in a else 0)
            fallback.append(1 if any(p in a for p in FALLBACK) else 0)
            rn, an = nums(ref), nums(a)
            dose_cov.append(len(rn & an) / len(rn) if rn else np.nan)
            extra = an - rn
            dose_bad.append(len(extra) / len(an) if an else np.nan)
            per_item[s].append(obs_score(a, ref)[0])
        ans_stats[s] = {
            "auto_score_mean": float(np.mean(per_item[s])),
            "source_tag_rate": float(np.mean(src_rate)),
            "fallback_rate": float(np.mean(fallback)),
            "num_coverage": float(np.nanmean(dose_cov)),
            "unsupported_num_rate": float(np.nanmean(dose_bad)),
            "mean_len": float(np.mean(lens)),
            "mean_time": float(np.nanmean(times)),
        }

    # 与 CSV 中 score_eval 的结果核对
    chk = {}
    for s in sch:
        csv_mean = float(np.mean([float(r[f"{s}_score"]) for r in rows if r[f"{s}_score"]]))
        chk[s] = {"reproduced": ans_stats[s]["auto_score_mean"], "csv": csv_mean,
                  "match": abs(csv_mean - ans_stats[s]["auto_score_mean"]) < 1e-6}

    # ---------------- 显著性 ----------------
    sig = {}
    for a, b in [("baseline", "naive"), ("naive", "optimized"), ("baseline", "optimized")]:
        try:
            st = wilcoxon(per_item[a], per_item[b], zero_method="wilcox")
            sig[f"{a}_vs_{b}"] = {"stat": float(st.statistic), "p": float(st.pvalue)}
        except Exception as e:
            sig[f"{a}_vs_{b}"] = {"error": str(e)}

    # 分层答案得分
    for c in sorted(set(cats)):
        idx = [i for i, x in enumerate(cats) if x == c]
        strat[c]["ans_B"] = float(np.mean([per_item["naive"][i] for i in idx]))
        strat[c]["ans_C"] = float(np.mean([per_item["optimized"][i] for i in idx]))
        strat[c]["n"] = len(idx)

    summary = {
        "n_items": len(rows),
        "n_chunks": len(docs),
        "retrieval": cfg,
        "retrieval_stratified": strat,
        "answer": ans_stats,
        "score_check": chk,
        "wilcoxon": sig,
        "rewrite_samples": [{"q": q, "rewritten": cache[q]} for q in qs[:5]],
        "rewrite_fail": sum(1 for q in qs if cache[q].startswith("[REWRITE_FAIL")),
    }
    json.dump(summary, open(OUT / "eval_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
