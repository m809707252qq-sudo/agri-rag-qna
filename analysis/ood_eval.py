# -*- coding: utf-8 -*-
"""域外问题拒答评测：对 10 个知识库外问题跑 A/B/C 三种方案，统计拒答率、来源标注和剂量数字个数，结果写入 results/ood_eval.csv。"""
import csv
import os
import re
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
import query as q  # noqa: E402

from llama_index.core import Settings  # noqa: E402
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402
from llama_index.llms.ollama import Ollama  # noqa: E402

MODEL_DIR = os.environ.get("BGE_MODEL_DIR", "")
# 若设置了本地快照目录则离线加载，否则回退到模型名（需能访问 HuggingFace）
embed_name = MODEL_DIR if MODEL_DIR else "BAAI/bge-small-zh-v1.5"

QUESTIONS = [
    "玉米大斑病叶片上长什么样，该怎么防治？",
    "水稻稻瘟病用什么药，一亩用多少？",
    "苹果树腐烂病怎么刮治，涂什么药？",
    "番茄晚疫病发病初期用什么药剂？",
    "棉花黄萎病怎么判断，怎么防治？",
    "大豆胞囊线虫用什么药拌种？",
    "马铃薯晚疫病什么天气容易发作？",
    "油菜菌核病在什么时期打药？",
    "黄瓜霜霉病用多少倍液喷雾？",
    "辣椒疫病根部腐烂怎么处理？",
]

# 只保留真正的拒答/资料缺失表述；“建议咨询当地”属于转介建议，不应计为拒答
REFUSE = ["资料中未", "未提及", "没有相关", "无法回答", "未提供", "知识库中未",
          "不能确定", "现有资料", "未找到", "无法确定", "不详",
          "无法从", "没有提及", "不涉及"]
NUM = re.compile(r"\d+(?:\.\d+)?(?:%|克|毫升|公斤|倍|片|袋)")

Settings.embed_model = HuggingFaceEmbedding(model_name=embed_name)
Settings.llm = Ollama(
    model=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
    request_timeout=120,
)
index = q.load_index()

rows = []
for i, question in enumerate(QUESTIONS, 1):
    rec = {"id": i, "question": question}
    for mode, fn in [("baseline", lambda x: q.answer_baseline(x)),
                     ("naive", lambda x: q.answer_naive(x, index)),
                     ("optimized", lambda x: q.answer_optimized(x, index))]:
        t0 = time.time()
        try:
            ans = fn(question)
        except Exception as e:  # noqa: BLE001
            ans = f"[ERROR] {e}"
        rec[f"{mode}_time"] = round(time.time() - t0, 2)
        rec[f"{mode}_answer"] = ans
        rec[f"{mode}_refuse"] = 1 if any(p in ans for p in REFUSE) else 0
        rec[f"{mode}_source"] = 1 if "【来源" in ans else 0
        rec[f"{mode}_nums"] = len(NUM.findall(ans))
    rows.append(rec)
    print(i, question, {m: (rec[f"{m}_refuse"], rec[f"{m}_nums"]) for m in ["baseline", "naive", "optimized"]}, flush=True)

out = BASE / "results" / "ood_eval.csv"
cols = list(rows[0].keys())
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print("saved", out)
