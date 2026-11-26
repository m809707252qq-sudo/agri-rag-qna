# -*- coding: utf-8 -*-
"""对逐题得分做 bootstrap 重采样，计算均值的 95% 置信区间，结果写入 analysis/bootstrap_ci.json。"""
import csv
import json
import random
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
N_BOOT = 2000
SEED = 42


def ci(vals):
    random.seed(SEED)
    means = sorted(statistics.mean(random.choice(vals) for _ in vals) for _ in range(N_BOOT))
    return [round(means[int(0.025 * N_BOOT)], 2), round(means[int(0.975 * N_BOOT)], 2)]


rows = list(csv.DictReader(open(BASE / "analysis" / "per_item.csv", encoding="utf-8-sig")))
out = {}
for s in "ABC":
    vals = [float(r[f"score_{s}"]) for r in rows]
    out[s] = {"mean": round(statistics.mean(vals), 3), "ci95": ci(vals), "n": len(vals)}
print(json.dumps(out, ensure_ascii=False, indent=1))
(BASE / "analysis" / "bootstrap_ci.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
