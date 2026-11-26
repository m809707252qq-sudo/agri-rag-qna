"""按参考答案关键词命中率给三种方案的答案打分，结果写入 results/eval_scored.csv。

命中率 >=0.8 记5分，>=0.6 记4分，>=0.4 记3分，>=0.2 记2分，其余记1分；
答案含“无法回答/没有相关资料”等表述时最高记2分。
"""
import csv
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT = BASE_DIR / "results" / "eval_template.csv"
OUTPUT = BASE_DIR / "results" / "eval_scored.csv"

# 各文档/题型的核心术语词典（辅助命中判断）
KEY_TERMS = [
    # 条锈病
    "条锈", "叶脉", "成行", "虚线", "鲜黄", "三唑酮", "丙环唑", "烯唑醇", "戊唑醇",
    "发病中心", "病叶率", "100克", "拌种", "立克秀",
    # 赤霉病
    "赤霉", "粉红", "霉层", "扬花", "见花就打", "氰烯菌酯", "戊唑", "咪鲜胺", "多菌灵",
    "抗药性", "呕吐毒素", "DON", "穗部", "麦穗", "补喷", "100毫升", "45", "60",
    # 蚜虫
    "蚜", "蜜虫", "长管蚜", "二叉蚜", "缢管蚜", "黄矮", "吡虫啉", "啶虫脒", "抗蚜威",
    "瓢虫", "草蛉", "食蚜蝇", "蚜茧蜂", "天敌", "蜜蜂", "蜜露", "500头", "20克",
    # 纹枯病
    "纹枯", "云纹", "花秆", "茎基", "拔节", "井冈霉素", "咯菌腈", "适乐时", "倒伏",
    "过密", "氮肥",
    # 施肥
    "尿素", "磷酸二铵", "氯化钾", "有机肥", "基肥", "拔节", "追肥", "磷酸二氢钾",
    "干热风", "粒重", "播期", "9月15日", "10月5日", "基本苗", "氮肥后移", "脱肥",
    "硫酸锌", "花白苗", "缺磷", "分蘖",
]

FALLBACK_PATTERNS = ["无法回答", "没有相关", "现有资料未", "未能提供", "不知道"]


def extract_keywords(reference: str):
    """从参考答案提取得分点：数字片段 + 命中词典的术语。"""
    kws = set()
    # 数字（含单位）：如 100克、45-60公斤、10%、500头
    for m in re.findall(r"\d+(?:[–\-~]\d+)?(?:%|克|毫升|公斤|头|万|天|倍|公斤/亩)?", reference):
        if m.strip():
            kws.add(m.strip())
    # 词典术语
    for term in KEY_TERMS:
        if term in reference:
            kws.add(term)
    # 若参考答案没提取到数字，用 2-4 字中文短语兜底
    if not kws:
        for seg in re.split(r"[，。、；：（）()\"'“”]", reference):
            seg = seg.strip()
            if len(seg) >= 2:
                kws.add(seg)
    return kws


def hit_rate(answer: str, keywords: set) -> float:
    if not answer or answer.strip().startswith("[ERROR]"):
        return 0.0
    if not keywords:
        return 0.5
    hit = sum(1 for k in keywords if k in answer)
    return hit / len(keywords)


def score_of(answer: str, keywords: set) -> int:
    if not answer or answer.strip().startswith("[ERROR]"):
        return 1
    rate = hit_rate(answer, keywords)
    score = 5 if rate >= 0.8 else 4 if rate >= 0.6 else 3 if rate >= 0.4 else 2 if rate >= 0.2 else 1
    # 兜底表述压分
    if any(p in answer for p in FALLBACK_PATTERNS):
        score = min(score, 2)
    return score


def main():
    rows = list(csv.DictReader(open(INPUT, encoding="utf-8-sig")))
    schemes = ["baseline", "naive", "optimized"]
    sums = {s: 0 for s in schemes}
    times = {s: [] for s in schemes}
    n = 0

    for row in rows:
        keywords = extract_keywords(row["reference"])
        n += 1
        for s in schemes:
            ans = row.get(f"{s}_answer", "")
            sc = score_of(ans, keywords)
            row[f"{s}_score"] = sc
            sums[s] += sc
            t = row.get(f"{s}_time", "")
            if t:
                try:
                    times[s].append(float(t))
                except ValueError:
                    pass

    fieldnames = list(rows[0].keys()) if rows else []
    with open(OUTPUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\n共 {n} 条测试题，三方案平均分（1-5分）：")
    for s in schemes:
        avg = sums[s] / n if n else 0
        avg_t = sum(times[s]) / len(times[s]) if times[s] else 0
        print(f"  {s:10s} 自动均分={avg:.2f}  平均耗时={avg_t:.1f}s")
    print(f"\n打分结果已保存: {OUTPUT}")
    print("\n三方案自动得分已写入 results/eval_scored.csv。")


if __name__ == "__main__":
    main()
