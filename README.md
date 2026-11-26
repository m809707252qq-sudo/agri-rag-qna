# 农业 RAG 问答系统

知识库为 5 篇小麦病虫害与施肥文档，测试集 30 题，比较三种方案：

- A 纯 LLM：不使用知识库，直接生成回答；
- B 朴素 RAG：向量检索 Top-3 后生成；
- C 优化 RAG：先做 Query 改写，召回 Top-6，再按同一相似度分数取 Top-3 后生成。

方案 C 的 Top-6 到 Top-3 只是在同一相似度分数上排序截断，没有额外的排序模型，也没有引入 BM25。

## 运行环境

Python 3.11。安装依赖：

```bash
pip install -r requirements.txt
```

本地生成模型使用 Ollama：

```bash
ollama pull qwen2.5:7b
```

如需改用云端 API，复制 `.env.example` 为 `.env`，填写 `DEEPSEEK_API_KEY`。`src/query.py` 和 `src/eval.py` 会使用 DeepSeek；`analysis/eval_suite.py` 的 Query 改写和 `analysis/ood_eval.py` 仍默认调用本地 Ollama。

## 运行步骤

```bash
# 1. 建库（可选；包内已带 storage/chroma，可直接从第 2 步开始）
python src/build_index.py

# 2. 单题查询
python src/query.py --mode baseline  --question "小麦条锈病初期叶片什么样？"
python src/query.py --mode naive     --question "小麦条锈病初期叶片什么样？"
python src/query.py --mode optimized --question "小麦条锈病初期叶片什么样？"

# 3. 批量生成 30 题答案
python src/eval.py

# 4. 关键词命中率自动打分
python src/score_eval.py

# 5. 检索指标、显著性检验和域外拒答评测
python analysis/eval_suite.py
python analysis/ood_eval.py
```

`storage/chroma` 已附带建好的索引，可直接从第 2 步开始运行。重新运行 `src/build_index.py` 会先删除 `agri_wheat` 集合再重建，不会在原有 10 个文本块上追加；只想验证环境时运行 `python scripts/smoke_test.py`。

## 目录

```text
data/knowledge/     5 篇知识库文档
data/testset.csv    30 题测试集
src/                建库、查询、批量生成、自动打分
analysis/           检索指标、bootstrap、显著性检验、域外评测
results/            逐题答案、自动得分、域外评测结果
storage/chroma/     Chroma 向量库
scripts/smoke_test.py
```

## 评测

检索部分算 Recall@1/3/6 和 MRR@6；答案部分算自动得分、来源标注率、数值覆盖率、无依据数值率和兜底率。来源标注率只统计答案中是否出现 `【来源...】` 标注，不校验标注是否对应知识库文档；得分按参考答案关键词命中率分档，统计检验用 Wilcoxon 符号秩检验和 bootstrap 区间。

评测结果在 `analysis/eval_summary.json`、`analysis/per_item.csv` 和 `results/ood_eval.csv` 中。随包附带的结果由本地 Qwen2.5-7B 生成；改用 DeepSeek API 后，`src/eval.py` 生成的答案和分数可能不同，`analysis` 脚本仍使用本地 Ollama。关键词命中率只是答案质量的代理指标，不能完全代表事实正确性，这也是关键词打分的主要局限。

## 向量度量

Chroma 集合设置 `hnsw:space = l2`。代码按平方 L2 距离计算和排序。由于 bge 向量已归一化，平方 L2 排序与内积排序等价。
