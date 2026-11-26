"""
eval.py
批量跑 30 条测试集，三种模式分别生成答案，输出 results/eval_template.csv

运行：python src/eval.py
"""
import csv
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
import chromadb

load_dotenv()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = Path(__file__).resolve().parent.parent
TESTSET = BASE_DIR / "data" / "testset.csv"
OUTPUT = BASE_DIR / "results" / "eval_template.csv"
PERSIST_DIR = BASE_DIR / "storage" / "chroma"


def get_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        return OpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=api_key,
            api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=0.1,
            timeout=120,
        )
    return Ollama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        request_timeout=120,
    )


def main():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
    Settings.llm = get_llm()

    chroma_client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    chroma_collection = chroma_client.get_or_create_collection("agri_wheat", metadata={"hnsw:space": "l2"})
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    rows = list(csv.DictReader(open(TESTSET, encoding="utf-8")))
    print(f"共 {len(rows)} 条测试问题")

    # 延迟导入避免循环
    import query as Q  # 同目录
    Q.setup_settings()

    out_rows = []
    for i, row in enumerate(rows, 1):
        q = row["question"]
        print(f"\n[{i}/{len(rows)}] {q}")
        rec = {"id": row["id"], "question": q, "reference": row["reference_answer"],
               "source": row["source_doc"], "category": row["category"]}

        for mode, fn in [("baseline", Q.answer_baseline),
                         ("naive", lambda x: Q.answer_naive(x, index)),
                         ("optimized", lambda x: Q.answer_optimized(x, index))]:
            try:
                t0 = time.time()
                if mode == "baseline":
                    ans = fn(q)
                else:
                    ans = fn(q)
                rec[f"{mode}_answer"] = ans
                rec[f"{mode}_time"] = round(time.time() - t0, 1)
                # 分数列先留空，由 score_eval.py 计算
                rec[f"{mode}_score"] = ""
                print(f"  {mode}: {rec[f'{mode}_time']}s")
            except Exception as e:
                rec[f"{mode}_answer"] = f"[ERROR] {e}"
                rec[f"{mode}_time"] = ""
                rec[f"{mode}_score"] = ""
                print(f"  {mode} ERROR: {e}")
        out_rows.append(rec)

    fieldnames = ["id", "question", "reference", "source", "category",
                  "baseline_answer", "baseline_time", "baseline_score",
                  "naive_answer", "naive_time", "naive_score",
                  "optimized_answer", "optimized_time", "optimized_score"]
    with open(OUTPUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"\n完成！结果已保存到: {OUTPUT}")
    print("下一步：运行 python src/score_eval.py，自动计算三方案得分并回写 eval_scored.csv。")


if __name__ == "__main__":
    main()
