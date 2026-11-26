"""三种查询模式：baseline 为纯 LLM，naive 检索 Top-3，optimized 改写后召回 Top-6 并截断到 Top-3。"""
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    PromptTemplate,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
import chromadb

load_dotenv()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = BASE_DIR / "storage" / "chroma"


# ---------- LLM 初始化 ----------
def get_llm():
    """优先用 DeepSeek API（如配置了 key），否则用本地 Ollama。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        print("[LLM] 使用 DeepSeek API")
        return OpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=api_key,
            api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=0.1,
            timeout=120,
        )
    print("[LLM] 使用本地 Ollama（请先 ollama serve）")
    return Ollama(model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"), request_timeout=120)


def setup_settings():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5"
    )
    Settings.llm = get_llm()


def load_index():
    chroma_client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    chroma_collection = chroma_client.get_or_create_collection("agri_wheat", metadata={"hnsw:space": "l2"})
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store)


# ---------- 三种模式 ----------
PROMPT_AGRIC = PromptTemplate(
    "你是一名经验丰富的小麦农业技术员。请根据下面提供的【参考资料】回答农民的问题。\n"
    "要求：\n"
    "1. 只使用参考资料中的信息回答，不要编造；\n"
    "2. 如果参考资料中没有相关内容，明确说'根据现有资料无法回答'；\n"
    "3. 给出具体可操作的建议（药剂、用量、时期）；\n"
    "4. 回答末尾用【来源：xx文档】标注依据。\n\n"
    "【参考资料】:\n{context_str}\n\n"
    "【农民问题】: {query_str}\n\n"
    "【回答】:"
)


def answer_baseline(question: str) -> str:
    """A：纯 LLM，无 RAG"""
    resp = Settings.llm.complete(
        f"你是一名小麦农业技术员，请回答：{question}"
    )
    return str(resp)


def answer_naive(question: str, index: VectorStoreIndex) -> str:
    """B：朴素 RAG——向量检索 Top-3，直接拼 Prompt"""
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
    nodes = retriever.retrieve(question)
    context = "\n---\n".join([n.get_content() for n in nodes])
    prompt = PROMPT_AGRIC.format(context_str=context, query_str=question)
    resp = Settings.llm.complete(prompt)
    return f"【检索到 {len(nodes)} 个片段】\n{resp}"


def answer_optimized(question: str, index: VectorStoreIndex) -> str:
    """C：优化 RAG——Query 改写 + 召回Top6后按同一相似度分数截断为Top3 + 引用来源"""
    # Step 1: Query 改写（把农民口语改成检索式）
    rewrite_prompt = (
        "把下面这句农民的口语问题改写成更适合检索农业技术文档的书面化查询，"
        "保留关键病害/操作术语，不要回答问题，只输出改写后的查询：\n"
        f"原问题：{question}\n改写后："
    )
    rewritten = str(Settings.llm.complete(rewrite_prompt)).strip().strip('"')
    print(f"  [Query改写] {question}  ->  {rewritten}")

    # Step 2: 向量检索（Top-6，为截断留出候选）
    retriever = VectorIndexRetriever(index=index, similarity_top_k=6)
    nodes = retriever.retrieve(rewritten)

    # Step 3: 截断——按同一相似度分数排序后取 Top-3
    nodes_sorted = sorted(nodes, key=lambda n: n.score or 0, reverse=True)[:3]
    context = "\n---\n".join(
        [f"[片段{i+1} 来源：{n.metadata.get('file_name','?')}]\n{n.get_content()}"
         for i, n in enumerate(nodes_sorted)]
    )

    # Step 4: 生成
    prompt = PROMPT_AGRIC.format(context_str=context, query_str=question)
    resp = Settings.llm.complete(prompt)
    # 只按同一相似度分数截断
    return f"【Query改写】{rewritten}\n【截断后Top-{len(nodes_sorted)}】\n{resp}"


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "naive", "optimized"],
                        required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    setup_settings()
    index = load_index()

    print(f"\n{'='*60}\n模式: {args.mode.upper()}\n问题: {args.question}\n{'='*60}")
    if args.mode == "baseline":
        out = answer_baseline(args.question)
    elif args.mode == "naive":
        out = answer_naive(args.question, index)
    else:
        out = answer_optimized(args.question, index)

    print(f"\n--- 回答 ---\n{out}\n")


if __name__ == "__main__":
    main()
