"""
build_index.py
构建小麦农业知识库的向量索引
运行：python src/build_index.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"
PERSIST_DIR = BASE_DIR / "storage" / "chroma"


def main():
    # 1. 配置 Embedding 模型（中文农业语料，用 bge-small-zh 足够）
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        embed_batch_size=16,
    )
    # 不在此处加载 LLM，建索引只用 embedding
    Settings.llm = None

    # 2. 读取知识库文档
    print(f"[1/4] 读取知识库: {KNOWLEDGE_DIR}")
    documents = SimpleDirectoryReader(input_dir=str(KNOWLEDGE_DIR)).load_data()
    print(f"      共加载 {len(documents)} 个文档块")

    # 3. 初始化 Chroma 持久化客户端
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    # 重建前先删掉旧集合，避免在已有 10 个文本块上继续追加，导致重复块
    try:
        chroma_client.delete_collection("agri_wheat")
    except Exception:
        pass
    chroma_collection = chroma_client.create_collection(
        "agri_wheat", metadata={"hnsw:space": "l2"}
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. 构建索引（chunk_size=1024, chunk_overlap=200）
    print("[2/4] 构建向量索引（首次会下载 bge-small-zh 模型，约 100MB）...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        insert_batch_size=10,
        transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=200)],
    )
    index.storage_context.persist()

    print(f"[3/4] 索引已保存到: {PERSIST_DIR}")
    print("[4/4] 完成！下一步运行 python src/query.py --mode naive --question '...' 测试")


if __name__ == "__main__":
    main()
