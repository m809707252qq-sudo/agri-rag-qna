# Windows 环境安装

## 1. 安装 Python 3.11

1. 打开 https://www.python.org/downloads/ ，下载 Python 3.11（不要用 3.13，部分库未适配）。
2. 安装时**务必勾选 `Add Python to PATH`**，再点 Install Now。
3. 按 `Win+R` 输入 `cmd` 打开命令行，验证：

```bash
python --version
pip --version
```

两条命令都能显示版本号即成功。

## 2. 进入项目目录

把 `agri-rag-qna` 文件夹放到路径不含中文、不含空格的位置，例如 `D:\agri-rag-qna`，命令行执行：

```bash
cd /d D:\agri-rag-qna
```

## 3. 创建虚拟环境（推荐）

```bash
python -m venv venv
venv\Scripts\activate
```

激活成功后命令行最前面出现 `(venv)`。

## 4. 安装依赖

```bash
pip install -r requirements.txt
```

下载慢则换清华源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 5. 安装 Ollama（本地大模型）

1. 官网 https://ollama.com 下载 Windows 版并安装。
2. 另开终端拉取模型：

```bash
ollama pull qwen2.5:7b
```

Windows 版装好后 Ollama 自动后台运行，无需手动 `serve`。

> 若电脑跑不动本地模型：查询和批量生成（`src/query.py`、`src/eval.py`）可改用 DeepSeek API：复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`。`analysis/eval_suite.py` 的 Query 改写和 `analysis/ood_eval.py` 仍默认使用本地 Ollama，跑这两个脚本前需要先安装 Ollama。

## 6. 验证环境

```bash
python scripts/smoke_test.py
```

项目已附带 `storage/chroma` 索引，正常情况下会看到 10 个文本块、512 维向量、30 题测试集全部通过。不要直接重复运行 `src/build_index.py` 来验证；如果确实要重建，脚本会先删除旧的 `agri_wheat` 集合再重建。
