# quick-loop

`quick-loop` 是面向 Search Agent 与 test-time latent loop 研究的统一项目。当前主干支持
Ouro R3/R4 推理、Search-R1/Hermes 协议、E5 + Wiki18 + FAISS Retriever、veRL GRPO
链路，以及 Qwen2.5 training-free latent loop reference backend。

仓库按稳定骨干组织：共享实现放在 `main/src/`，正式实验按项目放在 `main/eval/`，
外部数据、生成数据、外部模型和本项目 checkpoint 严格分开。此前的性能测试、Prompt
遵循测试、临时 benchmark 及其结果已经清空，后续实验应在对应实验目录中重新建立。

## 环境

项目使用 Python 3.10 和 `uv`，不使用 Conda。当前锁文件基于 PyTorch 2.7.1、
Transformers 4.54.1、FastAPI 0.115.12 和 Uvicorn 0.34.3。安装与下载只允许使用国内镜像：

```bash
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$PWD/main/models/huggingface"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_DATASETS_CACHE="$PWD/main/datasets/huggingface"

uv sync --extra retriever --extra train --index "$UV_INDEX_URL"
```

`.venv/` 位于仓库根目录；uv 缓存、第三方可编辑源码、编译头文件和运行时临时文件也
统一收在 `.venv/` 下，均不提交 Git。

## 数据与模型

- `main/datasets/wiki18/`：外部 Wiki18 语料与 E5 Flat index。
- `main/datasets/huggingface/`：外部数据集的 Hugging Face 缓存。
- `main/data/searchr1/`：由项目处理生成的 NQ + HotpotQA parquet。
- `main/models/huggingface/`：Ouro Base、Ouro Thinking、E5 等外部模型缓存。
- `main/models/searchr1/`：Search-R1 官方 Qwen2.5-3B/7B 权重。
- `main/models/vllm-ouro-*/`：从官方 Ouro snapshot 生成的固定深度 vLLM 视图。
- `main/checkpoints/`：本项目训练产生的 checkpoint。

真实 Retriever 第一次读取 `main/datasets/wiki18/wiki-18.jsonl` 时会在
`main/datasets/huggingface/` 构建 Arrow cache；这只是本地转换，不会重新下载语料。

下载和准备入口均从仓库根目录执行：

```bash
HF_ENDPOINT=https://hf-mirror.com uv run --no-sync python main/scripts/download_wiki18.py
HF_ENDPOINT=https://hf-mirror.com uv run --no-sync python main/scripts/prepare_searchr1_data.py
HF_ENDPOINT=https://hf-mirror.com uv run --no-sync python main/scripts/download_searchr1_baselines.py
HF_ENDPOINT=https://hf-mirror.com uv run --no-sync python main/scripts/prepare_vllm_ouro.py
```

下载脚本会拒绝非 `https://hf-mirror.com` 的 Hugging Face endpoint。

## Retriever

Retriever 是 `ouro_search.retriever` 包的一部分，Agent 只通过 HTTP 调用它。真实服务保持
Wiki18、E5、Flat index 与 top-k=3 不变：

```bash
bash main/scripts/launch_e5_retriever.sh
curl -X POST http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -d '{"query":"Who wrote Hamlet?","top_k":3}'
```

最小 Mock 服务：

```bash
bash main/scripts/launch_mock_retriever.sh
```

## Prompt 与轨迹

项目明确区分两条执行路径：共享 Agent 状态机继续支持 `search_r1` 和 `hermes`
Prompt Profile；正式 Search GRPO 与后续 loop intervention 默认使用独立的 Search-R1
v0.2 官方 rollout 路径。协议设计见
[main/docs/prompt_profiles.md](main/docs/prompt_profiles.md)。默认轨迹写入
`main/data/trajectories/`；训练专用轨迹写入 `main/data/trajectories/search_grpo/`。

## Search GRPO 实验

Search GRPO 的执行代码、专用配置和 veRL patch 集中在
`main/eval/search_grpo/`。主实验模型是 Search-R1 官方发布的 Qwen2.5-3B/7B GRPO v0.2
权重，默认选择 3B；Ouro 已移出默认路径，只保留为显式可选配置。外部 veRL 源码安装到
`.venv/src/verl/`：

- `configs/search_r1/v0_2.yaml`：固定 Search-R1 commit 上的官方 baseline；
- `configs/search_r1/local_override.yaml`：当前服务器和新版 veRL 的共享必要差异；
- `main/eval/search_grpo/config.yaml`：默认组合以上两份配置，并选择官方 v0.2 AgentLoop。

```bash
bash main/eval/search_grpo/install.sh
bash main/eval/search_grpo/run.sh 1 searchr1_3b
bash main/eval/search_grpo/run.sh 1 searchr1_7b
```

仅在需要恢复 Ouro 实验时显式运行：

```bash
bash main/eval/search_grpo/run.sh 1 ouro_r3
```

训练 checkpoint、轨迹和日志都会继续按 `searchr1_3b`、`searchr1_7b`、`ouro_r3`
分目录，避免模型之间交叉恢复或混写结果。
运行脚本会检查 Retriever 健康状态及目标 GPU 空闲显存，不会主动占用已有任务的 GPU。

## 项目结构

```text
quick_loop/
├── configs/                  # 跨实验复用的 Hydra 配置
├── main/
│   ├── datasets/             # 外部原始数据集与语料
│   ├── data/                 # 项目生成的数据、轨迹与实验中间产物
│   ├── checkpoints/          # 本项目训练产生的权重
│   ├── models/               # 外部权重与推理模型视图
│   ├── eval/                 # 正式实验，代码/配置/结果按实验集中
│   ├── tests/                # 小型验证与诊断实验
│   ├── src/                  # 可被多个实验复用的主干实现
│   ├── scripts/              # 跨实验复用的公共入口
│   └── docs/                 # 项目共享文档
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .venv/
└── .gitignore
```

更详细的目录职责与新增文件规则见
[main/docs/project_structure.md](main/docs/project_structure.md)。

## 质量检查

```bash
uv lock --check
uv run ruff check .
```

新的工程验证按独立验证项目放入 `main/tests/`，正式 benchmark 则放入 `main/eval/`。
