# Ouro R3 Search GRPO

该目录集中保存 Ouro R3 的 Search-R1 风格 GRPO 实验入口、实验专用配置和 veRL
兼容补丁。共享模型、Agent、Retriever 和 reward 实现仍放在 `main/src/ouro_search/`。

## 文件

- `config.yaml`：实际传给 veRL 的 Hydra 配置，response budget 为 3000。
- `agent_loop_r3.yaml`：多轮 Search Agent loop 配置，最多调用 4 次 Retriever。
- `verl-ouro-r3.patch`：固定 veRL commit 上的 Ouro/vLLM 兼容改动。
- `install.sh`：通过国内镜像安装训练依赖，并准备 `.venv/src/verl/` 工作树。
- `run.sh`：运行 1、5 或 20 step 验收；结果日志运行时写入 `results/`。

## 固定版本与路径

- veRL commit：`ddd86f527a4af75095e4677b02b5aa272913a088`
- Ouro Thinking revision：`f1edd81e7ac41355db670500ceaf204e0f73af68`
- 模型视图：`main/models/vllm-ouro-r3/`
- 训练数据：`main/data/searchr1/`
- checkpoint：`main/checkpoints/search_grpo/`
- 轨迹：`main/data/trajectories/search_grpo/`

第三方 veRL 源码、uv 缓存、Python 编译头文件和 Ray 会话都属于本地环境，统一位于
`.venv/`，不进入项目源码或实验数据目录。

## 运行

```bash
bash main/eval/search_grpo/install.sh
bash main/eval/search_grpo/run.sh 1
bash main/eval/search_grpo/run.sh 5
bash main/eval/search_grpo/run.sh 20
```

脚本只允许使用国内 PyPI/Hugging Face 镜像，并在启动训练前检查 Retriever 与目标 GPU。
