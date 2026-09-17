# Search-R1 v0.2 主实验

该目录集中保存 Search-R1 v0.2 主实验入口、实验专用配置和 veRL 兼容补丁。主实验模型为
Search-R1 官方发布的 Qwen2.5-3B/7B GRPO v0.2 权重，默认使用 3B。Ouro R3 暂时放在
独立配置中，不参与默认主实验。

## 文件

- `config.yaml`：实际传给 veRL 的 Hydra 配置，默认组合官方 baseline 与本机 override。
- `experiment_model/searchr1_3b.yaml`：默认 Search-R1 3B 模型及独立输出路径。
- `experiment_model/searchr1_7b.yaml`：Search-R1 7B 模型及独立输出路径。
- `experiment_model/ouro_r3.yaml`：保留的 Ouro R3 配置，必须显式选择。
- `agent_loop_search_r1_v0_2.yaml`：正式主实验使用的官方 v0.2 rollout 适配。
- `agent_loop_r3.yaml`：保留的 Search-R1-style 通用 Agent，不作为默认入口。
- `verl-ouro-r3.patch`：固定 veRL commit 上的 Ouro/vLLM 兼容改动。
- `install.sh`：通过国内镜像安装训练依赖，并准备 `.venv/src/verl/` 工作树。
- `run.sh`：运行 1、5 或 20 step 验收；结果日志运行时写入 `results/`。

## 固定版本与路径

- Search-R1 baseline commit：`598e61bd1d36895726d28a8d06b3a15bed19f5d3`
- veRL commit：`ddd86f527a4af75095e4677b02b5aa272913a088`
- Search-R1 3B revision：`7ff32234abe1e1bf1dbbbf3a1385686b52ecfea2`
- Search-R1 7B revision：`deaa9d14b92dd4d481414e77b9f733c934df830a`
- Search-R1 权重：`main/models/searchr1/`
- Ouro R3 保留配置：`experiment_model/ouro_r3.yaml`
- 训练数据：`main/data/searchr1/`
- checkpoint：`main/checkpoints/search_grpo/`
- 轨迹：`main/data/trajectories/search_grpo/`

第三方 veRL 源码、uv 缓存、Python 编译头文件和 Ray 会话都属于本地环境，统一位于
`.venv/`，不进入项目源码或实验数据目录。

## 配置边界

`configs/search_r1/v0_2.yaml` 只记录官方配置：v0.2 Prompt、4 个主循环 turn、2048
起始长度、4096 rolling context、每次生成 500、observation 500、temperature/top-p 1、
top-k=3、strict parser 和 invalid-action recovery。

`configs/search_r1/local_override.yaml` 只记录所有模型共享的本机或当前框架适配：
Wiki18/E5/FAISS 路径、单 GPU FP32 FAISS、PyTorch/vLLM/veRL 版本，以及 veRL 0.6
用一个字段承载完整多轮 response 所需的 4096 累积空间。模型路径、remote-code、
Ouro loop 参数和输出目录只存在于各自 `experiment_model` 文件中。

官方 v0.2 使用旧 vLLM 的 XFORMERS backend；当前 vLLM 0.10/V1 配置继续使用
FLASH_ATTN。这项运行时差异记录在官方 baseline 与各模型配置中，没有混入协议实现。

## 运行

```bash
bash main/eval/search_grpo/install.sh
bash main/eval/search_grpo/run.sh 1 searchr1_3b
bash main/eval/search_grpo/run.sh 1 searchr1_7b
```

第二个参数省略时默认 `searchr1_3b`。需要暂时恢复 Ouro 时使用
`bash main/eval/search_grpo/run.sh 1 ouro_r3`。三个配置分别写入自己的 checkpoint、
trajectory 和日志目录，不会自动跨模型恢复。

脚本只允许使用国内 PyPI/Hugging Face 镜像，并在启动训练前检查 Retriever 与目标 GPU。
