# 项目结构约定

## 目标

`quick-loop` 使用固定且精简的骨干目录。目录只表达已经存在的职责，不为尚未发生的需求
预建多层结构。共享实现与单项实验分离，同一实验的入口、专用配置、结果和报告尽量相邻。

## 目录职责

| 目录 | 职责 | 不应存放 |
| --- | --- | --- |
| `configs/` | 跨多个实验复用的 Hydra 配置 | 单一实验专用配置、结果 |
| `main/datasets/` | 外部下载的原始数据集、语料和索引 | 轨迹、人工标注、处理后数据 |
| `main/data/` | 处理后数据、轨迹和实验中间产物 | 依赖缓存、运行时文件、外部模型、本项目 checkpoint |
| `main/models/` | 外部模型权重和由其生成的推理视图 | 本项目训练产生的权重 |
| `main/checkpoints/` | 本项目训练产生的权重 | Hugging Face 官方权重 |
| `main/eval/` | 正式 benchmark、训练实验、消融及报告 | 可跨实验复用的主干实现 |
| `main/tests/` | Smoke、机制验证、性能诊断和快速假设验证 | 正式科研结论 |
| `main/src/` | 可被多个实验复用的 Python 主干实现 | 数据、结果、报告 |
| `main/scripts/` | 跨多个实验复用的下载、准备和服务入口 | 单一实验的运行脚本 |
| `main/docs/` | 全项目共享的设计和流程文档 | 单项实验报告 |

Hugging Face 模型缓存固定在 `main/models/huggingface/`，datasets 的下载与 Arrow 缓存
固定在 `main/datasets/huggingface/`，两者不得共用目录。

## 新增内容规则

1. 新实验先在 `main/eval/<experiment>/` 或 `main/tests/<validation>/` 建立一个浅层目录。
2. 实验专用的 `run.py`、`config.yaml`、`results/` 和 `report.md` 放在该目录内。
3. 只有确认会被多个实验复用的实现，才提升到 `main/src/` 或 `main/scripts/`。
4. 外部模型进入 `main/models/`，训练生成的模型只能进入 `main/checkpoints/`。
5. 原始外部数据进入 `main/datasets/`，处理后数据和轨迹进入 `main/data/`。
6. 不为“以后可能需要”预建子目录；真实内容出现时再创建。

依赖缓存、第三方可编辑源码、编译头文件和 Ray 会话都属于本地环境，统一放在
`.venv/` 下并由 Git 忽略，不进入 `main/data/` 或 `main/src/`。

## 当前正式实验

`main/eval/search_grpo/` 集中了 Ouro R3 Search-R1 GRPO 的安装脚本、运行脚本、veRL
配置和兼容 patch。它产生的 checkpoint 和轨迹因数据类型约束分别进入
`main/checkpoints/search_grpo/` 与 `main/data/trajectories/search_grpo/`，实验日志和报告留在
实验目录的 `results/` 中。
