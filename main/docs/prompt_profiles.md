# Agent Prompt Profile 架构

`AgentRunner` 负责检索状态机、token 预算、从 0 开始的轮次调度和轨迹记录。
`PromptProfile` 只负责协议相关行为：

- 构造 system prompt 和 user prompt；
- 定义 Search 函数 schema；
- 构造完整模型上下文；
- 格式化 Retriever observation；
- 解析 action 和最终答案。

`AgentPrompt` 携带结构化对话消息、assistant 原始生成流的续写内容，以及 profile
专用的停止序列。这里有意区分两种上下文组织方式：Search-R1 将检索结果追加到同一条
assistant 生成流；Hermes 则对包含 `assistant` 和 `tool` 消息的对话重新应用 ChatML
模板。

通过 Hydra 配置组选择 `prompt_profile=search_r1`（默认）或
`prompt_profile=hermes`。新增协议只需实现 `PromptProfile` 并在注册表中增加一项，
无须复制 Agent 主循环。

## Search-R1 协议

该实现遵循官方
[Search-R1 推理代码](https://github.com/PeterGriffinJin/Search-R1/blob/main/infer.py)：
保留原始用户指令；模型 action 使用 `<think>`、`<search>` 和 `<answer>`；每次检索结果
都以 `<information>...documents...</information>` 追加到同一条生成流中。官方
`Doc N(Title: ...) ...` 段落渲染格式也保持不变。配置中的最多 4 次 Search 安全上限
由 prompt 外部的状态机控制。

## Hermes 协议

该实现遵循 Nous Research 的
[Hermes function-calling prompt](https://github.com/NousResearch/Hermes-Function-Calling/blob/main/prompt_assets/sys_prompt.yml)、
[解析器和递归循环](https://github.com/NousResearch/Hermes-Function-Calling/blob/main/functioncall.py)，
以及官方
[Hermes 2 Pro 工具使用消息示例](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B#prompt-format-for-function-calling)。
system 消息在 `<tools>` 中暴露 OpenAI 风格的 Search 函数 schema。assistant action
是在 `<tool_call>` 内的严格 JSON；Retriever 结果是在 `tool` role 消息中、
`<tool_response>` 内的 JSON。最终答案是普通 assistant 文本，而不是 `<answer>` 块。
无效 XML、无效 JSON、一次输出多个调用、未知工具，以及缺失或为空的 `query` 参数，
都会被判定为 malformed。
