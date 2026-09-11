# Original Ouro R3 Zero-Shot Agent Evaluation

This is an inference-only comparison of the untrained-for-search
`ByteDance/Ouro-2.6B-Thinking` snapshot at revision
`f1edd81e7ac41355db670500ceaf204e0f73af68`. The model ran at fixed R3 in BF16 through
vLLM. No optimizer was created, the original weight file size and modification time were
unchanged, and no Search-R1/GRPO checkpoint was loaded.

The evaluation uses the same 20 IDs for both profiles: the first 10 NQ and first 10
HotpotQA rows in `benchmarks/zero_shot_agent_v1.jsonl`. Both runs use greedy decoding,
temperature 0, top-p 1, 500 new tokens per generation, a 3000-token aggregate response
budget, at most four searches, and top-k 3. Retrieval uses the 21,015,324-passage Wiki18
FP32 `GpuIndexFlat` plus E5 on a separate A800.

## Metrics

| Metric | Search-R1 | Hermes |
| --- | ---: | ---: |
| Strict format-valid rate | 0/20 (0%) | 19/20 (95%) |
| Valid action rate, per generation | 10/20 (50%) | 31/32 (96.875%) |
| Valid Search/tool action rate, per sample | 0/20 (0%) | 12/20 (60%) |
| Actual Retriever invocation rate | 0/20 (0%) | 12/20 (60%) |
| Valid continuation after observation | N/A (0 observations) | 12/12 (100%) |
| Mean Search turns | 0.0 | 0.6 |
| Multi-turn Search rate | 0% | 0% |
| Four-turn-limit rate | 0% | 0% |
| Parser-level final answer rate | 10/20 (50%) | 19/20 (95%) |
| Exact match | 1/20 (5%) | 0/20 (0%) |

Search-R1 strict format validity requires a closed `<think>` block followed by exactly
one closed `<search>` or `<answer>` action. Hermes accepts either one valid Search JSON
object inside `<tool_call>` or ordinary assistant text as its final response. The
post-observation metric requires the next generation to parse as a valid action.

The Hermes final-answer number is a protocol parser result, not a completion-quality
claim. Fourteen of its 19 parser-level final responses re-tokenize to 496--498 tokens
against the 500-token generation cap and visibly end mid-sentence. Only 5 of the 12
post-observation responses are below that near-cap threshold. This truncation diagnostic
does not change the observed 12 valid tool calls or 12 real Retriever invocations.

By dataset, Search-R1 searched 0/10 for both NQ and HotpotQA. Hermes searched 7/10 NQ
and 5/10 HotpotQA. Search-R1 EM was 1/10 NQ and 0/10 HotpotQA; Hermes strict EM was zero
for both. As a non-EM diagnostic, a normalized non-binary answer alias appeared somewhere
in 7 of 19 Hermes predictions, showing that zero EM partly reflects verbose/truncated
answers rather than zero factual signal.

## Representative Cases

- `nq-test-1022`, Search-R1: generated 498 re-tokenized tokens of untagged reasoning and
  stopped mid-sentence. It emitted neither `<search>` nor `<answer>`.
- `nq-test-1371`, Search-R1: directly produced `<answer>Rome, Italy</answer>` but omitted
  the required `<think>` block. The action was parseable, but the full protocol was not.
- `nq-test-1022`, Hermes: issued the valid query `Las Vegas Super Bowl Stadium location`,
  retrieved evidence containing the gold location, and continued after the tool response.
  Its verbose final response did not exactly match `Paradise, Nevada`.
- `hotpotqa-test-1815`, Hermes: issued a syntactically valid but off-track query,
  `The Clash formation location`, after incorrectly identifying Howard Devoto's band.
- `hotpotqa-test-1212`, Hermes: attempted a relevant tool call but hit the generation
  limit after the JSON arguments and before the `name` and closing XML tag. This was the
  only Hermes protocol failure.

## Interpretation

Original Ouro has basic tool-use capability when prompted with Hermes: 60% of samples
made valid calls, every valid call reached the real Retriever, and every observation was
followed by another parseable generation. It does not exhibit active Search-R1 behavior
zero-shot: there were no Search actions in 20 samples, including all 10 multi-hop items.

The large protocol gap means the previously low Search-R1 action rate is not evidence
that Ouro is categorically unable to call tools. Protocol familiarity is a major factor.
The zero strict EM, no multi-search behavior, off-track entity assumptions, and frequent
length truncation also show that a useful multi-hop search policy has not formed. For the
next short GRPO comparison, Hermes is the stronger zero-shot initialization; Search-R1
should remain as a separate compatibility baseline rather than the only prompt condition.

Machine-readable records, summaries, raw generations, complete retrieved documents, and
per-sample trajectories are under `outputs/evaluations/zero-shot-agent-v1-20/`.
