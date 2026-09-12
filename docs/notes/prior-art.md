# Prior-art audit for the PL-first investigation

Reviewed 2026-09-12. This is a targeted source audit, not an exhaustive novelty search.

| Work | What it establishes | What it does not establish for this project |
|---|---|---|
| [DOMINO, ICML 2024](https://files.sri.inf.ethz.ch/website/papers/beurerkellner2024domino.pdf), sections 2, 3.6 and 4.1 | Studies template-induced tokenization changes and downstream task accuracy; proposes grammar-aware speculative token prediction. | Does not supply the proposed paired PL-batching evaluation of quantized Qwen3 tool calls in llama.cpp. |
| [SGLang compressed FSM](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/), 2024 | Implements jump-forward; explicitly describes changed subsequent logits and whole-text retokenization to address boundaries. | Its system comparisons do not isolate the projected gain on the pinned llama.cpp backend or establish quality equivalence for PL. |
| [XGrammar, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/file/5c20ca4b0b20b0bd2f1d839dc605e70f-Paper-Conference.pdf), appendix B | Supports and evaluates combination with jump-forward in SGLang. | Its reported speedup is not a benchmark of the proposed llama.cpp integration. |
| [llguidance 1.0.1 fast-forward documentation](https://github.com/guidance-ai/llguidance/blob/d795912fedc7d393de740177ea9ea761e7905774/docs/fast_forward.md) | Describes canonical tokenization, boundary suffix removal and maskbench checks; the local implementation exposes PL and uses it in ordinary masks. | Does not measure the selected BFCL/Hermes workload or prove absence of task-accuracy regression. |
| Guidance token healing, discussed in DOMINO section 2 and linked by llguidance | Treats the transition between fixed prompt/template text and generated tokens; DOMINO reports remaining effects despite boundary healing. | The original Guidance notebook has not been independently reproduced in this audit. |
| [llama.cpp discussion #5455](https://github.com/ggml-org/llama.cpp/discussions/5455) | Identifies tokenization ambiguity as an obstacle and suggests grammar-constrained drafting. | A 2024 discussion is not proof of current feature absence; the pinned engine source is the relevant evidence. |
| [llama.cpp issue #23184](https://github.com/ggml-org/llama.cpp/issues/23184) | Requests pipelining speculative strategies. At review it is closed as not planned with a stale label. | Does not establish that composition exists or has upstream commitment; stale closure alone is not technical rejection. |

## Revised research claim

Tokenization-related quality changes and jump-forward decoding are established topics. The broad
claim in the original proposal that no one has measured the problem is withdrawn. The candidate
contribution is a reproducible evaluation of the latency, workload coverage and quality tradeoffs
of native llguidance fast-forward in consumer-hardware llama.cpp tool calling, together with a
small engine integration if the measurements support it.

This is a candidate contribution, not a confirmed novel result or a promise of publication.
Engineering usefulness and research novelty require separate decisions. Batching PL and testing
PL against model-selected tokenization answer different questions.

## Remaining literature work

Before a publication claim, inspect the current implementations and evaluations most directly
overlapping any observed result, reproduce the relevant Guidance boundary example if used, and
extend the search to newer grammar-aware speculation work. Do not infer a lack of prior art from
the absence of a feature in llama.cpp.
