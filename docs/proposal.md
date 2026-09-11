# Grammar Jump-Forward Decoding for JSON Tool Calls in llama.cpp

**Eliminating forward passes for structurally determined tokens in constrained decoding**

Rafael Marins Florenzano
Draft proposal, September 2026

---

## 1. Executive summary

When llama.cpp generates a JSON tool call under a grammar constraint, a large fraction of the emitted tokens are not decisions. The opening brace, the quotes, the function name, the argument keys, the colons and the commas are fully determined by the automaton state before the model runs. llama.cpp executes a complete forward pass for each one anyway.

Decode is memory bandwidth bound. Each of those passes costs a full read of the model weights from VRAM to produce a token the grammar already knew. On an agentic workload where most output is tool calls, this is the dominant waste.

**Proposal:** detect deterministic spans in the grammar automaton, resolve them to a canonical token sequence, and inject them into the KV cache as a single batched pass instead of N sequential decode steps. Implement it inside llama.cpp's existing speculative decoding framework, as a proposer that costs zero inference.

**Reference point for the payoff:** SGLang's compressed finite state machine, which does the equivalent for regex-derived FSMs, reported up to 2x latency reduction and up to 2.5x throughput improvement against guidance plus llama.cpp and outlines plus vLLM, and made constrained decoding faster than unconstrained decoding. That was published in February 2024. llama.cpp still does not have it.

---

## 2. Why JSON tool calls specifically

Tool calling is the highest-value target for this optimization, for four reasons.

1. **Highest deterministic fraction.** A tool-call schema fixes the function name, the argument object keys, and all JSON punctuation. Only the argument *values* are free. For a typical tool with three short arguments, most of the token budget is structural.
2. **Latency is the metric that matters.** Agent loops are sequential: the user waits for tool call, tool execution, next tool call. Time to complete a tool call sits directly on the critical path, unlike batch generation where throughput dominates.
3. **Largest affected population.** GBNF-constrained tool calling is the mechanism behind Ollama, LM Studio, llama-cpp-python and the local agent stacks built on them. An upstream improvement propagates to all of them without any user action.
4. **Clean benchmark surface.** Tool-call datasets and schemas are public and standardized, so results are reproducible by third parties.

---

## 3. Prior art audit

### 3.1 What already exists

| System | Capability | Available in llama.cpp? |
|---|---|---|
| GBNF grammar sampler | Per-token mask via byte-level stack matcher, exact at codepoint boundaries | Yes, native |
| llguidance backend | Earley parser, mask in roughly 50 microseconds per token on a 128k vocabulary, negligible startup cost | Yes, merged in b4613 (February 2025), opt-in via `--grammar-backend llguidance`, requires a Rust toolchain to build |
| XGrammar | Adaptive token mask cache, context-independent precomputation, under 40 microseconds per token | No. Default backend in vLLM, SGLang and TensorRT-LLM |
| SGLang compressed FSM | Multi-token emission per step for deterministic spans | No |
| Speculative decoding, multi-strategy | `draft`, `draft-eagle3`, `draft-mtp`, `ngram-mod`, mixable | Yes, native |

An important correction to a common assumption: mask *computation speed* is largely solved for llama.cpp. llguidance is merged and fast. The remaining problem is not how quickly the mask is computed. It is that the engine still spends one full forward pass per token regardless of how constrained that token is.

### 3.2 What does not exist, and why

Discussion #5455 asks precisely whether deterministic tokens are skipped under grammar constraints. The answer given is that it is not implemented, and the stated blocker is tokenization ambiguity: a deterministic string such as `hello` can be tokenized as `["h","ello"]`, `["he","llo"]`, `["hel","l","o"]` and so on, so it is not knowable in advance which representation the model would have produced. The suggested workaround is to run a draft model under the same grammar. A follow-up question in the same thread asks whether one could simply pick the longest matching token and move forward. **Nobody answers it.** That unanswered question is the core of this proposal.

There is a second, structural blocker that is less often stated. llama.cpp's grammar constraint is implemented as a `llama_sampler`, and the sampler interface operates on the candidate array for a single position. Even where the underlying grammar engine can identify forced continuations (llguidance's design explicitly allows skipping the sampling step when the grammar uniquely determines the next token), a sampler has no way to inject multiple tokens or to suppress a forward pass. **The capability gap is in the engine's decode loop, not in the grammar engine.** Verifying exactly what the current llguidance integration exposes here is the first task of phase 0.

### 3.3 Adjacent work that does not close the gap

- **CFGzip** (arXiv:2605.29986) compresses the token search space offline, reporting up to two orders of magnitude latency reduction in the grammar engine and up to 7.5x on total constrained generation time. This attacks mask computation cost, not the number of forward passes.
- **Formatron / ZapFormat** (arXiv:2506.01151) reports 1.5x to 2x over prior engines via Earley state pruning. Same category: faster masks, same pass count.
- **Pre3** (arXiv:2506.03887) targets batch-scale validation bottlenecks in serving. Not the single-stream consumer case.

All of these make each step cheaper. None of them remove steps.

---

## 4. Design

### 4.1 The naive approach and why it fails

The tempting shortcut: at each step, compute the grammar mask; if exactly one token is allowed, accept it without running the model.

This almost never fires. Consider the grammar forcing the literal `"name": `. At that state the mask admits every token that is a viable prefix: `"`, `"n`, `"na`, `"name`, `"name"`, and so on. Determinism is a property of the *string* continuation, not of the token set. Any implementation must confront tokenization ambiguity directly rather than hoping to route around it.

### 4.2 Forced-span detection

Operate on the grammar automaton, not the token mask.

From the current parser state, walk the automaton over the alphabet. A span is *forced* if, from the current state, exactly one byte sequence can be produced up to some frontier state. In GBNF terms this is a chain of states each having a single viable outgoing transition. In JSON schema terms, forced spans correspond to structural literals: `{"`, the key text, `":`, separators, closing delimiters, and any `const` or single-variant `enum` value.

Output of this stage: a byte string `S` that the model has no choice but to produce, and the parser state reached after consuming it.

Termination conditions for the walk: a branch point (more than one viable byte), a free region (a string value, a number), or a configurable maximum span length.

### 4.3 Canonical tokenization policy

Given forced byte string `S`, choose the token sequence to inject. Three candidate policies, to be evaluated rather than assumed:

- **P1, greedy longest-match.** Apply the tokenizer's own longest-match segmentation to `S`. Cheapest, and matches the tokenizer's natural output for that substring in isolation.
- **P2, boundary-aware greedy.** Same as P1, but the final token of `S` is held back if it could merge with the first bytes of the following free region. Prevents the classic token-healing failure where the forced span steals bytes that belong to the model's next choice.
- **P3, model-preferred prefix.** Run one forward pass, take the highest-probability token among those consistent with `S`, consume it, and re-segment the remainder greedily. Costs one pass per span but starts each span on the token the model would have picked.

P2 is the expected default. P1 is the baseline. P3 is the safety net if quality measurements come out badly.

The last token of a forced span is the dangerous one. Where `S` ends immediately before a free region, the boundary must not be resolved by the tokenizer alone. This is the same boundary mismatch problem the literature raises for character-level versus token-level automata, and it is where an implementation is most likely to be subtly wrong.

### 4.4 Injection path

Do not write a new batch-injection mechanism. llama.cpp's speculative decoding framework already accepts multiple tokens per decode and already supports zero-inference-cost proposers: `ngram-mod` drafts purely from pattern matching over the token history, with no model call.

Implement forced spans as a new strategy, `--spec-type grammar`:

1. Grammar proposer emits the forced token sequence as a draft.
2. The existing verification path submits the tokens in a single batch and populates the KV cache.
3. Verification is trivially satisfied: the grammar mask leaves the target no alternative at those positions, so every forced token is accepted by construction.
4. The same pass produces the logits for the first free position, so the free token comes at no additional cost.

Two consequences worth stating explicitly. First, this reuses tested plumbing instead of introducing a parallel code path, which materially raises the odds of upstream acceptance. Second, it composes with existing strategies. Issue #23184 already asks for speculative strategies to be pipelined rather than run independently; grammar plus a draft model is exactly that pattern, and the maintainers' own suggested workaround in #5455 was a grammar-constrained draft model.

### 4.5 Server-level details that will bite

- **Streaming.** Injected tokens must be emitted to the client in order, without appearing as a single burst that breaks incremental JSON parsers downstream.
- **Stop conditions and token limits.** A forced span can overshoot `n_predict`. Truncation mid-span must leave the grammar state coherent.
- **Slot state and prompt cache.** Injected tokens participate in the KV cache like any other. Interaction with `--cache-ram` prefix restoration needs a test.
- **Per-slot sampler state.** Grammar state advances across the whole span at once; penalties and other stateful samplers must see the same token history they would have seen otherwise.
- **Probability reporting.** Requests that ask for logprobs will have positions with no sampled distribution. Report the forced status explicitly rather than fabricating a value.

---

## 5. Correctness and the research question

### 5.1 Structural correctness

Output remains grammar-valid by construction: forced tokens come from the automaton itself. This is straightforward to verify with an independent recognizer. Note that an external audit already found a real soundness hole in llama.cpp's existing mask for character classes mixing codepoints of different UTF-8 byte lengths, where the decoder can be led to emit invalid UTF-8 and signal end of sequence as if it were valid. Any new work in this area should include a differential test against an independent byte-level recognizer, both to avoid inheriting that class of bug and to prove the new path does not add one.

### 5.2 The open question nobody has answered

Forcing a tokenization the model would not have chosen leaves the model in an off-distribution KV state. Every system that ships jump-forward decoding assumes this is benign. No published measurement supports that assumption.

**RQ1. Does forced canonical tokenization degrade the quality of the free regions that follow the deterministic span?**

Argument-value correctness is the natural measurable: if the KV state after `{"city": ` is subtly wrong because the key was tokenized unnaturally, the model may produce a worse value. This is directly testable and directly matters for tool calling.

**RQ2. How does the effect vary with policy (P1, P2, P3) and with tokenizer?**

Byte-level BPE vocabularies with heavy merge tables (Qwen, Llama 3) should be more sensitive than smaller vocabularies, because there are more alternative segmentations of the same string.

**RQ3. What is the deterministic-token fraction in real tool-call traffic?**

The upper bound on speedup is a property of the workload, not of the implementation. Nobody has published this distribution for tool-call schemas. It is the cheapest useful result in this project and gates everything else.

**RQ4. Does grammar jump-forward compose additively with draft-model speculation, or do they compete?**

They consume the same speedup budget: both remove sequential forward passes. If the draft model already predicts structural tokens reliably (structural JSON is highly predictable, so acceptance should be high), the marginal gain from jump-forward may be small. This is a genuine risk to the project's value proposition and needs measuring, not assuming.

RQ1 and RQ4 are the parts that make this a paper rather than a pull request.

---

## 6. Evaluation

### 6.1 Metrics

| Metric | Why |
|---|---|
| Deterministic token fraction, per schema | Theoretical ceiling on speedup |
| Forward passes per tool call | Direct measure of the mechanism |
| Time to complete tool call | What agent users feel |
| Tokens per second, end to end | Comparable to existing llama.cpp benchmarks |
| Grammar validity rate | Must stay at 100 percent |
| Argument-value accuracy | RQ1, the quality question |
| Mask computation time per step | Isolates grammar engine cost from pass cost |

### 6.2 Conditions

Full factorial over: grammar backend (GBNF, llguidance) times jump-forward (off, P1, P2, P3) times speculation (off, draft model) at fixed temperature 0. Temperature must be pinned, since sampling temperature independently affects speculative acceptance and would confound the comparison.

### 6.3 Workloads

1. Public tool-call benchmark schemas, for reproducibility by third parties.
2. Multi-tool agent traces, where schemas repeat across turns and prompt caching interacts.
3. A high-determinism control: a schema of all `enum` and `const` fields, establishing the upper bound.
4. A low-determinism control: a single free-text argument, establishing the floor.

### 6.4 Hardware

Development on RTX 2080 Ti, 11 GB, Nobara Linux. The contribution is control flow and CPU-side logic, not kernel work, so a consumer card is sufficient. Model scale should stay in the 4B to 9B range so that the full factorial is affordable. Reporting a CPU-only configuration in addition is worthwhile, since the relative gain from removing forward passes is larger where each pass is slower.

---

## 7. Go/no-go gates

**Gate 0, before writing any C++ (one to two days).**
Instrument the grammar automaton offline and measure the deterministic byte fraction across a corpus of real tool-call schemas. Convert to a token-fraction estimate under P1.

Threshold: if fewer than 25 percent of output tokens fall inside forced spans on realistic tool-call traffic, the ceiling is too low and the project should be abandoned or retargeted to a higher-determinism DSL.

**Gate 1, after the prototype (four to six weeks).**
Threshold: at least 1.5x reduction in time to complete a tool call versus the same grammar backend without jump-forward, at 100 percent grammar validity, with no measurable regression in argument-value accuracy.

**Gate 2, composition (two weeks after Gate 1).**
Measure jump-forward with and without a draft model. If the marginal gain over draft-model speculation alone is under 15 percent, the honest conclusion is that draft models already capture most of the available win, and the contribution should be reframed as the RQ1 quality study rather than as a performance feature.

Gate 2 is the one most likely to fail. It should be scheduled early rather than deferred.

---

## 8. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Draft-model speculation already captures most of the gain (RQ4) | **High** | Gate 2, scheduled early. Reframe as quality study if it fails |
| Deterministic fraction too low on real workloads | High | Gate 0, costs two days |
| Token boundary handling introduces subtle invalid output | High | Differential testing against an independent byte-level recognizer, per section 5.1 |
| Quality regression in free regions (RQ1) | Medium | P3 policy as fallback; the measurement is itself the deliverable |
| Upstream resistance to touching the sampler and server loop | Medium | Implement as a speculative strategy behind a flag, default off, no change to existing paths |
| llguidance already exposes forced tokens and the integration is closer than assumed | Medium | Phase 0 task. If true, the work shrinks to wiring plus the quality study, which is still publishable and still a merged feature |
| Scope creep into a general structured-output engine | Medium | Stay inside llama.cpp. Do not attempt to compete with XGrammar on mask computation |

---

## 9. Roadmap

| Phase | Duration | Output |
|---|---|---|
| 0. Prior-art verification and Gate 0 | 1 week | Deterministic-fraction measurement; exact status of llguidance forced-token support in the current integration |
| 1. Offline forced-span extractor | 2 weeks | Standalone tool: schema in, forced spans and token estimates out. Useful on its own |
| 2. Prototype in the speculative framework | 4 weeks | `--spec-type grammar`, P1 policy, single-slot only |
| 3. Boundary policies and correctness | 3 weeks | P2 and P3, differential recognizer tests |
| 4. Server integration | 2 weeks | Streaming, stop conditions, prompt cache, multi-slot |
| 5. Full evaluation and Gate 2 | 3 weeks | Factorial results, RQ1 through RQ4 |
| 6. Upstream PR and writeup | 3 weeks | PR against llama.cpp, paper draft |

Approximately 18 weeks of effective work.

---

## 10. Upstream strategy

llama.cpp maintainers are conservative about the sampler and server loop, correctly. Three things improve the odds:

1. **Land phase 1 separately.** An offline tool that reports the deterministic fraction of a schema is independently useful, uncontroversial, and establishes credibility before proposing anything invasive.
2. **Reuse the speculative framework rather than adding a parallel path.** This is the single most important design decision in the proposal.
3. **Default off, behind a flag, with benchmarks in the PR.** Note the llguidance precedent: it took from PR #10224 in November 2024 to merge in b4613 in February 2025, and part of the friction was the Rust build dependency. A native C++ implementation with no new dependencies avoids the main objection that slowed that work.

---

## 11. Fit with the broader research direction

- **Continuity with the TCC.** The compact DSL work is the same problem one level up. A DSL designed with high automaton determinism is one that jump-forward decoding accelerates most, which suggests a follow-on question: how should an output language be *designed* so that constrained decoding is fast?
- **Composition with draft-model adaptation.** The companion proposal on adapting draft models to quantized targets and to specific domains intersects here through RQ4. Whether the two techniques are additive or redundant is a real open question and either answer is worth publishing.
- **PhD-scale extension.** Joint optimization of the output language and the decoding strategy: given a task, co-design the schema and the speculation policy to minimize sequential forward passes subject to expressiveness constraints. That is a coherent multi-year program, and this project is its first concrete piece.

---

## 12. References

- Are deterministic tokens skipped when using grammar constraints? llama.cpp Discussion #5455: https://github.com/ggml-org/llama.cpp/discussions/5455
- Fast JSON Decoding for Local LLMs with Compressed Finite State Machine, LMSYS: https://www.lmsys.org/blog/2024-02-05-compressed-fsm/
- llguidance: https://github.com/guidance-ai/llguidance
- llguidance support in llama.cpp, PR #10224: https://github.com/ggml-org/llama.cpp/pull/10224
- llama.cpp speculative decoding documentation: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- Pipeline speculative decoding strategies, Issue #23184: https://github.com/ggml-org/llama.cpp/issues/23184
- GBNF documentation and known performance gotchas: https://github.com/ggml-org/llama.cpp/tree/master/grammars
- grammarexact, independent audit of llama.cpp's GBNF mask soundness: https://github.com/v-code01/grammarexact
- CFGzip, Accelerating Constrained Decoding with Token Space Compression: https://arxiv.org/html/2605.29986
- Formatron, Earley-Driven Dynamic Pruning for Efficient Structured Decoding: https://arxiv.org/pdf/2506.01151
- Pre3, Deterministic Pushdown Automata for Faster Structured LLM Generation: https://arxiv.org/pdf/2506.03887
