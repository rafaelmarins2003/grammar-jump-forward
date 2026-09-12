# llguidance native fast-forward policy

Source audit: 2026-09-12. Engine `95ef7fc16054e63b427a3ef00188e055ef7586d8`,
llguidance `d795912fedc7d393de740177ea9ea761e7905774` (1.0.1).
These are source findings, not experimental quality or latency results.

## Finding and consequence

The current backend already imposes its native canonical fast-forward tokenization through the
ordinary per-token mask. An engine optimization that batches those same tokens can target the
existing policy. Calling it a new tokenization intervention relative to default llguidance would
confound the experiment.

In `parser/src/tokenparser.rs:449`, `compute_mask_inner()` calls `ff_tokens()` when forcing is
enabled. If tokens are available, it returns `singleton_token_set(ff_tokens[0])`. The default
`no_forcing` option is false (`parser/src/api.rs:35`). With `no_forcing`, tokenization is left to
the model; this is the relevant alternative policy, not default llguidance without batching.

The llama.cpp integration (`common/llguidance.cpp:44`) computes/applies that mask and consumes
individual accepted tokens. It does not call the fast-forward or multi-token consumption API.
Its tokenizer callback delegates to `llama_tokenize`, and the C tokenizer regards a supplied
callback as canonical (`parser/src/ffi.rs:73`).

## Policy and boundaries

`ff_tokens()` (`parser/src/tokenparser.rs:663`) prepends the most recent token's decoded bytes,
appends the forced bytes, and calls the canonical tokenizer. If this changes the already emitted
token, it retries without that prefix. It then calls `tokenize_and_chop()` and removes the old
token from the returned sequence. It never authorizes changing already committed token IDs.

`tokenize_and_chop()` (`parser/src/tokenparser.rs:196`) consults the token trie and grammar
recognizer to remove a suffix that could participate in a different tokenization at the boundary.
This suffix can include more than one token. Residual bytes constrain the next sampled token.
This is not equivalent to longest-match vocabulary segmentation, nor to always withholding
exactly one token.

The return can be empty at a grammar branch, when all available bytes belong to a boundary suffix,
when no bytes can be forced, or when the tokenizer cannot provide a canonical tokenization.
`Matcher::compute_ff_tokens()` also returns an empty vector on an error, so checking the matcher's
error state is mandatory: an empty vector is not by itself evidence of a legitimate free region.

The upstream [fast-forward explanation](https://github.com/guidance-ai/llguidance/blob/d795912fedc7d393de740177ea9ea761e7905774/docs/fast_forward.md)
documents token healing, including examples where retaining only one boundary token is
insufficient. Its maskbench observations are not a measurement of this project's tool-call corpus.

## Speculative acceptance

The existing llama.cpp verifier compares draft IDs with target-sampled IDs
(`common/sampling.cpp:678`). It stops at the first mismatch. A byte-forced string can have several
valid token representations; consequently a grammar-valid arbitrary draft does not have guaranteed
acceptance. PL is a narrower case: where the backend already forces its first token with a
singleton mask, the draft can agree by following exactly that policy. Sequential-mask agreement
across a full proposed span still needs a characterization experiment, including boundaries.

Numerical differences between single-token and batch execution, sampler bookkeeping, triggers,
stopping and KV state remain integration concerns. Source inspection is not an end-to-end proof.

## Lazy grammar scope

The llguidance matcher consumes a compiled grammar; lazy transitions can be represented in its
Lark grammar using lazy lexemes. The llama.cpp `%llguidance` path constructs this matcher directly
(`common/sampling.cpp:213`), while native GBNF lazy trigger handling is in the other branch.
The two mechanisms must not be assumed interchangeable.

Before a tool trigger, unconstrained text must remain free. After the trigger, only the actual
remaining forced bytes may be counted. Offline JSON-body results start after activation and
therefore do not measure savings across a trigger. A dynamic trigger check remains outstanding.

## Required measurements

- Compare each PL span with sequential singleton masks, and retain token IDs and mismatch states.
- Check `no_forcing` as a distinct tokenization policy under the same grammar.
- Exercise optional keys, common tool-name prefixes, free values, UTF-8 boundaries and termination.
- Verify native tokenizer parity with the GGUF tokenizer before making engine claims.

Decision from source audit: use PL as the initial candidate and separate batching from policy
changes. No Gate 0 decision follows from this audit alone.

## Experimental follow-up — 2026-09-12

The [Gate 0 report](../gate0-report.md) now records strict PL replay with sequential-mask agreement,
literal/Unicode/lazy-lexeme characterization, and GGUF/HF body-token parity. The lazy test keeps
unconstrained text free until the terminator is consumed; this is not a test of real server chat
triggers. A naive empty-matching lazy lexeme was rejected during instrument development and was
replaced with a terminator-bearing lexeme as prescribed in llguidance `docs/syntax.md`.

The generated-output check replays raw sampled IDs. Its noncanonical whole-text segmentations
reinforce why a completed response must not be silently re-tokenized to measure the execution.
`no_forcing` was characterized as a different mask policy, not evaluated for task quality.
Real batching, numerical divergence, KV bookkeeping and native GBNF compatibility remain untested.
