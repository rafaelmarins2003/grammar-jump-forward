# Baseline metadata corrections found during prototype development

The completed baseline measurements and raw responses are preserved. Two labels in the old report
were incorrect; do not use them as descriptions of what ran.

- `stock` sent `json_schema` to the llguidance-enabled server. At the pinned commit,
  `common/json-schema-to-grammar.cpp::json_schema_to_grammar` returns `%llguidance` unless
  `force_gbnf` is explicitly true. The response `generation_settings.grammar` confirms this route.
  Thus this was **default-schema llguidance**, not native GBNF. The earlier source audit followed
  the sampler's native branch but missed the preceding schema conversion. The measured comparison
  against compact llguidance remains real; it must not be described as a native-backend comparison.
- The n-gram run passed `--spec-draft-n-max 8`, but that setting controls the draft-model field.
  `ngram-simple` takes its cap from `ngram_simple.size_m`, whose unchanged default is 48 in
  `common/common.h`; `common_speculative_n_max` confirms that dispatch. The fixed n-gram run was
  not an eight-token-cap experiment. No result is evidence against every possible n-gram tuning.

These corrections do not turn the previous approximate jump-forward projections into measurements.
The prospective prototype comparison remains identical compact llguidance with batching off/on;
it does not rely on either incorrect label. Native GBNF fallback is checked separately by sending
an explicit GBNF grammar string in the prototype smoke tests.

The first completed smoke screen deliberately treated a `json_schema` request as native and
flagged unexpected drafting. Its raw response exposed the incorrect assumption. That failed
instrument run is retained; the corrected development test sends actual GBNF. Held-out evaluation
had not started, and no evaluation question or performance threshold was changed in response.
