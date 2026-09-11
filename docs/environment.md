# Environment

Verified 2026-09-10 on the development machine. Machine-readable pins live in
`config/llama_cpp.json`, `config/models.json` and `data/raw/MANIFEST.json`. This file explains
how to rebuild the environment from those pins.

## Hardware and system

| Item | Value |
|---|---|
| GPU | RTX 2080 Ti, 11,264 MiB, Turing (`sm_75`), driver 610.57.04 |
| CPU / RAM | i9-9900K (16 threads), 23 GB |
| Disk | ~640 GB free on `/home` |
| CUDA / GCC / CMake | 13.3 / 16.2.1 / 4.4.3 |
| clang-format, clangd | system packages (`/usr/bin`) |
| uv / Python | 0.12.7 / 3.12 for this project (`.python-version`) |

## Toolchain pinned with mise

`mise.toml` pins `rust 1.98.1` (needed to build llguidance), `ninja 1.13.2` and `ccache 4.14`.

```bash
mise install
mise exec -- cargo --version   # cargo 1.98.1
```

## llama.cpp

Two checkouts at the same base commit, `95ef7fc16` (b10791):

| Path | Role |
|---|---|
| `~/llama.cpp` (branch `master`) | Untouched reference. Shared with other projects. `LLAMA_LLGUIDANCE=OFF`. No-patch baseline binaries |
| `~/llama.cpp-jf` (branch `grammar-jump-forward`) | Git worktree of the same repo. All C++ changes for this project go here |

The worktree was created with:

```bash
git -C ~/llama.cpp worktree add -b grammar-jump-forward ~/llama.cpp-jf 95ef7fc16054e63b427a3ef00188e055ef7586d8
```

### Builds in `~/llama.cpp-jf`

Both builds fetch and compile llguidance v1.0.1 (`d795912f`) with cargo, so run them under mise.
Both export `compile_commands.json` for clangd.

```bash
cd ~/llama.cpp-jf

# Release, CUDA sm_75: benchmarks and GPU runs
mise exec rust@1.98.1 ninja@1.13.2 ccache@4.14 -- cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DLLAMA_LLGUIDANCE=ON -DLLAMA_OPENSSL=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
mise exec rust@1.98.1 ninja@1.13.2 ccache@4.14 -- cmake --build build -j 14

# Debug, CPU only: stepping through sampler / speculative / server control flow with gdb
mise exec rust@1.98.1 ninja@1.13.2 ccache@4.14 -- cmake -S . -B build-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DGGML_CUDA=OFF -DLLAMA_LLGUIDANCE=ON -DLLAMA_OPENSSL=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
mise exec rust@1.98.1 ninja@1.13.2 ccache@4.14 -- cmake --build build-debug -j 14
```

Logs from the initial build: `~/llama.cpp-jf/build-release.log`, `~/llama.cpp-jf/build-debug.log`.

For clangd, `~/llama.cpp-jf/compile_commands.json` is a symlink to `build/compile_commands.json`.
llama.cpp's `.gitignore` doesn't cover it, so it's excluded in the shared
`~/llama.cpp/.git/info/exclude` and can't slip into a patch.

Release build verified 2026-09-10:
- `llama-server --version` reports `build 10791, commit 95ef7fc16`
- `libllama-common.so` exports 44 `llg_*` symbols
- `test-grammar-llguidance` is built

Debug build verified 2026-09-10:
- `build-debug/bin/llama-server --version` reports `build 10791, commit 95ef7fc16`
- 44 `llg_*` symbols exported, 120 binaries built

### Exporting the C++ work into this repo

The branch lives in the llama.cpp repository. At each milestone, export it so this repo records
exactly what was measured:

```bash
git -C ~/llama.cpp-jf format-patch 95ef7fc16..HEAD -o ~/scripts/adaptation-toolkit/patches/<milestone>/
```

Run manifests must record the branch HEAD commit, not only the base commit.

## Python

```bash
uv sync
uv run prek install
```

- `llguidance` is pinned to **1.0.1**, the same version llama.cpp builds, so offline measurements
  and the engine use the same grammar implementation.
- No PyTorch: nothing in Phase 0–1 needs it. (PyTorch 2.14.0+cu130 was verified on this card for the
  shelved draft-adaptation idea: `sm_75` present, fp16 matmul OK.)

## Data

Pinned in `data/raw/MANIFEST.json` with repo, revision and per-file sha256. To re-fetch:

```bash
hf download gorilla-llm/Berkeley-Function-Calling-Leaderboard --repo-type dataset \
  --revision 61fc0608cfd831fcfbbaa676ebdfef0ed963eeda --local-dir data/raw/datasets/bfcl
hf download NousResearch/hermes-function-calling-v1 --repo-type dataset \
  --revision dae3e1d28cfbcf4b915c04ea1e072030529b4bda --local-dir data/raw/datasets/hermes-fc-v1
```

Tokenizers (pass each `--include` separately; a space-separated list after one `--include` is
parsed as positional filenames and silently skips `tokenizer.json`):

```bash
INC=(--include 'tokenizer*' --include '*chat_template*' --include special_tokens_map.json \
     --include added_tokens.json --include generation_config.json --include config.json)
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir data/raw/tokenizers/qwen3 "${INC[@]}"
hf download unsloth/Llama-3.1-8B-Instruct --revision 4699cc75b550f9c6f3173fb80f4703b62d946aa5 --local-dir data/raw/tokenizers/llama-3.1 "${INC[@]}"
hf download unsloth/mistral-7b-instruct-v0.3 --revision 60b99dcbc7f853b7cd18472a607e8d045b96f03e --local-dir data/raw/tokenizers/mistral-v0.3 "${INC[@]}"
hf download unsloth/gemma-3-4b-it --revision bf46152c47f5dd20b896357cb51abc4c03b8ee8c --local-dir data/raw/tokenizers/gemma-3 "${INC[@]}"
```

Not fetched: `Salesforce/xlam-function-calling-60k` (gated) and `glaiveai/glaive-function-calling-v2`
(271 MB, older format; a 5k subset already ships inside hermes-fc-v1).

## Models

Local GGUFs with sha256 are in `config/models.json`: Qwen3-8B and Qwen3-4B Q4_K_M as targets,
Qwen3-0.6B and Qwen3-1.7B Q8_0 as RQ4 drafts. GGUFs for the other RQ2 tokenizer families
(Llama 3.1, Mistral v0.3, Gemma 3) are not downloaded yet; pick exact repos when RQ2 starts.
