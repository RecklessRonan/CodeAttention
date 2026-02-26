# AGENTS.md

## Cursor Cloud specific instructions

### Project overview
CodeAttention is a Python ML research project for studying attention patterns in pre-trained code models (CodeBERT, GraphCodeBERT, CodeT5, etc.) across code intelligence tasks (summarize, translate, refine, concode, clone, defect). It is a single-purpose research repo with no external services (no databases, queues, or APIs).

### Dependencies
- **Python 3.12** with `pip install -r requirements.txt`
- Requires `transformers>=4.20,<4.40` (older versions needed for `AdamW` import in `main.py`)
- CPU-only PyTorch is sufficient for development/testing; install via `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- `tree-sitter==0.20.4` with language version 14 grammars (v0.20.x tags from tree-sitter grammar repos)

### System dependencies for tree-sitter builds
Building tree-sitter `.so` files requires `gcc`, `g++`, `libstdc++-13-dev`, and `libstdc++.so` symlinked into the linker path. The default `cc` must be set to `gcc` (not `clang`), since tree-sitter compiles mixed `.c`/`.cc` sources. If `cc` points to `clang`, C++ standard library headers won't be found; if `cc` points to `g++`, C99 designated initializers in `.c` files fail. Use `gcc` as `cc` and the linker will invoke `c++` (clang++) for linking, which needs `-lstdc++` to be findable.

### Tree-sitter grammars
- Grammar repos are expected at `/data/code/tree-sitter/tree-sitter-{ruby,javascript,go,python,java,php,c-sharp}`
- Use `v0.20.x` tags (e.g. `v0.20.0`, `v0.20.4`) to match `tree-sitter==0.20.4` language version 14 requirement. The latest `main` branches generate language version 15 which is incompatible.
- Main library: `build/my-language.so` (built in workspace root)
- Evaluator library: `evaluator/CodeBLEU/parser/my-languages.so`

### Pre-trained models
The code in `models.py` loads from `/data/huggingface_models/{model-name}`. Download models there using `transformers` `save_pretrained()`. Available model names: `roberta-base`, `codebert-base`, `graphcodebert-base`, `t5-base`, `codet5-base`, `bart-base`, `plbart-base`.

### Running
- Training: `PYTHONPATH=/workspace TOKENIZERS_PARALLELISM=false python3 main.py --task summarize --sub_task ruby --model_name codebert --do_train --do_eval --no_cuda --data_num 5 --data_dir data --output_dir outputs/summarize/ruby/codebert --cache_path outputs/summarize/ruby/codebert/cache_data --res_dir results/summarize/ruby/codebert`
- Attention analysis: see `run_att.sh`
- Use `--no_cuda` for CPU-only environments
- Use `--data_num N` to limit to N samples for quick tests
- Cached data files in `cache_data/` may become incompatible if PyTorch version changes; delete them if you get `_pickle.UnpicklingError`

### Linting
No linting tool is configured in the repo. Use `python3 -m pyflakes` for basic checks; existing files have some unused-import warnings which are expected.

### Testing
No automated test suite exists. Validate the environment by running `main.py` with `--data_num 5` on the summarize/ruby task with CodeBERT.
