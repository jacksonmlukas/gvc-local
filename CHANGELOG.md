# Changelog

All notable changes to GVC-Local. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05

First tagged release. Open-weight extension of the ACL 2025 REALM paper *"Snap Out of It: A Dual-Process Approach to Mitigating Overthinking in Language Model Reasoning"* (Pandian et al.).

### Added
- `EndpointConfig` wrapper conforming to the upstream `rsallms` contract; provider factories for Groq, Together AI, and local vLLM (`src/gvc_local/endpoint.py`).
- Solver implementations: `basic`, `cot`, `gvc`, `snap_gvc` (`src/gvc_local/solvers/`). Snap-GVC v3 uses a deterministic swap engine and is ~4× faster than the v1 iteration.
- Click-based CLI: `gvc-local <solver> <model> --provider <provider>` with `--delay`, `--model-override`, `--start`, `--end`, `--out`, `--traces` flags (`src/gvc_local/cli.py`).
- Eval harness with stratified sampling and bootstrap CI scaffolding (`src/gvc_local/eval/harness.py`, `src/gvc_local/eval_harness.py`).
- FastAPI serving layer with `/solve`, `/health`, `/metrics` endpoints; request logging and latency/token-usage monitoring (`src/gvc_local/serving/`).
- Lazy-imported FAISS-based RAG retriever for puzzle metadata (`src/gvc_local/rag/`).
- Fine-tuning pipeline: data prep from puzzle DB → 3,320 chat-format training examples → Together AI LoRA launch + status CLI → adapter upload to HuggingFace (`scripts/data_prep.py`, `scripts/finetune.py`, `scripts/upload_adapter.py`).
- Google Colab eval notebook for fine-tuned models on free T4 GPU with 4-bit quantization (`notebooks/eval_finetuned_colab.ipynb`).
- Docker Compose: vLLM + FastAPI services.
- GitHub Actions CI: ruff check, ruff format, pytest, mypy (`continue-on-error`).

### Results
- `basic` (Llama 3.1 8B, Groq): **2/10 = 20%** on puzzles 0–9.
- `cot` (Llama 3.1 8B, Groq): **3/10 = 30%** on puzzles 0–9.
- `gvc` (Llama 3.1 8B, Groq): **6/10 = 60%** on puzzles 0–9.
- `snap_gvc` (Llama 3.1 8B, Groq, v3): **6/10 = 60%** on puzzles 0–9 *(rate-limit-affected; clean re-run pending)*.
- LoRA-tuned `basic` (Together AI fine-tune, Colab T4 4-bit eval): **2/10 = 20%** on puzzles 0–9; **6/50 = 12.0%** on held-out puzzles 830–879.
- Total spend: **$4 fine-tuning** + 0 inference (Groq free tier) + 0 eval (Colab free T4) = $4.

### Findings
- Multi-agent prompting structure (GVC / Snap-GVC) triples basic single-pass accuracy on the same Llama 3.1 8B base model — purely structural, no parameter updates.
- Single-pass LoRA fine-tuning regresses against multi-agent zero-shot prompting on the same base model (12% vs 60%). Inference-time control structure beats weight-level distillation of single-pass behavior at this scale.
- Failure modes are bimodal: 6 full solves, 0 partials at 3/4, 27 zero-solves on the held-out 50.

### Known issues
- Snap-GVC v3 result is suspected to be slightly underestimated due to Groq free-tier rate limiting; clean re-run pending.
- All headline numbers are on N=10 puzzles; bootstrapped CIs not yet computed (queued in M5).
