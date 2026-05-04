# GVC-Local v0.1.0 — Open-Model Multi-Agent Reasoning for NYT Connections

First tagged release. Extends the [ACL 2025 REALM paper](https://aclanthology.org/2025.realm-1.16/) *"Snap Out of It: A Dual-Process Approach to Mitigating Overthinking in Language Model Reasoning"* to open-weight models, with LoRA fine-tuning and a production serving layer — all built on a $10 budget with no local GPU.

## Headline result

**Multi-agent prompting structure triples open-model accuracy without any training.** GVC and Snap-GVC reach **60%** on Llama 3.1 8B; the same model gets **20%** with basic single-pass prompting.

| Solver | Puzzles | Solved | Rate |
|--------|---------|--------|------|
| `basic` | 0–9 (10) | 2 | 20% |
| `cot` | 0–9 (10) | 3 | 30% |
| `gvc` | 0–9 (10) | 6 | **60%** |
| `snap_gvc` | 0–9 (10) | 6 | **60%** |
| `basic` (LoRA ft) | 830–879 (50, held-out) | 6 | 12% |

**Key finding from the LoRA experiment:** at 8B scale, single-pass fine-tuning regresses against multi-agent zero-shot prompting on the same base model. Distilling single-pass behavior into weights did not capture what the multi-agent loop is doing at inference time. See README "Key Findings" for the full discussion.

## What's in this release

- Four solvers: `basic`, `cot`, `gvc`, `snap_gvc` (Snap-GVC v3 uses a deterministic swap engine, ~4× faster than v1)
- Provider-agnostic inference: Groq, Together AI, or local vLLM via the same `EndpointConfig` abstraction
- Full fine-tuning pipeline: puzzle DB → 3,320 chat-format examples → Together AI LoRA → HuggingFace adapter
- FastAPI serving layer in Docker (`/solve`, `/health`, `/metrics`)
- FAISS-based RAG retriever (lazy-imported, optional)
- CI green on GitHub Actions (ruff, pytest, mypy)
- LoRA adapter on HF: [`jacksonlukas/gvc-connections-lora`](https://huggingface.co/jacksonlukas/gvc-connections-lora)

## Limitations

- Headline runs on N=10 puzzles; held-out fine-tune eval N=50. No bootstrapped CIs yet — that's the next milestone.
- Single base model (Llama 3.1 8B). Qwen 2.5 7B replication is queued.
- Snap-GVC v3 was rate-limit-affected on Groq free tier; a clean re-run is pending.

## What's next

- M5 — proper eval harness: stratified sampling, bootstrapped 95% CIs on every metric.
- Trace-distilled fine-tuning experiment: distill multi-agent *reasoning loops* rather than puzzle answer keys, and see whether weight-level methods can close the gap that way.
- M6 — out-of-domain transfer to GAIA level-1 questions.
- M7 — blog post + HuggingFace Space demo.

## Citation

If you use this work, please cite the original paper:

```bibtex
@inproceedings{pandian-etal-2025-snap,
    title = "Snap Out of It: A Dual-Process Approach to Mitigating Overthinking in Language Model Reasoning",
    author = "Pandian, Ashish and Lojo, Nelson and Lai, Wei Xun and Lukas, Jackson",
    booktitle = "Proceedings of the 1st Workshop for Research on Agent Language Models (REALM 2025)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.realm-1.16/",
    doi = "10.18653/v1/2025.realm-1.16",
    pages = "228--249"
}
```

---

*Maintained independently by [Jackson Lukas](https://github.com/jacksonmlukas), co-author on the original paper.*
