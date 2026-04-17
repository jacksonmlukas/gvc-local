# GVC-Local: Open-Model Multi-Agent Reasoning + Serving Pipeline

Extending the [ACL 2025 REALM paper](https://aclanthology.org/2025.realm-1.16/) *"Snap Out of It: A Dual-Process Approach to Mitigating Overthinking in Language Model Reasoning"* to open-weight models served locally via vLLM, with LoRA fine-tuning, RAG-augmented context, and a production serving layer.

The original paper reports headline numbers on GPT-4o and GPT-4o-mini but leaves open-model rows in Table 1 empty. This project fills them in and asks: does the dual-process escape hatch generalize to 8B open models, and can fine-tuning + retrieval close the gap?

## Architecture

```
                       +------------------+
                       |   FastAPI (8080)  |
                       |   /solve /health  |
                       |   /metrics        |
                       +--------+---------+
                                |
                   +------------+------------+
                   |                         |
          +--------v--------+       +--------v--------+
          |  Snap-GVC Solver |       |   GVC Solver    |
          |  (dual-process)  |       |  (System-2 only)|
          +--------+---------+       +--------+--------+
                   |                          |
       +-----------+-----------+    +---------+---------+
       |           |           |    |         |         |
  +----v---+ +----v----+ +----v-+ +---v---+ +-v------+ |
  |Guesser | |Validator| | Snap | |Guesser| |Validat.| |
  | Agent  | | Agent   | |Guessr| | Agent | | Agent  | |
  +----+---+ +----+----+ +--+---+ +---+---+ +---+----+ |
       |          |          |         |         |      |
       +----------+----------+---------+---------+      |
                             |                          |
                    +--------v--------+                 |
                    |  vLLM Server    |     +-----------+
                    |  (Llama / Qwen) |     |
                    |  + LoRA adapter |     |  +----------------+
                    +-----------------+     +->| RAG Retriever  |
                                               | (FAISS + MiniLM)|
                                               +----------------+
```

## Key Features

**Open-model inference** — Drop-in vLLM-backed endpoint compatible with the upstream solver interface. Supports Llama 3.1 8B, Llama 3.3 70B, and Qwen 2.5 7B.

**LoRA/QLoRA fine-tuning** — Full pipeline to fine-tune open models on successful puzzle-solving traces, using 4-bit quantization (QLoRA) with PEFT + TRL's SFTTrainer. Includes data preparation, training, and adapter merging scripts.

**RAG retrieval layer** — FAISS-indexed historical puzzles and solve traces, retrieved at inference time to provide contextual grounding to agents. Replaces brute-force prompt context stuffing.

**Production serving** — FastAPI service with per-request latency/token monitoring, Dockerized with docker-compose (API + vLLM as separate services), and CI via GitHub Actions.

**Evaluation harness** — Stratified-sample eval with bootstrapped 95% CIs on solve rate, semantic grounding, and guesses-per-puzzle. Extends beyond Connections to GAIA Level-1. Experiment tracking via Weights & Biases.

## Quickstart

```bash
# Clone and install
git clone https://github.com/jacksonmlukas/gvc-local.git
cd gvc-local
pip install -e "."

# Start vLLM (requires GPU)
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000

# Solve puzzles 0-10 with Snap-GVC
gvc-local snap_gvc llama-3.1-8b --start 0 --end 10

# Or run the full eval harness with W&B tracking
python scripts/run_eval.py --solver snap_gvc --model llama-3.1-8b \
    --start 0 --end 50 --wandb-project gvc-local-eval
```

### Docker (recommended)

```bash
docker compose up        # starts vLLM + FastAPI
curl localhost:8080/health
curl -X POST localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d '{"words": ["CRICKET","FROG","HARE","KANGAROO","CRUSH","MASH","PRESS","SQUASH","BREAKING","HOCKEY","SKELETON","TRAMPOLINE","MOOD","RECORD","TABLE","VOLLEYBALL"]}'
```

### Fine-tuning

```bash
# 1. Collect traces by running the solver with --trace-path
gvc-local snap_gvc llama-3.1-8b --start 0 --end 100 --trace-path data/traces/

# 2. Prepare training data
python -m gvc_local.finetune.data_prep --trace-path data/traces/ --output-dir data/sft/

# 3. Fine-tune with QLoRA
python -m gvc_local.finetune.train --config configs/finetune.yaml

# 4. Merge adapter for serving
python -m gvc_local.finetune.merge \
    --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --adapter-path outputs/checkpoints/final \
    --output-dir outputs/merged/
```

### Building the RAG index

```bash
python scripts/build_rag_index.py \
    --puzzles data/puzzles/connections.json \
    --traces data/traces/ \
    --output data/rag_index/
```

## Repository structure

```
src/gvc_local/
  agents/             # Guesser, Validator, Snap agents backed by vLLM
  solvers/            # GVC and Snap-GVC solver implementations
  rag/                # FAISS indexer + retriever for puzzle context
  finetune/           # LoRA/QLoRA training pipeline (data prep, train, merge)
  serving/            # FastAPI app with monitoring
  eval/               # Eval harness, GAIA adapter, W&B tracking
  game.py             # Connections game logic (standalone or upstream import)
  endpoint.py         # vLLM-backed OpenAI-compatible client
  cli.py              # CLI entrypoint
configs/              # YAML configs for fine-tuning and evaluation
scripts/              # CLI scripts for building indexes, running evals
tests/                # Unit tests
Dockerfile            # API container
docker-compose.yml    # API + vLLM orchestration
```

## Relationship to the upstream paper

This work builds on the codebase at [`Chrislai502/the_amazing_connections`](https://github.com/Chrislai502/the_amazing_connections). The upstream repository contains the original GVC / Snap-GVC implementation and the prompts in Appendix A.4 of the paper. This repo reimplements the core solver logic to work with open models via vLLM, replacing the AutoGen + OpenAI dependency with direct API calls.

## Citation

If you use this work, please cite the original paper:

```bibtex
@inproceedings{pandian-etal-2025-snap,
    title = "Snap Out of It: A Dual-Process Approach to Mitigating Overthinking in Language Model Reasoning",
    author = "Pandian, Ashish  and
      Lojo, Nelson  and
      Lai, Wei Xun  and
      Lukas, Jackson",
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

## License

MIT — see [LICENSE](LICENSE).

## Author

[Jackson Lukas](https://github.com/jacksonmlukas) — co-author on the original paper, maintaining this extension independently.
