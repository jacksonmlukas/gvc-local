# GVC-Local: Open-Model Multi-Agent Reasoning for NYT Connections

Extending the [ACL 2025 REALM paper](https://aclanthology.org/2025.realm-1.16/) *"Snap Out of It: A Dual-Process Approach to Mitigating Overthinking in Language Model Reasoning"* to open-weight models via cloud inference, with LoRA fine-tuning and a production serving layer.

The original paper reports headline numbers on GPT-4o and GPT-4o-mini but leaves open-model rows in Table 1 empty. This project fills them in and asks: does the dual-process escape hatch generalize to 8B open models, and can LoRA fine-tuning on puzzle data close the gap?

## Results

Evaluated on NYT Connections puzzles using Llama 3.1 8B Instruct.

| Solver | Method | Puzzles | Solved | Rate |
|--------|--------|---------|--------|------|
| `basic` | Single-pass | 0–9 (10) | 2 | 20% |
| `cot` | Chain-of-thought | 0–9 (10) | 3 | 30% |
| `gvc` | Multi-agent consensus | 0–9 (10) | 6 | **60%** |
| `snap_gvc` | Multi-agent dual-process | 0–9 (10) | 6 | **60%**\* |
| `basic` (LoRA ft) | Single-pass, fine-tuned | 0–9 (10) | 2 | 20% |
| `basic` (LoRA ft) | Single-pass, fine-tuned | 830–879 (50, held-out) | 6 | 12% |

\*Snap-GVC result may be affected by Groq free-tier rate limiting; a clean re-run is pending.

The multi-agent GVC and Snap-GVC pipelines triple the basic single-pass accuracy on the same model without any additional training. The fine-tuned model was trained on puzzle data via Together AI LoRA ($4, 8 minutes) and evaluated on a fully held-out set using Google Colab's free T4 GPU.

## Key Findings

**1. Multi-agent prompting structure triples open-model accuracy without any training.** GVC and Snap-GVC reach 60% on the same Llama 3.1 8B that gets 20% with basic single-pass prompting. The lift is purely structural — Guesser/Validator consensus loops + a deterministic grounding check, plus the dual-process escape hatch in Snap-GVC. No parameter updates required.

**2. At 8B scale, single-pass LoRA fine-tuning regresses against multi-agent zero-shot prompting.** The LoRA-tuned model reaches 12% on a held-out 50-puzzle eval and matches the basic single-pass baseline (2/10) on puzzles 0–9 — well below the 60% of multi-agent prompting on the same base model. This is a clean comparison: same base model, same puzzles. Distilling single-pass behavior into weights did not capture what the multi-agent loop is doing at inference time.

This is consistent with a working hypothesis that for hard reasoning tasks at smaller open-model scale, *inference-time control structure* (multi-agent loops, validation, the dual-process escape hatch) does more work than weight-level distillation of single-pass behavior. A natural follow-up: distill the multi-agent *traces* (which encode the reasoning loop) rather than the single-pass answer key, and see whether weight-level methods can close the gap that way.

**3. Failure modes are bimodal.** On the 50-puzzle held-out set the fine-tuned model produced 6 full solves, 0 partials at 3/4, 3 at 2/4, 14 at 1/4, and 27 at 0/4. When it fails it tends to repeat similar wrong guesses until exhausting strikes — suggesting LoRA sharpened a confidence prior without expanding the model's actual category-recognition coverage.

**4. Total budget: $10. No local GPU at any stage.** All inference via Groq free tier; LoRA fine-tuning $4 / 8 minutes on Together AI; held-out eval on Google Colab's free T4 GPU.

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
                    | Cloud Inference |     +-----------+
                    | (Groq/Together) |     |
                    | or local vLLM   |     |  +----------------+
                    +-----------------+     +->| RAG Retriever  |
                                               | (FAISS + MiniLM)|
                                               +----------------+
```

### Solver Progression

The project implements four solver strategies of increasing sophistication:

**Basic** makes a single-pass guess per round. **CoT** adds chain-of-thought prompting to encourage reasoning before guessing. **GVC** introduces a multi-agent loop: a Guesser proposes a group, a deterministic grounding check validates it against the board, and a Validator LLM reviews the proposal — the guess is only submitted when consensus is reached. **Snap-GVC** wraps this in a dual-process architecture inspired by cognitive science: when the deliberative GVC loop stalls (too many wrong guesses or failures to reach consensus), it switches to a high-temperature "snap" phase for fast intuitive guessing. A correct snap guess restores confidence and switches back to deliberation.

### Fine-Tuning Pipeline

Rather than distilling multi-agent solver traces, the fine-tuning data is constructed directly from the Connections puzzle answer database. Each of the 1,037 puzzles generates 4 training examples (one per round, easiest to hardest group), simulating multi-round play with the remaining words as context. This produced 3,320 training examples (~786K tokens). LoRA fine-tuning ran on Together AI (rank 16, 3 epochs, lr 1e-5) in under 8 minutes for $4.

The LoRA adapter is available on HuggingFace: [jacksonlukas/gvc-connections-lora](https://huggingface.co/jacksonlukas/gvc-connections-lora)

## Quickstart

```bash
# Clone and install
git clone https://github.com/jacksonmlukas/gvc-local.git
cd gvc-local
pip install -e "."
```

### Cloud inference (no GPU required)

```bash
# Set your API key (Groq free tier)
export GROQ_API_KEY=your-key-here

# Run basic solver on puzzles 0-9
gvc-local basic llama-3.1-8b --provider groq --start 0 --end 10

# Run Snap-GVC with rate-limit delay
gvc-local snap_gvc llama-3.1-8b --provider groq --start 0 --end 10 --delay 2

# Save results to a file
gvc-local snap_gvc llama-3.1-8b --provider groq --start 0 --end 10 \
    --out results/snap_gvc_groq.jsonl

# Save solver interaction traces (for analysis)
gvc-local snap_gvc llama-3.1-8b --provider groq --start 0 --end 10 \
    --traces data/traces/

# Use Together AI
export TOGETHER_API_KEY=your-key-here
gvc-local basic llama-3.1-8b --provider together --start 0 --end 10

# Use a fine-tuned model via Together AI
gvc-local basic llama-3.1-8b --provider together \
    --model-override "your-finetuned-model-id"
```

### Local vLLM (requires GPU)

```bash
# Start vLLM server
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000

# Run solver against local endpoint
gvc-local snap_gvc llama-3.1-8b --start 0 --end 10
```

### Fine-tuning

```bash
# 1. Prepare training data from puzzle database
python scripts/data_prep.py --output-dir data/finetune/

# 2. Launch LoRA fine-tuning on Together AI
export TOGETHER_API_KEY=your-key-here
python scripts/finetune.py

# 3. Check job status
python scripts/finetune.py --status <job-id>

# 4. Upload adapter to HuggingFace
export HF_TOKEN=your-token
python scripts/upload_adapter.py --adapter-dir /tmp/adapter \
    --hf-repo your-username/your-adapter-name

# 5. Evaluate on Google Colab (free T4 GPU)
#    Upload notebooks/eval_finetuned_colab.ipynb to Colab
```

### Docker

```bash
docker compose up        # starts vLLM + FastAPI
curl localhost:8080/health
curl -X POST localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d '{"words": ["CRICKET","FROG","HARE","KANGAROO","CRUSH","MASH","PRESS","SQUASH","BREAKING","HOCKEY","SKELETON","TRAMPOLINE","MOOD","RECORD","TABLE","VOLLEYBALL"]}'
```

## Inference Providers

| Provider | Cost | Notes |
|----------|------|-------|
| **Groq** | Free tier | Primary provider for solver evals |
| **Together AI** | ~$5 credits | Fine-tuning API + inference |
| **Local vLLM** | Free (requires GPU) | OpenAI-compatible, supports LoRA adapters |

All providers use the same OpenAI-compatible API via the `EndpointConfig` abstraction in `endpoint.py`.

## Repository Structure

```
src/gvc_local/
    cli.py              # Click CLI: gvc-local <solver> <model> --provider <provider>
    endpoint.py         # EndpointConfig + Client, provider factory methods
    game.py             # Connections game logic, puzzle loading from GitHub
    agents/             # Guesser, Validator, Snap agents
    solvers/            # GVC and Snap-GVC solver implementations
    serving/            # FastAPI app with monitoring
    rag/                # FAISS indexer + retriever (optional, lazy imports)
    eval/               # Eval harness, GAIA adapter, W&B tracking
scripts/
    data_prep.py        # Puzzle DB → chat-format JSONL for fine-tuning
    finetune.py         # Together AI LoRA job launch + management
    upload_adapter.py   # Upload LoRA adapter to HuggingFace
    eval_finetuned.py   # Local GPU eval script (alternative to Colab)
notebooks/
    eval_finetuned_colab.ipynb  # Google Colab eval (T4 GPU, 4-bit quantized)
results/                # JSONL result files per solver/model combo
data/finetune/          # Train/test JSONL splits
configs/                # YAML configs for fine-tuning and eval
```

## Relationship to the Upstream Paper

This work builds on the codebase at [`Chrislai502/the_amazing_connections`](https://github.com/Chrislai502/the_amazing_connections). The upstream repository contains the original GVC / Snap-GVC implementation and the prompts in Appendix A.4 of the paper. This repo reimplements the core solver logic to work with open models via cloud inference providers, replacing the AutoGen + OpenAI dependency with direct API calls.

## Limitations & What's Next

Honest disclosure of what hasn't been done yet:

- **Sample size.** Headline solver runs are on 10 puzzles; held-out fine-tuned eval is 50. No bootstrapped confidence intervals yet — that's the next milestone.
- **Single base model.** Llama 3.1 8B Instruct only. A Qwen 2.5 7B replication is queued and will give a proper second column.
- **Single fine-tuning recipe.** Rank-16 LoRA, 3 epochs, puzzle-DB as training data. Higher-rank fine-tunes or a recipe that distills multi-agent *traces* (rather than puzzle answers) might close the gap. Untested.
- **Snap-GVC v3 result has known noise.** Groq free-tier rate limiting affected the run; a clean re-run is pending. The headline 60% is conservative — possibly understated.
- **No out-of-domain transfer yet.** Whether the dual-process escape hatch generalizes beyond NYT Connections (e.g. to GAIA level-1 questions) is the open question — that's a planned milestone.

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
