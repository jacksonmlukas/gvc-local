# Roadmap

Milestones are sized to land the whole extension in 6–8 weeks at ~8 hrs/week. Ordering prioritizes getting a runnable end-to-end pipeline in front of everything else, so later milestones become increments rather than integrations.

**Infrastructure:** all inference runs through cloud providers (Groq free tier, Together AI) — no local GPU required. Fine-tuning uses Together AI's LoRA API (~$1–5 total).

## M1 — Cloud endpoint + smoke run on Llama 3.1 8B · ✅ complete

- [x] Implement `EndpointConfig` wrapper conforming to the upstream `rsallms` contract.
- [x] Add cloud provider factories (Groq, Together AI) reading API keys from env vars.
- [x] Wire solvers (basic, cot, gvc, snap_gvc) into CLI, eval harness, and serving API.
- [x] Run `basic` solver end-to-end on 10 puzzles via Groq; pipeline confirmed working.
- [x] CI green (ruff check, ruff format, pytest, mypy continue-on-error).

## M2 — GVC and Snap-GVC on Llama 3.1 8B · ✅ complete

- [x] Run `basic` solver on 10 puzzles via Groq: **2/10 (20%)**.
- [x] Run `cot` solver on 10 puzzles via Groq: **3/10 (30%)**.
- [x] Run `gvc` solver on 10 puzzles via Groq: **6/10 (60%)**.
- [x] Run `snap_gvc` solver on 10 puzzles via Groq: **6/10 (60%)** *(v3 deterministic-swap iteration; see README footnote on Groq rate-limit noise)*.
- [x] Collect solver interaction traces (JSONL) — 10 each for basic + snap_gvc + finetuned.
- [ ] Scale all 4 solvers to 100 puzzles with bootstrapped CIs (rolled into M5).

**Headline result (Llama 3.1 8B via Groq, 10 puzzles):**

| Solver | Solved | Rate |
|--------|--------|------|
| basic | 2/10 | 20% |
| cot | 3/10 | 30% |
| gvc | 6/10 | **60%** |
| snap_gvc | 6/10 | **60%** |

Multi-agent prompting structure triples open-model accuracy with no parameter updates.

**Exit criterion met:** four-strategy table published for Llama 3.1 8B in README.

## M3 — Qwen 2.5 7B replication + fine-tuning data · 3–5 days

- [ ] Swap in Qwen 2.5 7B Instruct via Groq / Together AI.
- [ ] Full Table 1 row. Compare against Llama 3.1 8B — which model benefits more from Snap-GVC?
- [ ] Build fine-tuning dataset: convert solver traces to chat-format JSONL (system + user + assistant turns distilled from multi-agent interactions).

**Exit criterion:** two open-model columns complete. Fine-tuning dataset ready (≥200 examples).

## M4 — LoRA fine-tuning via Together AI · ✅ complete (negative result)

- [x] `scripts/data_prep.py`: converts 1,037 Connections puzzles into 3,320 train / 828 test chat-format examples (~786K train tokens).
- [x] `scripts/finetune.py`: uploads data and launches LoRA fine-tuning on Together AI with job management CLI.
- [x] LoRA fine-tuning job completed on `meta-llama/Meta-Llama-3.1-8B-Instruct-Reference` (3 epochs, batch 8, lr 1e-5, rank 16). Actual cost: **$4.00**, runtime ~8 minutes.
- [x] Adapter uploaded to HuggingFace: [`jacksonlukas/gvc-connections-lora`](https://huggingface.co/jacksonlukas/gvc-connections-lora).
- [x] Held-out eval on 50 puzzles (830–879) via Google Colab free T4 GPU + 4-bit quantized inference: **6/50 = 12.0%**.
- [x] Apples-to-apples eval on puzzles 0–9: **2/10 = 20%** (matches basic single-pass baseline; LoRA did not help).

**Finding:** distilling single-pass behavior into 8B weights regresses against multi-agent zero-shot prompting on the same base model. Working hypothesis: at this scale, inference-time control structure does more work than weight-level distillation of single-pass answers. Follow-up question (queued for a future milestone): would distilling multi-agent *traces* close the gap?

**Exit criterion met:** clear comparison between single-pass LoRA and multi-agent zero-shot, both reported in README.

## M5 — Proper eval harness · ✅ complete (v0.2.0)

- [x] Replace ad-hoc puzzle-index slicing with stratified sampling: implemented `gvc_local.eval.tagger` (heuristic, 5 strata) and tagged the full 1,078-puzzle corpus.
- [x] Bootstrapped 95% confidence intervals on solve rate and strikes-per-puzzle. 5,000 bootstrap resamples in headline results.
- [x] Write up variance observations — README per-stratum breakdown shows the LoRA-finetuned solver is at 0% on wordplay and cultural strata while overall is 12%. Single-overall-metric would have hidden this.

**Exit criterion met:** v0.2.0 tagged. README results table regenerated with CIs.

## M6 — Second benchmark: GAIA (level 1) · 1–2 weeks

- [ ] Integrate the GAIA benchmark harness. Stick to level-1 questions — multi-step web search with a known answer, similar in spirit to Connections but genuinely agentic.
- [ ] Wrap Snap-GVC around a tool-using base agent (web search + retrieval). This is the stretch goal — the orchestration changes and the "Snap" mechanism has to be re-derived for a non-Connections task. If the transfer doesn't work, write that up as a negative result.
- [ ] Report solve rate with vs. without the Snap escape hatch.

**Exit criterion:** a defensible answer to "does the Snap-GVC pattern transfer outside Connections?"

## M7 — Blog post + demo · 3–5 days

- [ ] Blog post: *"Does Snap-GVC transfer to open models? Reproducing and extending our ACL 2025 paper."* Target length 1500–2000 words. Include results tables, a failure-mode gallery, and honest notes on what didn't work.
- [ ] Hugging Face Space: user pastes a 16-word Connections board, picks a solver, watches Snap-GVC run. Minimal UI, strong demo value.
- [ ] Post to X (tag co-authors, plus AK, Philipp Schmid, relevant HF folks), LinkedIn, r/MachineLearning.

**Exit criterion:** blog post published, Space live, post has at least 1 retweet from someone relevant. Repo README updated to link all of the above.

---

## Cadence

- Push something every weekend. Even a stub. The goal is to never have a "last commit was 3 weeks ago" moment on the GitHub profile between now and M7.
- Tag releases at the end of M2, M3, M4, M5, M6 (`v0.2` through `v0.6`) so the repo has clean checkpoint points a hiring manager can skim.

## What I will deliberately NOT do

- No local GPU. All inference via Groq (free) and Together AI ($5 credits). Fine-tuning via Together AI's LoRA API.
- No leaderboard-chasing. Goal is to report honest numbers on two open models across all strategies, not to crown a winner.
- No web UI beyond the final HF Space. Time on polish is time not spent on results.
