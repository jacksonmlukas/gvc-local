# Roadmap

Milestones are sized to land the whole extension in 6–8 weeks at ~8 hrs/week. Ordering prioritizes getting a runnable end-to-end pipeline in front of everything else, so later milestones become increments rather than integrations.

**Infrastructure:** all inference runs through cloud providers (Groq free tier, Together AI) — no local GPU required. Fine-tuning uses Together AI's LoRA API (~$1–5 total).

## M1 — Cloud endpoint + smoke run on Llama 3.1 8B · ✅ complete

- [x] Implement `EndpointConfig` wrapper conforming to the upstream `rsallms` contract.
- [x] Add cloud provider factories (Groq, Together AI) reading API keys from env vars.
- [x] Wire solvers (basic, cot, gvc, snap_gvc) into CLI, eval harness, and serving API.
- [x] Run `basic` solver end-to-end on 10 puzzles via Groq; pipeline confirmed working.
- [x] CI green (ruff check, ruff format, pytest, mypy continue-on-error).

## M2 — GVC and Snap-GVC on Llama 3.1 8B · in progress

- [x] Run `basic` solver on 10 puzzles via Groq: **2/10 (20%)**.
- [x] Run `snap_gvc` solver on 9 puzzles via Groq: **6/9 (67%)**. Snap-GVC triples basic accuracy.
- [x] Collect solver interaction traces (JSONL) — 10 basic + 9 snap_gvc traces saved.
- [ ] Run `cot` and `gvc` solvers on 10 puzzles via Groq.
- [ ] Scale all 4 solvers to 100 puzzles for full Table 1.

**Preliminary results (Llama 3.1 8B via Groq):**

| Solver | Puzzles | Solved | Rate |
|--------|---------|--------|------|
| basic | 10 | 2 | 20% |
| snap_gvc | 9 | 6 | 67% |

**Exit criterion:** published results table for Llama 3.1 8B across all four strategies.

## M3 — Qwen 2.5 7B replication + fine-tuning data · 3–5 days

- [ ] Swap in Qwen 2.5 7B Instruct via Groq / Together AI.
- [ ] Full Table 1 row. Compare against Llama 3.1 8B — which model benefits more from Snap-GVC?
- [ ] Build fine-tuning dataset: convert solver traces to chat-format JSONL (system + user + assistant turns distilled from multi-agent interactions).

**Exit criterion:** two open-model columns complete. Fine-tuning dataset ready (≥200 examples).

## M4 — LoRA fine-tuning via Together AI · in progress

- [x] `scripts/data_prep.py`: converts 1,037 Connections puzzles into 3,320 train / 828 test chat-format examples (~786K train tokens).
- [x] `scripts/finetune.py`: uploads data and launches LoRA fine-tuning on Together AI with job management CLI.
- [x] LoRA fine-tuning job launched on `meta-llama/Meta-Llama-3.1-8B-Instruct-Reference` (3 epochs, batch 8, lr 1e-5, rank 16). Estimated cost ~$6.29.
- [ ] Evaluate fine-tuned model on held-out 207 puzzles using `basic` solver (single pass, no multi-agent).
- [ ] Add Table 1 row for fine-tuned model. Compare single-pass fine-tuned vs. multi-agent Snap-GVC.

**Exit criterion:** fine-tuned Llama 3.1 8B evaluated on 50+ puzzles. Clear comparison: does distilling multi-agent reasoning into weights work?

## M5 — Proper eval harness · 1 week

- [ ] Replace the current ad-hoc puzzle-index slicing with stratified sampling: sample puzzles such that the eval set covers known category types (wordplay, cultural, category/tag, silent-letter, etc.) proportional to their frequency. Will need a small labelling pass — ~10 minutes of manual tagging.
- [ ] Bootstrapped 95% confidence intervals on all three metrics (solve rate, grounding, guesses-per-puzzle). 1000 bootstrap resamples.
- [ ] Write up variance observations — the paper's narrative improves if we can show CoT has high variance and Snap-GVC has tight CIs.

**Exit criterion:** every number in the Table 1 row now has a CI. Results table regenerated for M2–M4.

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
