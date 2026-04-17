# Roadmap

Milestones are sized to land the whole extension in 6–8 weeks at ~8 hrs/week. Ordering prioritizes getting a runnable end-to-end pipeline in front of everything else, so later milestones become increments rather than integrations.

## M1 — vLLM endpoint + smoke run on Llama 3.1 8B · 1 week

- [ ] Stand up vLLM serving Llama 3.1 8B Instruct, OpenAI-compatible API.
- [ ] Implement `endpoint/llama.py` wrapper that conforms to the upstream `EndpointConfig` contract used by `rsallms`.
- [ ] Run `basic` and `cot` solvers end-to-end on 10 puzzles; confirm solve rate matches the expected LLaMa 3.3 CoT number (~76%) roughly — 8B will be lower but the pipeline should be clearly working.
- [ ] Pin a Dockerfile that reproduces this end-to-end on a single A10/A100.

**Exit criterion:** one command brings up vLLM, runs 10 Connections puzzles with CoT on Llama 3.1 8B, writes results to a local SQLite file. Published as `v0.1`.

## M2 — GVC and Snap-GVC on Llama 3.1 8B · 1–2 weeks

- [ ] Port GVC agents to the local endpoint. Expect prompt-formatting friction — Llama chat templates differ from OpenAI's.
- [ ] Implement Snap-GVC with the same failure-threshold and temperature-switch logic as upstream.
- [ ] Run full 100-puzzle eval for `basic`, `cot`, `gvc`, `snap_gvc`. Produce Table 1 row for Llama 3.1 8B.
- [ ] Sanity-check results against upstream's LLaMa 3.1 8B rows for the baselines — if they don't line up, figure out why before going further.

**Exit criterion:** published results table for Llama 3.1 8B across all four strategies. Snap-GVC should measurably beat CoT, and the gap should be at least as large as the upstream GPT-4o-mini gap (42 percentage points). If not, write up why.

## M3 — Qwen 2.5 7B replication · 3–5 days

- [ ] Swap in Qwen 2.5 7B Instruct. Only the endpoint wrapper and chat template should need to change.
- [ ] Full Table 1 row. Compare against Llama 3.1 8B — which model benefits more from Snap-GVC?

**Exit criterion:** two open-model columns complete. Preliminary hypothesis documented on why one benefits more.

## M4 — Proper eval harness · 1 week

- [ ] Replace the current ad-hoc puzzle-index slicing with stratified sampling: sample puzzles such that the eval set covers known category types (wordplay, cultural, category/tag, silent-letter, etc.) proportional to their frequency. Will need a small labelling pass — ~10 minutes of manual tagging.
- [ ] Bootstrapped 95% confidence intervals on all three metrics (solve rate, grounding, guesses-per-puzzle). 1000 bootstrap resamples.
- [ ] Write up variance observations — the paper's narrative improves if we can show CoT has high variance and Snap-GVC has tight CIs.

**Exit criterion:** every number in the Table 1 row now has a CI. Results table regenerated for M2–M3.

## M5 — Second benchmark: GAIA (level 1) · 1–2 weeks

- [ ] Integrate the GAIA benchmark harness. Stick to level-1 questions — multi-step web search with a known answer, similar in spirit to Connections but genuinely agentic.
- [ ] Wrap Snap-GVC around a tool-using base agent (web search + retrieval). This is the stretch goal — the orchestration changes and the "Snap" mechanism has to be re-derived for a non-Connections task. If the transfer doesn't work, write that up as a negative result.
- [ ] Report solve rate with vs. without the Snap escape hatch.

**Exit criterion:** a defensible answer to "does the Snap-GVC pattern transfer outside Connections?"

## M6 — Blog post + demo · 3–5 days

- [ ] Blog post: *"Does Snap-GVC transfer to open models? Reproducing and extending our ACL 2025 paper."* Target length 1500–2000 words. Include results tables, a failure-mode gallery, and honest notes on what didn't work.
- [ ] Hugging Face Space: user pastes a 16-word Connections board, picks a solver, watches Snap-GVC run. Minimal UI, strong demo value.
- [ ] Post to X (tag co-authors, plus AK, Philipp Schmid, relevant HF folks), LinkedIn, r/MachineLearning.

**Exit criterion:** blog post published, Space live, post has at least 1 retweet from someone relevant. Repo README updated to link all of the above.

---

## Cadence

- Push something every weekend. Even a stub. The goal is to never have a "last commit was 3 weeks ago" moment on the GitHub profile between now and M6.
- Tag releases at the end of M2, M3, M4, M5 (`v0.2` through `v0.5`) so the repo has clean checkpoint points a hiring manager can skim.

## What I will deliberately NOT do

- No fine-tuning. This project is about inference-time reasoning, not training. Keeps compute budget sane.
- No leaderboard-chasing. Goal is to report honest numbers on two open models across all five strategies, not to crown a winner.
- No web UI beyond the final HF Space. Time on polish is time not spent on results.
