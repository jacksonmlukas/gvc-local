### Solver comparison (overall, with bootstrap 95% CIs)

Bootstrapped 95% CIs from 5,000 resamples.

| Solver | n | Solve rate | Strikes / puzzle |
|---|---:|---|---|
| basic_llama8b_groq | 10 | 20.0% [0.0, 50.0] | 18.80 [17.10, 20.00] |
| cot_llama8b_groq | 10 | 30.0% [0.0, 60.0] | 16.70 [13.70, 19.40] |
| finetuned_lora_heldout | 50 | 12.0% [4.0, 22.0] | 18.78 [17.66, 19.68] |
| gvc_llama8b_groq | 10 | 60.0% [30.0, 90.0] | 7.50 [3.00, 12.00] |
| snap_gvc_llama8b_groq | 10 | 60.0% [30.0, 90.0] | 8.70 [5.00, 12.90] |
| snap_gvc_v3_llama8b_groq | 10 | 40.0% [10.0, 70.0] | 15.30 [10.70, 19.40] |

### basic_llama8b_groq — stratified results

Bootstrapped 95% CIs from 5,000 resamples. Strata assigned by `gvc_local.eval.tagger` (heuristic, not hand-labelled).

| Stratum | n | Solve rate | Strikes / puzzle |
|---|---:|---|---|
| OVERALL | 10 | 20.0% [0.0, 50.0] | 18.80 [17.10, 20.00] |
| wordplay | 2 | 0.0% [0.0, 0.0] | 20.00 [20.00, 20.00] |
| tag-fillin | 2 | 0.0% [0.0, 0.0] | 20.00 [20.00, 20.00] |
| cultural | 1 | 0.0% [0.0, 0.0] | 20.00 [20.00, 20.00] |
| category | 5 | 40.0% [0.0, 80.0] | 17.60 [14.80, 20.00] |

### cot_llama8b_groq — stratified results

Bootstrapped 95% CIs from 5,000 resamples. Strata assigned by `gvc_local.eval.tagger` (heuristic, not hand-labelled).

| Stratum | n | Solve rate | Strikes / puzzle |
|---|---:|---|---|
| OVERALL | 10 | 30.0% [0.0, 60.0] | 16.70 [13.70, 19.40] |
| wordplay | 2 | 0.0% [0.0, 0.0] | 20.00 [20.00, 20.00] |
| tag-fillin | 2 | 50.0% [0.0, 100.0] | 19.00 [18.00, 20.00] |
| cultural | 1 | 0.0% [0.0, 0.0] | 20.00 [20.00, 20.00] |
| category | 5 | 40.0% [0.0, 80.0] | 13.80 [9.40, 18.20] |

### finetuned_lora_heldout — stratified results

Bootstrapped 95% CIs from 5,000 resamples. Strata assigned by `gvc_local.eval.tagger` (heuristic, not hand-labelled).

| Stratum | n | Solve rate | Strikes / puzzle |
|---|---:|---|---|
| OVERALL | 50 | 12.0% [4.0, 22.0] | 18.78 [17.66, 19.68] |
| wordplay | 5 | 0.0% [0.0, 0.0] | 20.00 [20.00, 20.00] |
| tag-fillin | 15 | 20.0% [0.0, 40.0] | 17.87 [14.80, 20.00] |
| cultural | 12 | 0.0% [0.0, 0.0] | 20.00 [20.00, 20.00] |
| category | 18 | 16.7% [0.0, 33.3] | 18.39 [16.67, 20.00] |

### gvc_llama8b_groq — stratified results

Bootstrapped 95% CIs from 5,000 resamples. Strata assigned by `gvc_local.eval.tagger` (heuristic, not hand-labelled).

| Stratum | n | Solve rate | Strikes / puzzle |
|---|---:|---|---|
| OVERALL | 10 | 60.0% [30.0, 90.0] | 7.50 [3.00, 12.00] |
| wordplay | 2 | 100.0% [100.0, 100.0] | 13.50 [13.00, 14.00] |
| tag-fillin | 2 | 50.0% [0.0, 100.0] | 9.50 [2.00, 17.00] |
| cultural | 1 | 100.0% [100.0, 100.0] | 7.00 [7.00, 7.00] |
| category | 5 | 40.0% [0.0, 80.0] | 4.40 [0.20, 12.20] |

### snap_gvc_llama8b_groq — stratified results

Bootstrapped 95% CIs from 5,000 resamples. Strata assigned by `gvc_local.eval.tagger` (heuristic, not hand-labelled).

| Stratum | n | Solve rate | Strikes / puzzle |
|---|---:|---|---|
| OVERALL | 10 | 60.0% [30.0, 90.0] | 8.70 [5.00, 12.90] |
| wordplay | 2 | 100.0% [100.0, 100.0] | 4.00 [2.00, 6.00] |
| tag-fillin | 2 | 100.0% [100.0, 100.0] | 6.00 [4.00, 8.00] |
| cultural | 1 | 0.0% [0.0, 0.0] | 20.00 [20.00, 20.00] |
| category | 5 | 40.0% [0.0, 80.0] | 9.40 [4.40, 15.00] |

### snap_gvc_v3_llama8b_groq — stratified results

Bootstrapped 95% CIs from 5,000 resamples. Strata assigned by `gvc_local.eval.tagger` (heuristic, not hand-labelled).

| Stratum | n | Solve rate | Strikes / puzzle |
|---|---:|---|---|
| OVERALL | 10 | 40.0% [10.0, 70.0] | 15.30 [10.70, 19.40] |
| wordplay | 2 | 50.0% [0.0, 100.0] | 12.00 [4.00, 20.00] |
| tag-fillin | 2 | 0.0% [0.0, 0.0] | 20.00 [20.00, 20.00] |
| cultural | 1 | 0.0% [0.0, 0.0] | 20.00 [20.00, 20.00] |
| category | 5 | 60.0% [20.0, 100.0] | 13.80 [6.60, 19.60] |
