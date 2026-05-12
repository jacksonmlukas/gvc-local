#!/usr/bin/env python3
"""Evaluate the fine-tuned LoRA model on held-out Connections puzzles.

Designed to run on Google Colab (free T4 GPU) or any machine with a GPU.
Loads the base Llama 3.1 8B + LoRA adapter from HuggingFace, then runs
the basic solver loop on test puzzles.

Usage (Colab or local GPU)::

    pip install torch transformers peft accelerate bitsandbytes requests
    python scripts/eval_finetuned.py --max-puzzles 50

Environment::

    export HF_TOKEN=hf_xxx  # needed for gated Llama model access
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    adapter_repo: str = "jacksonlukas/gvc-connections-lora",
    load_in_4bit: bool = True,
):
    """Load base model with LoRA adapter merged."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading base model: {base_model}")
    print(f"Loading adapter: {adapter_repo}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    # Load and merge the LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_repo)
    model.eval()

    print("Model loaded successfully!")
    return model, tokenizer


# ---------------------------------------------------------------------------
# System prompt (must match training data format from data_prep.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert at the NYT Connections puzzle. Given a list of 16 words \
(or fewer if some groups have been solved), identify groups of 4 related \
words and their category names.

You MUST format your response EXACTLY as follows:

<UNDERSTANDING_OF_BOARD>
Group1: word1, word2, word3, word4
Group2: word5, word6, word7, word8
Group3: word9, word10, word11, word12
Group4: word13, word14, word15, word16
<END_UNDERSTANDING_OF_BOARD>

<GUESS_FOR_THIS_ROUND>
Group: word_a, word_b, word_c, word_d
Category: category_name
<END_GUESS_FOR_THIS_ROUND>

Pick the group you are MOST confident about as your guess."""


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_response(model, tokenizer, user_prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a response using the fine-tuned model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_guess(reply: str) -> tuple[list[str], str] | None:
    """Parse the model's structured reply into (group, category)."""
    guess_match = re.search(
        r"<GUESS_FOR_THIS_ROUND>(.*?)<END_GUESS_FOR_THIS_ROUND>",
        reply, re.DOTALL,
    )
    if not guess_match:
        return None

    guess_text = guess_match.group(1).strip()
    group_m = re.search(r"Group:\s*(.+)", guess_text, re.IGNORECASE)
    cat_m = re.search(r"Category:\s*(.+)", guess_text, re.IGNORECASE)

    if not group_m or not cat_m:
        return None

    group = [w.strip().upper().replace(",", "") for w in re.split(r",\s*", group_m.group(1)) if w.strip()]
    category = cat_m.group(1).strip()

    if len(group) != 4:
        return None

    return group, category


# ---------------------------------------------------------------------------
# Puzzle evaluation
# ---------------------------------------------------------------------------

def load_puzzles():
    """Fetch all Connections puzzles from the public dataset."""
    import requests

    url = (
        "https://raw.githubusercontent.com/Eyefyre/NYT-Connections-Answers/"
        "refs/heads/main/connections.json"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def eval_puzzle(model, tokenizer, puzzle: dict, max_strikes: int = 20) -> dict:
    """Evaluate the model on a single puzzle. Returns result dict."""
    categories = puzzle["answers"]
    # Build category lookup
    cat_lookup = {}
    for cat in categories:
        key = frozenset(m.upper() for m in cat["members"])
        cat_lookup[key] = cat

    remaining_cats = list(categories)
    all_words = [w for cat in remaining_cats for w in cat["members"]]
    random.shuffle(all_words)

    strikes = 0
    solved = 0
    failed_guesses = []

    while remaining_cats and strikes < max_strikes:
        # Build prompt
        remaining_words = [w for cat in remaining_cats for w in cat["members"]]
        random.shuffle(remaining_words)
        user_prompt = f"Words: {', '.join(remaining_words)}"

        # Generate
        reply = generate_response(model, tokenizer, user_prompt)
        result = parse_guess(reply)

        if result is None:
            strikes += 1
            continue

        group, category = result

        # Check against remaining categories
        guess_set = frozenset(group)
        matched = False
        for i, cat in enumerate(remaining_cats):
            cat_set = frozenset(m.upper() for m in cat["members"])
            if guess_set == cat_set:
                solved += 1
                remaining_cats.pop(i)
                matched = True
                break

        if not matched:
            strikes += 1
            sorted_guess = sorted(group)
            if sorted_guess in failed_guesses:
                # Repeated guess, skip
                continue
            failed_guesses.append(sorted_guess)

    return {
        "solved": solved == 4,
        "categories_solved": solved,
        "strikes": strikes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LoRA model")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter", default="jacksonlukas/gvc-connections-lora")
    parser.add_argument("--max-puzzles", type=int, default=50)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/finetuned_lora_eval.jsonl")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_repo=args.adapter,
        load_in_4bit=not args.no_4bit,
    )

    # Load puzzles
    print("Loading puzzles...")
    puzzles = load_puzzles()
    end = min(args.start + args.max_puzzles, len(puzzles))
    puzzles = puzzles[args.start:end]
    print(f"Evaluating puzzles {args.start} to {end} ({len(puzzles)} total)")

    # Run eval
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_solved = 0
    t0 = time.time()

    with open(out_path, "w") as f:
        for i, puzzle in enumerate(puzzles):
            print(f"\n--- Puzzle {args.start + i} ({i+1}/{len(puzzles)}) ---")
            result = eval_puzzle(model, tokenizer, puzzle)
            result["puzzle_id"] = args.start + i

            if result["solved"]:
                total_solved += 1
                print(f"  SOLVED ({result['categories_solved']}/4, {result['strikes']} strikes)")
            else:
                print(f"  FAILED ({result['categories_solved']}/4, {result['strikes']} strikes)")

            f.write(json.dumps(result) + "\n")
            f.flush()

    elapsed = time.time() - t0
    rate = total_solved / len(puzzles) * 100

    print(f"\n{'='*50}")
    print(f"Model:    {args.adapter}")
    print(f"Puzzles:  {len(puzzles)}")
    print(f"Solved:   {total_solved}/{len(puzzles)} ({rate:.1f}%)")
    print(f"Time:     {elapsed:.1f}s")
    print(f"Results:  {out_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
