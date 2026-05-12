#!/usr/bin/env python3
"""Upload LoRA adapter to HuggingFace and register on Together AI.

Usage::

    export HF_TOKEN=hf_xxx
    export TOGETHER_API_KEY=xxx
    python scripts/upload_adapter.py --adapter-dir /tmp/adapter --hf-repo jacksonmlukas/gvc-connections-lora
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def fix_adapter_config(adapter_dir: str) -> None:
    """Update adapter_config.json to point to the public HF base model."""
    config_path = Path(adapter_dir) / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Together AI uses an internal model path — replace with the public HF path
    old_base = config.get("base_model_name_or_path", "")
    new_base = "meta-llama/Llama-3.1-8B-Instruct"
    if old_base != new_base:
        print(f"Updating base_model_name_or_path:")
        print(f"  {old_base} -> {new_base}")
        config["base_model_name_or_path"] = new_base
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


def upload_to_hf(adapter_dir: str, repo_id: str, token: str) -> str:
    """Upload adapter files to a HuggingFace model repo."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=token)

    # Create the repo (if it doesn't exist)
    print(f"Creating HF repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Upload all files in the adapter directory
    print(f"Uploading adapter files from {adapter_dir}...")
    api.upload_folder(
        folder_path=adapter_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload GVC-Connections LoRA adapter (Llama 3.1 8B)",
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Upload complete: {url}")
    return url


def register_on_together(repo_id: str, together_key: str) -> None:
    """Register the HF adapter on Together AI for serverless LoRA inference."""
    try:
        import requests
    except ImportError:
        print("ERROR: requests not installed.")
        sys.exit(1)

    adapter_name = repo_id.replace("/", "--")
    print(f"\nRegistering adapter on Together AI...")
    print(f"  Adapter name: {adapter_name}")
    print(f"  Base model: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    print(f"  Source: https://huggingface.co/{repo_id}")

    resp = requests.post(
        "https://api.together.xyz/v1/models",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {together_key}",
        },
        json={
            "model_name": adapter_name,
            "model_source": repo_id,
            "model_type": "adapter",
            "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "description": "GVC-Connections LoRA adapter: distills multi-agent puzzle-solving into single-pass inference",
        },
    )

    if resp.ok:
        result = resp.json()
        model_name = result.get("model_name", adapter_name)
        print(f"\nRegistered! Model name for inference: {model_name}")
        print(f"\nTo run eval:")
        print(f"  gvc-local basic llama-3.1-8b --provider together \\")
        print(f'    --model-override "{model_name}" \\')
        print(f"    --start 0 --end 10 --out results/finetuned_lora.jsonl")
    else:
        print(f"\nRegistration failed ({resp.status_code}):")
        print(resp.text)
        print("\nYou may need to register manually via the Together AI dashboard.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload LoRA adapter to HF + Together AI")
    parser.add_argument("--adapter-dir", type=str, default="/tmp/adapter",
                        help="Path to extracted adapter checkpoint")
    parser.add_argument("--hf-repo", type=str, required=True,
                        help="HuggingFace repo ID (e.g. jacksonmlukas/gvc-connections-lora)")
    parser.add_argument("--skip-together", action="store_true",
                        help="Skip Together AI registration")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.")
        sys.exit(1)

    together_key = os.environ.get("TOGETHER_API_KEY")
    if not together_key and not args.skip_together:
        print("WARNING: TOGETHER_API_KEY not set — skipping Together AI registration.")
        args.skip_together = True

    # Step 1: Fix the adapter config
    fix_adapter_config(args.adapter_dir)

    # Step 2: Upload to HuggingFace
    upload_to_hf(args.adapter_dir, args.hf_repo, hf_token)

    # Step 3: Register on Together AI
    if not args.skip_together:
        register_on_together(args.hf_repo, together_key)


if __name__ == "__main__":
    main()
