#!/usr/bin/env python3
"""Launch and monitor a LoRA fine-tuning job on Together AI.

Uploads the training data, starts a LoRA fine-tuning job on
Meta-Llama-3.1-8B-Instruct, and polls until completion.

Prerequisites::

    pip install together
    export TOGETHER_API_KEY=your_key_here

Usage::

    # Launch fine-tuning with defaults
    python scripts/finetune.py

    # Custom training file and epochs
    python scripts/finetune.py --train-file data/finetune/train.jsonl --epochs 3

    # Just check status of a running job
    python scripts/finetune.py --status ft-abc123

    # List all fine-tuning jobs
    python scripts/finetune.py --list
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _get_client():
    """Create a Together AI client."""
    try:
        from together import Together
    except ImportError:
        print("ERROR: 'together' package not installed. Run: pip install together")
        sys.exit(1)

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: TOGETHER_API_KEY environment variable not set.")
        print("Get your API key from https://api.together.xyz/settings/api-keys")
        sys.exit(1)

    return Together(api_key=api_key)


def upload_file(client, filepath: str) -> str:
    """Upload a JSONL file and return the file ID."""
    print(f"Uploading {filepath}...")
    result = client.files.upload(file=filepath, purpose="fine-tune")
    file_id = result.id
    print(f"Uploaded: {file_id}")
    return file_id


def launch_job(
    client,
    file_id: str,
    *,
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    lora_rank: int = 16,
    suffix: str = "gvc-connections",
) -> str:
    """Launch a LoRA fine-tuning job and return the job ID."""
    print(f"\nLaunching LoRA fine-tuning job...")
    print(f"  Model:         {model}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  LoRA rank:     {lora_rank}")
    print(f"  Suffix:        {suffix}")

    response = client.fine_tuning.create(
        training_file=file_id,
        model=model,
        n_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora=True,
        lora_r=lora_rank,
        suffix=suffix,
        train_on_inputs=False,  # Only train on assistant responses
    )

    job_id = response.id
    print(f"\nJob launched: {job_id}")
    return job_id


def check_status(client, job_id: str) -> dict:
    """Check the status of a fine-tuning job."""
    job = client.fine_tuning.retrieve(id=job_id)
    return {
        "id": job.id,
        "status": job.status,
        "model": getattr(job, "model", "N/A"),
        "output_name": getattr(job, "output_name", None),
        "events": getattr(job, "events", []),
    }


def poll_until_done(client, job_id: str, poll_interval: int = 60) -> dict:
    """Poll a fine-tuning job until it completes or fails."""
    print(f"\nPolling job {job_id} every {poll_interval}s...")
    print("(You can safely Ctrl+C and check later with --status)\n")

    while True:
        info = check_status(client, job_id)
        status = info["status"]
        print(f"  [{time.strftime('%H:%M:%S')}] Status: {status}")

        if status in ("completed", "succeeded"):
            output_name = info.get("output_name")
            print(f"\nFine-tuning complete!")
            print(f"Model name: {output_name}")
            print(f"\nTo use in gvc-local, run:")
            print(f"  gvc-local basic llama-3.1-8b --provider together \\")
            print(f"    --base-url https://api.together.xyz/v1")
            print(f"\n(The model will be available as: {output_name})")
            return info

        if status in ("failed", "cancelled", "error"):
            print(f"\nJob {status}.")
            events = info.get("events", [])
            if events:
                print("Recent events:")
                for ev in events[-5:]:
                    print(f"  - {ev}")
            return info

        time.sleep(poll_interval)


def list_jobs(client) -> None:
    """List all fine-tuning jobs."""
    jobs = client.fine_tuning.list()
    if not jobs.data:
        print("No fine-tuning jobs found.")
        return

    print(f"{'ID':<30} {'Status':<15} {'Model':<40}")
    print("-" * 85)
    for job in jobs.data:
        print(f"{job.id:<30} {job.status:<15} {getattr(job, 'model', 'N/A'):<40}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch LoRA fine-tuning on Together AI."
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/finetune/train.jsonl",
        help="Path to training JSONL file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
        help="Base model for fine-tuning.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5).",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="gvc-connections",
        help="Model name suffix (default: gvc-connections).",
    )
    parser.add_argument(
        "--no-poll",
        action="store_true",
        help="Don't poll for completion — just launch and exit.",
    )
    parser.add_argument(
        "--status",
        type=str,
        default=None,
        metavar="JOB_ID",
        help="Check status of an existing job instead of launching.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all fine-tuning jobs.",
    )

    args = parser.parse_args()
    client = _get_client()

    # Mode: list jobs
    if args.list:
        list_jobs(client)
        return

    # Mode: check status
    if args.status:
        info = check_status(client, args.status)
        print(f"Job ID:  {info['id']}")
        print(f"Status:  {info['status']}")
        print(f"Model:   {info['model']}")
        if info.get("output_name"):
            print(f"Output:  {info['output_name']}")
        return

    # Mode: launch fine-tuning
    if not os.path.exists(args.train_file):
        print(f"ERROR: Training file not found: {args.train_file}")
        print("Run `python scripts/data_prep.py` first to generate training data.")
        sys.exit(1)

    # Count examples for cost estimate
    with open(args.train_file) as f:
        n_examples = sum(1 for _ in f)
    print(f"Training file: {args.train_file} ({n_examples} examples)")

    file_id = upload_file(client, args.train_file)
    job_id = launch_job(
        client,
        file_id,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        suffix=args.suffix,
    )

    if args.no_poll:
        print(f"\nJob launched. Check status later with:")
        print(f"  python scripts/finetune.py --status {job_id}")
        return

    poll_until_done(client, job_id)


if __name__ == "__main__":
    main()
