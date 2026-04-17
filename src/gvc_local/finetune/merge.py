"""Merge LoRA adapters into the base model for vLLM serving.

After fine-tuning produces a LoRA adapter checkpoint, this script merges the
adapter weights back into the base model and saves a standalone model that
can be loaded directly by vLLM (or any HuggingFace-compatible inference
framework) without PEFT at serving time.

Usage::

    python -m gvc_local.finetune.merge \\
        --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --adapter-path checkpoints/lora/final \\
        --output-dir models/llama-3.1-8b-gvc-merged

Then serve with vLLM::

    vllm serve models/llama-3.1-8b-gvc-merged --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def merge_and_save(
    base_model_name: str,
    adapter_path: str | Path,
    output_dir: str | Path,
    *,
    trust_remote_code: bool = False,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
) -> Path:
    """Load base model + LoRA adapter, merge weights, and save.

    Parameters
    ----------
    base_model_name:
        HuggingFace model ID or local path for the base model.
    adapter_path:
        Path to the LoRA adapter directory (contains ``adapter_config.json``).
    output_dir:
        Where to write the merged model and tokenizer.
    trust_remote_code:
        Passed through to ``from_pretrained``; required for some models (Qwen).
    push_to_hub:
        If True, push the merged model to the HuggingFace Hub.
    hub_repo_id:
        Repository ID for the Hub push (required when ``push_to_hub=True``).

    Returns
    -------
    Path
        The resolved output directory.
    """
    adapter_path = Path(adapter_path)
    output_dir = Path(output_dir)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    # ── Load base model in full precision ────────────────────────────
    logger.info("Loading base model: %s", base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )

    # ── Load tokenizer ───────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    # ── Attach and merge LoRA adapter ────────────────────────────────
    logger.info("Loading LoRA adapter from: %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    logger.info("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    # ── Save merged model ────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving merged model to: %s", output_dir)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    # ── Optional Hub push ────────────────────────────────────────────
    if push_to_hub:
        if hub_repo_id is None:
            raise ValueError("--hub-repo-id is required when --push-to-hub is set.")
        logger.info("Pushing merged model to Hub: %s", hub_repo_id)
        model.push_to_hub(hub_repo_id, safe_serialization=True)
        tokenizer.push_to_hub(hub_repo_id)

    logger.info("Merge complete.")
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("merge")
@click.option(
    "--base-model",
    required=True,
    help="HuggingFace model ID or local path (e.g. meta-llama/Meta-Llama-3.1-8B-Instruct).",
)
@click.option(
    "--adapter-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the LoRA adapter directory.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory for the merged model output.",
)
@click.option("--trust-remote-code", is_flag=True, default=False, help="Trust remote code (Qwen).")
@click.option("--push-to-hub", is_flag=True, default=False, help="Push merged model to HF Hub.")
@click.option("--hub-repo-id", default=None, help="Hub repository ID for push.")
def main(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    trust_remote_code: bool,
    push_to_hub: bool,
    hub_repo_id: str | None,
) -> None:
    """Merge a LoRA adapter into its base model for serving."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    merge_and_save(
        base_model_name=base_model,
        adapter_path=adapter_path,
        output_dir=output_dir,
        trust_remote_code=trust_remote_code,
        push_to_hub=push_to_hub,
        hub_repo_id=hub_repo_id,
    )


if __name__ == "__main__":
    main()
