"""LoRA / QLoRA fine-tuning for GVC-local agent models.

Trains a causal-LM with parameter-efficient LoRA adapters on the chat-format
datasets produced by :mod:`gvc_local.finetune.data_prep`.  Designed for
single-GPU (24 GB+) training of Llama 3.1 8B or Qwen 2.5 7B via 4-bit QLoRA.

Usage::

    python -m gvc_local.finetune.train --config configs/finetune.yaml

The YAML schema is documented in ``configs/finetune.yaml``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    trust_remote_code: bool = False


@dataclass
class LoRAHyperparams:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    task_type: str = "CAUSAL_LM"
    bias: str = "none"


@dataclass
class QuantConfig:
    enabled: bool = True
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class DataSettings:
    trace_path: str = "data/traces/solver_traces.jsonl"
    output_dir: str = "data/sft"
    val_ratio: float = 0.1
    max_seq_length: int = 2048
    include_near_misses: bool = False
    roles: list[str] = field(default_factory=lambda: ["guesser", "validator", "snap_guesser"])


@dataclass
class WandbSettings:
    project: str = "gvc-local-finetune"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=lambda: ["gvc-local", "lora"])


@dataclass
class FinetuneConfig:
    """Top-level configuration assembled from the YAML file."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAHyperparams = field(default_factory=LoRAHyperparams)
    quantization: QuantConfig = field(default_factory=QuantConfig)
    data: DataSettings = field(default_factory=DataSettings)
    training: dict[str, Any] = field(default_factory=dict)
    wandb: WandbSettings = field(default_factory=WandbSettings)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FinetuneConfig":
        """Load configuration from a YAML file."""
        with open(path) as fh:
            raw = yaml.safe_load(fh)

        return cls(
            model=ModelConfig(**raw.get("model", {})),
            lora=LoRAHyperparams(**raw.get("lora", {})),
            quantization=QuantConfig(**raw.get("quantization", {})),
            data=DataSettings(**raw.get("data", {})),
            training=raw.get("training", {}),
            wandb=WandbSettings(**raw.get("wandb", {})),
        )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_sft_dataset(jsonl_path: str | Path) -> list[dict]:
    """Read a JSONL file produced by data_prep and return chat samples."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"SFT dataset not found: {path}")

    samples: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # The conversations field may be a JSON string (HF round-trip) or a list.
            convos = obj["conversations"]
            if isinstance(convos, str):
                convos = json.loads(convos)
            samples.append({"conversations": convos})
    return samples


def _format_chat(
    sample: dict,
    tokenizer: AutoTokenizer,
) -> str:
    """Apply the tokenizer's chat template to a sample's conversations."""
    messages = sample["conversations"]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


# ---------------------------------------------------------------------------
# Model + tokenizer construction
# ---------------------------------------------------------------------------


def build_quantization_config(qcfg: QuantConfig) -> BitsAndBytesConfig | None:
    """Create a BitsAndBytesConfig if quantisation is enabled."""
    if not qcfg.enabled:
        return None

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(qcfg.bnb_4bit_compute_dtype, torch.bfloat16)

    return BitsAndBytesConfig(
        load_in_4bit=qcfg.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=qcfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=qcfg.bnb_4bit_use_double_quant,
    )


def load_model_and_tokenizer(
    cfg: FinetuneConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model (optionally quantised) and its tokenizer."""
    bnb_config = build_quantization_config(cfg.quantization)

    logger.info("Loading model: %s (4-bit=%s)", cfg.model.name, cfg.quantization.enabled)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=torch.bfloat16 if not cfg.quantization.enabled else None,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    if cfg.quantization.enabled:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.training.get("gradient_checkpointing", True),
        )

    return model, tokenizer


def apply_lora(model: AutoModelForCausalLM, lora_cfg: LoRAHyperparams) -> AutoModelForCausalLM:
    """Wrap the model with LoRA adapters."""
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=lora_cfg.target_modules,
        task_type=lora_cfg.task_type,
        bias=lora_cfg.bias,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_training_args(cfg: FinetuneConfig) -> TrainingArguments:
    """Construct HuggingFace ``TrainingArguments`` from the config dict."""
    ta_kwargs: dict[str, Any] = {**cfg.training}

    # Ensure output_dir is set.
    ta_kwargs.setdefault("output_dir", "checkpoints/lora")

    # Remove keys that are not valid TrainingArguments fields but may live
    # in the YAML for convenience.
    for extra_key in ("max_seq_length",):
        ta_kwargs.pop(extra_key, None)

    return TrainingArguments(**ta_kwargs)


def train(cfg: FinetuneConfig) -> None:
    """End-to-end fine-tuning run.

    1. Load & prepare model + tokenizer.
    2. Load SFT data (JSONL produced by ``data_prep``).
    3. Train with TRL's ``SFTTrainer``.
    4. Save the final LoRA adapter.
    """
    # ── W&B setup ────────────────────────────────────────────────────
    if cfg.training.get("report_to") == "wandb":
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)
        if cfg.wandb.entity:
            os.environ.setdefault("WANDB_ENTITY", cfg.wandb.entity)
        if cfg.wandb.run_name:
            os.environ.setdefault("WANDB_NAME", cfg.wandb.run_name)

    # ── Model ────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg.lora)

    # ── Data ─────────────────────────────────────────────────────────
    sft_dir = Path(cfg.data.output_dir)
    train_path = sft_dir / "train.jsonl"
    val_path = sft_dir / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Run `python -m gvc_local.finetune.data_prep` first."
        )

    train_data = _load_sft_dataset(train_path)
    val_data = _load_sft_dataset(val_path) if val_path.exists() else None

    logger.info(
        "Loaded %d train samples%s",
        len(train_data),
        f" / {len(val_data)} val samples" if val_data else "",
    )

    # Convert to text via the tokenizer's chat template.
    def formatting_func(examples: dict) -> list[str]:
        """Format a batch of samples into templated chat strings."""
        outputs = []
        for convo in examples["conversations"]:
            messages = convo if isinstance(convo, list) else json.loads(convo)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            outputs.append(text)
        return outputs

    # Wrap in HF Dataset objects for SFTTrainer.
    from datasets import Dataset

    def _to_hf_dataset(samples: list[dict]) -> Dataset:
        convos = [
            json.dumps(s["conversations"]) if isinstance(s["conversations"], list) else s["conversations"]
            for s in samples
        ]
        return Dataset.from_dict({"conversations": convos})

    train_dataset = _to_hf_dataset(train_data)
    val_dataset = _to_hf_dataset(val_data) if val_data else None

    # ── Trainer ──────────────────────────────────────────────────────
    training_args = build_training_args(cfg)

    # Detect the assistant response template for completion-only masking.
    # Llama 3.1 uses "<|start_header_id|>assistant<|end_header_id|>\n\n"
    # Qwen 2.5 uses "<|im_start|>assistant\n"
    model_lower = cfg.model.name.lower()
    if "qwen" in model_lower:
        response_template = "<|im_start|>assistant\n"
    else:
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        data_collator=collator,
        max_seq_length=cfg.data.max_seq_length,
        tokenizer=tokenizer,
        packing=False,
    )

    # ── Run ──────────────────────────────────────────────────────────
    logger.info("Starting training...")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────
    final_dir = Path(training_args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved final LoRA adapter to %s", final_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("train")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the YAML config file (e.g. configs/finetune.yaml).",
)
def main(config_path: str) -> None:
    """Fine-tune a causal-LM with LoRA / QLoRA on GVC solver traces."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = FinetuneConfig.from_yaml(config_path)
    train(cfg)


if __name__ == "__main__":
    main()
