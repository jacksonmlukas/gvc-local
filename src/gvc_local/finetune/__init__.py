"""Fine-tuning pipeline for GVC-local agent models.

Submodules
----------
data_prep
    Convert solver traces (JSONL) into supervised fine-tuning chat datasets.
train
    LoRA / QLoRA training with TRL's SFTTrainer.
merge
    Merge LoRA adapters back into the base model for vLLM serving.
"""

from gvc_local.finetune.data_prep import (
    GroupInfo,
    TraceRecord,
    Turn,
    load_traces,
    save_huggingface,
    save_jsonl,
    split_dataset,
    traces_to_conversations,
)
from gvc_local.finetune.merge import merge_and_save
from gvc_local.finetune.train import FinetuneConfig, train

__all__ = [
    # data_prep
    "TraceRecord",
    "Turn",
    "GroupInfo",
    "load_traces",
    "traces_to_conversations",
    "split_dataset",
    "save_jsonl",
    "save_huggingface",
    # train
    "FinetuneConfig",
    "train",
    # merge
    "merge_and_save",
]
