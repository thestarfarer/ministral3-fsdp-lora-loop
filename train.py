#!/usr/bin/env python3
"""
Custom FSDP + LoRA training script for Ministral 3 models.
Works around broken axolotl implementations for this architecture.

Usage:
    torchrun --nproc_per_node=4 train.py --train-file data.jsonl
"""

import os
import json
import math
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import Mistral3ForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from tqdm import tqdm
import functools
import argparse
from datetime import datetime

# Config
CONFIG = {
    "model_name": "mistralai/Ministral-3-14B-Base-2512",  # Or path to local model
    "train_file": "train.jsonl",
    "output_dir": "./output",

    # LoRA
    "lora_r": 256,
    "lora_alpha": 256,
    "lora_dropout": 0.0,
    "use_rslora": True,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    # Training
    "max_seq_len": 8192,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 8,  # effective batch = world_size * micro_batch * grad_accum
    "num_epochs": 1,
    "learning_rate": 2e-5,
    "min_lr_ratio": 0.1,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,

    # Eval/Save
    "val_size": 256,
    "eval_steps": 50,
    "save_steps": 100,
    "logging_steps": 1,

    # WandB
    "wandb_project": "ministral-finetune",

    # Optional: resume from step (manual fast-forward)
    "seed": 42,
    "start_step": 0,
    "skip_batches": 0,
}


class TextDataset(Dataset):
    """Simple text dataset from JSONL with 'text' field"""

    def __init__(self, texts, processor, max_length):
        self.texts = texts
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize
        encoding = self.processor(
            text=text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }


def collate_fn(batch, max_length, pad_token_id):
    """Pad batch to max_length"""
    input_ids = []
    attention_masks = []
    labels = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_length - seq_len

        if pad_len > 0:
            input_ids.append(torch.cat([
                item["input_ids"],
                torch.full((pad_len,), pad_token_id, dtype=torch.long)
            ]))
            attention_masks.append(torch.cat([
                item["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long)
            ]))
            labels.append(torch.cat([
                item["labels"],
                torch.full((pad_len,), -100, dtype=torch.long)  # Ignore padding in loss
            ]))
        else:
            input_ids.append(item["input_ids"][:max_length])
            attention_masks.append(item["attention_mask"][:max_length])
            labels.append(item["labels"][:max_length])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels)
    }


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Cosine schedule with warmup and minimum LR"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), dist.get_rank()


def cleanup_distributed():
    dist.destroy_process_group()


def log_rank0(msg, rank):
    if rank == 0:
        print(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default=CONFIG["train_file"])
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    # Distributed setup
    local_rank, world_size, global_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    log_rank0(f"Starting training with {world_size} GPUs", global_rank)

    # Set seed
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])

    # Load processor
    log_rank0("Loading processor...", global_rank)
    processor = AutoProcessor.from_pretrained(CONFIG["model_name"])
    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    # Load data
    log_rank0(f"Loading data from {args.train_file}...", global_rank)
    with open(args.train_file, "r") as f:
        all_texts = [json.loads(line)["text"] for line in f]

    # Split train/val
    val_texts = all_texts[-CONFIG["val_size"]:]
    train_texts = all_texts[:-CONFIG["val_size"]]
    log_rank0(f"Train: {len(train_texts)}, Val: {len(val_texts)}", global_rank)

    dist.barrier()

    # Create datasets
    train_dataset = TextDataset(train_texts, processor, CONFIG["max_seq_len"])
    val_dataset = TextDataset(val_texts, processor, CONFIG["max_seq_len"])

    # Distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True
    )

    collate = functools.partial(collate_fn, max_length=CONFIG["max_seq_len"], pad_token_id=pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["micro_batch_size"],
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["micro_batch_size"],
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True
    )

    # Load model
    log_rank0("Loading model...", global_rank)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Apply LoRA
    log_rank0("Applying LoRA...", global_rank)
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=CONFIG["use_rslora"],
    )
    model = get_peft_model(model, lora_config)

    # Cast LoRA parameters to bfloat16 to match base model (required for FSDP)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log_rank0(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)", global_rank)

    # FSDP wrap
    log_rank0("Wrapping with FSDP...", global_rank)

    # Get the decoder layer class for wrapping
    from transformers.models.ministral3.modeling_ministral3 import Ministral3DecoderLayer

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Ministral3DecoderLayer}
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        device_id=local_rank,
        use_orig_params=True,  # Required for LoRA
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.999),
    )

    # Calculate steps
    steps_per_epoch = len(train_loader) // CONFIG["gradient_accumulation_steps"]
    total_steps = steps_per_epoch * CONFIG["num_epochs"]
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    
    # Warmup: use warmup_steps if set, otherwise calculate from warmup_ratio
    if CONFIG.get("warmup_steps"):
        warmup_steps = CONFIG["warmup_steps"]
    else:
        warmup_steps = int(total_steps * CONFIG.get("warmup_ratio", 0.03))

    log_rank0(f"Total steps: {total_steps}, Warmup: {warmup_steps}", global_rank)

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, CONFIG["min_lr_ratio"]
    )

    # WandB
    if global_rank == 0:
        wandb.init(
            project=CONFIG["wandb_project"],
            config={**CONFIG, "world_size": world_size, "total_steps": total_steps},
            name=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

    # Training loop
    log_rank0("Starting training...", global_rank)
    model.train()

    global_step = CONFIG.get("start_step", 0)  # Resume from start_step
    accumulated_loss = 0.0
    optimizer.zero_grad()
    skip_batches = CONFIG.get("skip_batches", 0)

    os.makedirs(args.output_dir, exist_ok=True)

    # Eval on start
    if CONFIG.get("eval_on_start", False):
        log_rank0("Running eval on start...", global_rank)
        model.eval()
        val_loss = 0.0
        val_tokens = 0

        with torch.no_grad():
            for val_batch in val_loader:
                val_input_ids = val_batch["input_ids"].to(device)
                val_attention_mask = val_batch["attention_mask"].to(device)
                val_labels = val_batch["labels"].to(device)

                val_outputs = model(
                    input_ids=val_input_ids,
                    attention_mask=val_attention_mask,
                    labels=val_labels,
                )

                n_tokens = (val_labels != -100).sum().item()
                val_loss += val_outputs.loss.item() * n_tokens
                val_tokens += n_tokens

        avg_val_loss = val_loss / val_tokens if val_tokens > 0 else 0

        if global_rank == 0:
            wandb.log({
                "eval/loss": avg_val_loss,
            }, step=global_step)
            log_rank0(f"Step {global_step} (eval on start): val_loss = {avg_val_loss:.4f}", global_rank)

        model.train()

    for epoch in range(CONFIG["num_epochs"]):
        train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, disable=global_rank != 0, desc=f"Epoch {epoch+1}")

        # Fast-forward logging
        if skip_batches > 0:
            log_rank0(f"Fast-forwarding {skip_batches} batches to skip already-seen data...", global_rank)

        for batch_idx, batch in enumerate(pbar):
            # Fast-forward: skip already-seen batches
            if batch_idx < skip_batches:
                if batch_idx % 200 == 0:
                    log_rank0(f"Skipping batch {batch_idx}/{skip_batches}...", global_rank)
                continue

            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / CONFIG["gradient_accumulation_steps"]

            # Backward
            loss.backward()
            accumulated_loss += loss.item()

            # Optimizer step
            if (batch_idx + 1) % CONFIG["gradient_accumulation_steps"] == 0:
                # Gradient clipping - capture grad norm before clipping
                grad_norm = model.clip_grad_norm_(CONFIG["max_grad_norm"])

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Logging
                if global_step % CONFIG["logging_steps"] == 0:
                    avg_loss = accumulated_loss
                    lr = scheduler.get_last_lr()[0]

                    if global_rank == 0:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                        }, step=global_step)
                        pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                        log_rank0(f"Step {global_step}/{total_steps}: loss={avg_loss:.4f}, grad_norm={grad_norm:.2f}, lr={lr:.2e}", global_rank)

                    accumulated_loss = 0.0

                # Evaluation
                if global_step % CONFIG["eval_steps"] == 0:
                    model.eval()
                    val_loss = 0.0
                    val_tokens = 0

                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_input_ids = val_batch["input_ids"].to(device)
                            val_attention_mask = val_batch["attention_mask"].to(device)
                            val_labels = val_batch["labels"].to(device)

                            val_outputs = model(
                                input_ids=val_input_ids,
                                attention_mask=val_attention_mask,
                                labels=val_labels,
                            )

                            # Count non-ignored tokens
                            n_tokens = (val_labels != -100).sum().item()
                            val_loss += val_outputs.loss.item() * n_tokens
                            val_tokens += n_tokens

                    avg_val_loss = val_loss / val_tokens if val_tokens > 0 else 0

                    if global_rank == 0:
                        wandb.log({
                            "eval/loss": avg_val_loss,
                        }, step=global_step)
                        log_rank0(f"Step {global_step}: val_loss = {avg_val_loss:.4f}", global_rank)

                    model.train()

                # Save checkpoint
                if global_step % CONFIG["save_steps"] == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    # Gather full state dict from all FSDP shards
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                        state_dict = model.state_dict()

                        if global_rank == 0:
                            os.makedirs(save_path, exist_ok=True)
                            # Extract only LoRA parameters (keys containing 'lora_')
                            lora_state_dict = {k: v for k, v in state_dict.items() if 'lora_' in k}
                            # Save using safetensors
                            from safetensors.torch import save_file
                            save_file(lora_state_dict, os.path.join(save_path, "adapter_model.safetensors"))
                            # Save adapter config
                            lora_config.save_pretrained(save_path)
                            log_rank0(f"Saved checkpoint to {save_path}", global_rank)

                # Check max steps
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Final save - gather full state dict from all FSDP shards
    final_path = os.path.join(args.output_dir, "final")
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()

        if global_rank == 0:
            os.makedirs(final_path, exist_ok=True)
            # Extract only LoRA parameters (keys containing 'lora_')
            lora_state_dict = {k: v for k, v in state_dict.items() if 'lora_' in k}
            # Save using safetensors
            from safetensors.torch import save_file
            save_file(lora_state_dict, os.path.join(final_path, "adapter_model.safetensors"))
            # Save adapter config and processor
            lora_config.save_pretrained(final_path)
            processor.save_pretrained(final_path)
            log_rank0(f"Saved final model to {final_path}", global_rank)
            wandb.finish()

    cleanup_distributed()
    log_rank0("Training complete!", global_rank)


if __name__ == "__main__":
    main()
