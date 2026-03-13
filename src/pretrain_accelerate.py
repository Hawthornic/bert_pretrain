"""
BERT Pre-training with HuggingFace Accelerate for distributed training.
Supports multi-GPU, DeepSpeed, FSDP, etc.
"""

import argparse
import glob
import json
import logging
import math
import os
import pickle
import random
import time

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    BertConfig,
    BertForPreTraining,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)

from dataset import BertPretrainDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_documents(data_dir: str, max_shards: int = 0):
    """Load preprocessed document shards."""
    shard_files = sorted(glob.glob(os.path.join(data_dir, "documents_*.pkl")))
    if not shard_files:
        raise FileNotFoundError(f"No document shards found in {data_dir}")
    if max_shards > 0:
        shard_files = shard_files[:max_shards]

    documents = []
    for shard_file in shard_files:
        logger.info(f"Loading shard: {shard_file}")
        with open(shard_file, "rb") as f:
            documents.extend(pickle.load(f))
    logger.info(f"Loaded {len(documents)} documents from {len(shard_files)} shards")
    return documents


def train(args):
    # Initialize Accelerate
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if args.fp16 else "no",
        log_with="tensorboard",
        project_dir=args.output_dir,
    )

    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.init_trackers("bert_pretrain")

    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)

    # Model config
    if args.config_file:
        with open(args.config_file, "r") as f:
            config_dict = json.load(f)
        config = BertConfig(**config_dict)
    else:
        config = BertConfig()
    config.vocab_size = tokenizer.vocab_size

    # Create or load model
    if args.resume_from:
        model = BertForPreTraining.from_pretrained(args.resume_from)
    else:
        model = BertForPreTraining(config)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Load data
    all_documents = []
    for data_dir in args.data_dirs:
        docs = load_documents(data_dir, max_shards=args.max_shards)
        all_documents.extend(docs)

    dataset = BertPretrainDataset(
        documents=all_documents,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        mlm_probability=args.mlm_probability,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )

    # Scheduler
    total_steps = args.num_train_epochs * math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    warmup_steps = int(total_steps * args.warmup_ratio)
    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Prepare with Accelerate
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Training
    logger.info("***** Running pre-training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Per-device batch size = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {total_steps}")
    logger.info(f"  Warmup steps = {warmup_steps}")
    logger.info(f"  Num processes = {accelerator.num_processes}")

    global_step = 0
    running_loss = 0.0
    log_loss = 0.0
    start_time = time.time()

    for epoch in range(args.num_train_epochs):
        model.train()
        logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs}")

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    token_type_ids=batch["token_type_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    next_sentence_label=batch["next_sentence_label"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()

            if accelerator.sync_gradients:
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    elapsed = time.time() - start_time
                    avg_loss = (running_loss - log_loss) / args.logging_steps
                    current_lr = scheduler.get_last_lr()[0]

                    if accelerator.is_main_process:
                        logger.info(
                            f"Step {global_step}/{total_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Time: {elapsed:.1f}s"
                        )
                        accelerator.log(
                            {
                                "train/loss": avg_loss,
                                "train/learning_rate": current_lr,
                            },
                            step=global_step,
                        )

                    log_loss = running_loss
                    start_time = time.time()

                # Save checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        save_dir = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        os.makedirs(save_dir, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(save_dir)
                        tokenizer.save_pretrained(save_dir)
                        logger.info(f"Saved checkpoint to {save_dir}")

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        # Save at end of epoch
        if accelerator.is_main_process:
            save_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch + 1}")
            os.makedirs(save_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    # Final save
    if accelerator.is_main_process:
        save_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(save_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved final model to {save_dir}")

    accelerator.end_training()
    logger.info("Pre-training complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="BERT Pre-training with Accelerate")
    parser.add_argument("--data_dirs", nargs="+", required=True)
    parser.add_argument("--max_shards", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output/bert_pretrain")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
