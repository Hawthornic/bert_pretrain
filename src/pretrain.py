"""
BERT Pre-training: Main training loop with MLM + NSP.
Supports single-GPU and multi-GPU (via Accelerate) training.
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
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def create_dataloader(
    documents,
    tokenizer,
    max_seq_length: int,
    batch_size: int,
    mlm_probability: float,
    num_workers: int = 4,
):
    """Create DataLoader from documents."""
    dataset = BertPretrainDataset(
        documents=documents,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        mlm_probability=mlm_probability,
    )
    logger.info(f"Created dataset with {len(dataset)} instances")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


def train(args):
    """Main training function."""
    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"Device: {device}, n_gpu: {n_gpu}")

    os.makedirs(args.output_dir, exist_ok=True)

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)

    # Load model config and create model
    if args.config_file:
        logger.info(f"Loading config from: {args.config_file}")
        with open(args.config_file, "r") as f:
            config_dict = json.load(f)
        config = BertConfig(**config_dict)
    else:
        logger.info("Using default bert-base-uncased config")
        config = BertConfig()

    # Ensure config matches tokenizer
    config.vocab_size = tokenizer.vocab_size

    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        model = BertForPreTraining.from_pretrained(args.resume_from)
    else:
        logger.info("Creating new BERT model from scratch")
        model = BertForPreTraining(config)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    model.to(device)

    # Multi-GPU
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Load data
    all_documents = []
    for data_dir in args.data_dirs:
        docs = load_documents(data_dir, max_shards=args.max_shards)
        all_documents.extend(docs)
    logger.info(f"Total documents: {len(all_documents)}")

    # Create dataloader
    dataloader = create_dataloader(
        documents=all_documents,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        batch_size=args.train_batch_size,
        mlm_probability=args.mlm_probability,
        num_workers=args.num_workers,
    )

    # Calculate total training steps
    total_steps = args.num_train_epochs * len(dataloader)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
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

    # LR Scheduler with warmup
    warmup_steps = int(total_steps * args.warmup_ratio)
    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Load optimizer/scheduler state if resuming
    if args.resume_from:
        opt_path = os.path.join(args.resume_from, "optimizer.pt")
        sched_path = os.path.join(args.resume_from, "scheduler.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            logger.info("Loaded optimizer state")
        if os.path.exists(sched_path):
            scheduler.load_state_dict(torch.load(sched_path, map_location=device))
            logger.info("Loaded scheduler state")

    # Mixed precision
    scaler = None
    if args.fp16:
        scaler = torch.amp.GradScaler("cuda")
        logger.info("Using FP16 mixed precision training")

    # Training loop
    logger.info("***** Running pre-training *****")
    logger.info(f"  Num examples = {len(dataloader.dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    effective_batch = args.train_batch_size * args.gradient_accumulation_steps * max(1, n_gpu)
    logger.info(f"  Effective batch size = {effective_batch}")
    logger.info(f"  Total optimization steps = {total_steps}")
    logger.info(f"  Warmup steps = {warmup_steps}")

    global_step = 0
    total_loss = 0.0
    total_mlm_loss = 0.0
    total_nsp_loss = 0.0
    logging_loss = 0.0
    model.zero_grad()
    start_time = time.time()

    for epoch in range(args.num_train_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        model.train()

        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            if args.fp16:
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        token_type_ids=batch["token_type_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        next_sentence_label=batch["next_sentence_label"],
                    )
                    loss = outputs.loss
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    token_type_ids=batch["token_type_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    next_sentence_label=batch["next_sentence_label"],
                )
                loss = outputs.loss

            if n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )

                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                model.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    elapsed = time.time() - start_time
                    avg_loss = (total_loss - logging_loss) / args.logging_steps
                    current_lr = scheduler.get_last_lr()[0]
                    steps_per_sec = args.logging_steps / elapsed

                    logger.info(
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Steps/s: {steps_per_sec:.2f}"
                    )

                    tb_writer.add_scalar("train/loss", avg_loss, global_step)
                    tb_writer.add_scalar("train/learning_rate", current_lr, global_step)
                    tb_writer.add_scalar("train/steps_per_second", steps_per_sec, global_step)

                    logging_loss = total_loss
                    start_time = time.time()

                # Save checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, args, global_step)

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        # Save at end of epoch
        save_checkpoint(model, optimizer, scheduler, args, global_step, epoch=epoch)

    # Save final model
    save_checkpoint(model, optimizer, scheduler, args, global_step, final=True)
    tb_writer.close()
    logger.info("Pre-training complete!")


def save_checkpoint(model, optimizer, scheduler, args, global_step, epoch=None, final=False):
    """Save model checkpoint."""
    if final:
        checkpoint_dir = os.path.join(args.output_dir, "final_model")
    elif epoch is not None:
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch + 1}")
    else:
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(checkpoint_dir)

    # Save optimizer and scheduler
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

    # Save training args
    with open(os.path.join(checkpoint_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f"Saved checkpoint to {checkpoint_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="BERT Pre-training")

    # Data
    parser.add_argument(
        "--data_dirs",
        nargs="+",
        required=True,
        help="Directories containing preprocessed document shards",
    )
    parser.add_argument("--max_shards", type=int, default=0, help="Max shards to load (0=all)")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="MLM masking probability")

    # Model
    parser.add_argument("--config_file", type=str, default=None, help="Model config JSON file")
    parser.add_argument(
        "--tokenizer_name", type=str, default="bert-base-uncased", help="Tokenizer name or path"
    )
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")

    # Training
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size per GPU")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1=use epochs)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps (overrides ratio if > 0)")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")

    # Logging & saving
    parser.add_argument("--output_dir", type=str, default="output/bert_pretrain", help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every N steps")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
