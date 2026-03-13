"""
Evaluate a pre-trained BERT model on MLM perplexity and NSP accuracy.
"""

import argparse
import glob
import json
import logging
import os
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForPreTraining, BertTokenizerFast

from dataset import BertPretrainDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    model = BertForPreTraining.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizerFast.from_pretrained(
        args.model_path if os.path.exists(os.path.join(args.model_path, "tokenizer_config.json"))
        else args.tokenizer_name
    )

    # Load eval data
    shard_files = sorted(glob.glob(os.path.join(args.data_dir, "documents_*.pkl")))
    if not shard_files:
        raise FileNotFoundError(f"No data found in {args.data_dir}")

    # Use only first shard for evaluation (or all if small)
    shard_files = shard_files[:max(1, args.max_shards)]
    documents = []
    for sf in shard_files:
        with open(sf, "rb") as f:
            documents.extend(pickle.load(f))
    # Use a subset for evaluation
    if args.max_docs > 0:
        documents = documents[:args.max_docs]

    logger.info(f"Evaluating on {len(documents)} documents")

    dataset = BertPretrainDataset(
        documents=documents,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        mlm_probability=0.15,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    total_mlm_loss = 0.0
    total_nsp_loss = 0.0
    total_nsp_correct = 0
    total_nsp_count = 0
    total_mlm_tokens = 0
    num_batches = 0

    mlm_loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    nsp_loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                token_type_ids=batch["token_type_ids"],
                attention_mask=batch["attention_mask"],
            )

            # MLM loss
            prediction_scores = outputs.prediction_logits
            mlm_labels = batch["labels"]
            mlm_mask = mlm_labels != -100
            if mlm_mask.sum() > 0:
                mlm_loss = mlm_loss_fn(
                    prediction_scores[mlm_mask],
                    mlm_labels[mlm_mask],
                )
                total_mlm_loss += mlm_loss.item()
                total_mlm_tokens += mlm_mask.sum().item()

            # NSP accuracy
            seq_scores = outputs.seq_relationship_logits
            nsp_labels = batch["next_sentence_label"]
            nsp_loss = nsp_loss_fn(seq_scores, nsp_labels)
            total_nsp_loss += nsp_loss.item()

            nsp_preds = seq_scores.argmax(dim=-1)
            total_nsp_correct += (nsp_preds == nsp_labels).sum().item()
            total_nsp_count += nsp_labels.size(0)
            num_batches += 1

    # Compute metrics
    avg_mlm_loss = total_mlm_loss / max(total_mlm_tokens, 1)
    mlm_perplexity = torch.exp(torch.tensor(avg_mlm_loss)).item()
    avg_nsp_loss = total_nsp_loss / max(total_nsp_count, 1)
    nsp_accuracy = total_nsp_correct / max(total_nsp_count, 1)

    results = {
        "mlm_loss": avg_mlm_loss,
        "mlm_perplexity": mlm_perplexity,
        "nsp_loss": avg_nsp_loss,
        "nsp_accuracy": nsp_accuracy,
        "num_eval_examples": total_nsp_count,
        "num_mlm_tokens": total_mlm_tokens,
    }

    logger.info("***** Evaluation Results *****")
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key} = {value:.4f}")
        else:
            logger.info(f"  {key} = {value}")

    # Save results
    output_file = os.path.join(args.model_path, "eval_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pre-trained BERT")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to eval data directory")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_shards", type=int, default=1)
    parser.add_argument("--max_docs", type=int, default=10000)
    args = parser.parse_args()
    evaluate(args)
