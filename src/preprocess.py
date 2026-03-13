"""
Data preprocessing: download and prepare Wikipedia dataset for BERT pre-training.
Produces document-level pickle files for efficient loading.
"""

import argparse
import os
import pickle
import re
from typing import List

from datasets import load_dataset
from tqdm import tqdm


def clean_text(text: str) -> str:
    """Clean Wikipedia article text."""
    # Remove references like [1], [2], etc.
    text = re.sub(r"\[\d+\]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str) -> List[str]:
    """Simple sentence splitter."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Filter out empty or too-short sentences
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def process_wikipedia(
    output_dir: str,
    max_articles: int = 0,
    shard_size: int = 100000,
    language: str = "en",
):
    """
    Download and process Wikipedia into document format.
    Each document = list of sentences from one article.
    Output: sharded pickle files.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Wikipedia ({language}) dataset from HuggingFace...")
    # Use wikimedia/wikipedia which is the new standard Parquet-based dataset
    use_streaming = max_articles > 0
    dataset = load_dataset(
        "wikimedia/wikipedia", f"20231101.{language}", split="train",
        streaming=use_streaming,
    )

    documents: List[List[str]] = []
    shard_idx = 0
    skipped = 0
    total_processed = 0

    total_hint = max_articles if max_articles > 0 else None
    print("Processing articles into documents...")
    for i, article in enumerate(tqdm(dataset, desc="Processing", total=total_hint)):
        if max_articles > 0 and total_processed >= max_articles:
            break

        text = clean_text(article["text"])
        sentences = split_into_sentences(text)

        # Skip articles with too few sentences (need at least 2 for NSP)
        if len(sentences) < 2:
            skipped += 1
            continue

        documents.append(sentences)
        total_processed += 1

        # Save shard
        if len(documents) >= shard_size:
            shard_path = os.path.join(output_dir, f"documents_{shard_idx:04d}.pkl")
            with open(shard_path, "wb") as f:
                pickle.dump(documents, f)
            print(f"Saved shard {shard_idx} with {len(documents)} documents")
            documents = []
            shard_idx += 1

    # Save remaining documents
    if documents:
        shard_path = os.path.join(output_dir, f"documents_{shard_idx:04d}.pkl")
        with open(shard_path, "wb") as f:
            pickle.dump(documents, f)
        print(f"Saved shard {shard_idx} with {len(documents)} documents")

    print(f"\nDone! Processed {total_processed} articles, skipped {skipped}")
    print(f"Output directory: {output_dir}")


def process_bookcorpus(output_dir: str, shard_size: int = 100000):
    """
    Download and process BookCorpus.
    Note: The HuggingFace bookcorpus may not always be available.
    Falls back gracefully if not accessible.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading BookCorpus dataset from HuggingFace...")
    try:
        dataset = load_dataset("bookcorpus", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"BookCorpus not available: {e}")
        print("This is expected — BookCorpus has access restrictions.")
        print("The model can be pre-trained with Wikipedia alone.")
        return

    # BookCorpus is line-based; group lines into "documents" (books)
    # Each line is roughly a sentence
    documents: List[List[str]] = []
    current_doc: List[str] = []
    shard_idx = 0

    print("Processing BookCorpus...")
    for item in tqdm(dataset, desc="Processing"):
        line = item["text"].strip()
        if not line:
            # Empty line = document boundary
            if len(current_doc) >= 2:
                documents.append(current_doc)
            current_doc = []
        else:
            if len(line) > 10:
                current_doc.append(line)

        if len(documents) >= shard_size:
            shard_path = os.path.join(output_dir, f"documents_{shard_idx:04d}.pkl")
            with open(shard_path, "wb") as f:
                pickle.dump(documents, f)
            print(f"Saved shard {shard_idx} with {len(documents)} documents")
            documents = []
            shard_idx += 1

    # Don't forget the last document
    if len(current_doc) >= 2:
        documents.append(current_doc)

    if documents:
        shard_path = os.path.join(output_dir, f"documents_{shard_idx:04d}.pkl")
        with open(shard_path, "wb") as f:
            pickle.dump(documents, f)
        print(f"Saved shard {shard_idx} with {len(documents)} documents")


def main():
    parser = argparse.ArgumentParser(description="Preprocess data for BERT pre-training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed documents",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wikipedia", "bookcorpus", "all"],
        default="wikipedia",
        help="Which dataset to process",
    )
    parser.add_argument(
        "--max_articles",
        type=int,
        default=0,
        help="Max articles to process (0 = all)",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=100000,
        help="Number of documents per shard file",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Wikipedia language edition",
    )
    args = parser.parse_args()

    if args.dataset in ("wikipedia", "all"):
        wiki_dir = os.path.join(args.output_dir, "wikipedia")
        process_wikipedia(
            wiki_dir,
            max_articles=args.max_articles,
            shard_size=args.shard_size,
            language=args.language,
        )

    if args.dataset in ("bookcorpus", "all"):
        book_dir = os.path.join(args.output_dir, "bookcorpus")
        process_bookcorpus(book_dir, shard_size=args.shard_size)


if __name__ == "__main__":
    main()
