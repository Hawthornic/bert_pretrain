"""
BERT Pre-training Dataset: MLM + NSP
Follows the original BERT paper (Devlin et al., 2019).
"""

import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


class BertPretrainDataset(Dataset):
    """
    Dataset for BERT pre-training with Masked Language Modeling (MLM)
    and Next Sentence Prediction (NSP).

    Each sample consists of two segments (sentence A and sentence B).
    - 50% of the time, sentence B is the actual next sentence (label=0).
    - 50% of the time, sentence B is a random sentence (label=1).

    Input documents should be a list of documents, where each document
    is a list of sentences (strings).
    """

    def __init__(
        self,
        documents: List[List[str]],
        tokenizer: PreTrainedTokenizerFast,
        max_seq_length: int = 512,
        mlm_probability: float = 0.15,
        short_seq_prob: float = 0.1,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.short_seq_prob = short_seq_prob
        self.documents = documents

        # Build instances from documents
        self.instances = self._create_instances()

    def _create_instances(self) -> List[Dict]:
        """Create training instances from documents."""
        instances = []
        for doc_idx, document in enumerate(
            tqdm(self.documents, desc="Creating training instances")
        ):
            instances.extend(
                self._create_instances_from_document(doc_idx, document)
            )
        random.shuffle(instances)
        return instances

    def _create_instances_from_document(
        self, doc_idx: int, document: List[str]
    ) -> List[Dict]:
        """Create instances from a single document following BERT paper."""
        instances = []
        # Account for [CLS], [SEP], [SEP]
        max_num_tokens = self.max_seq_length - 3

        # Sometimes use shorter sequences to minimize mismatch
        # between pre-training and fine-tuning
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)

        current_chunk: List[str] = []
        current_length = 0
        i = 0

        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            tokens = self.tokenizer.tokenize(segment)
            current_length += len(tokens)

            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk and len(current_chunk) >= 2:
                    # Pick a split point for sentence A / B
                    a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(self.tokenizer.tokenize(current_chunk[j]))

                    tokens_b = []
                    is_random_next = False

                    # 50% chance of random next sentence
                    if len(self.documents) > 1 and random.random() < 0.5:
                        is_random_next = True
                        # Pick a random document
                        for _ in range(10):
                            random_doc_idx = random.randint(0, len(self.documents) - 1)
                            if random_doc_idx != doc_idx:
                                break
                        random_document = self.documents[random_doc_idx]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(
                                self.tokenizer.tokenize(random_document[j])
                            )
                            if len(tokens_b) >= target_seq_length // 2:
                                break
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(self.tokenizer.tokenize(current_chunk[j]))

                    # Truncate to max length
                    self._truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    if tokens_a and tokens_b:
                        instances.append(
                            {
                                "tokens_a": tokens_a,
                                "tokens_b": tokens_b,
                                "is_random_next": is_random_next,
                            }
                        )

                current_chunk = []
                current_length = 0

            i += 1

        return instances

    @staticmethod
    def _truncate_seq_pair(
        tokens_a: List[str], tokens_b: List[str], max_num_tokens: int
    ):
        """Truncate a pair of sequences to a maximum total length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _apply_mlm(
        self, token_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Apply Masked Language Modeling following the BERT paper:
        - 15% of tokens are selected for prediction
        - Of those: 80% -> [MASK], 10% -> random token, 10% -> unchanged
        """
        labels = [-100] * len(token_ids)
        output_ids = list(token_ids)

        # Identify candidate positions (skip [CLS], [SEP], [PAD])
        special_token_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }
        candidate_indices = [
            i for i, tid in enumerate(token_ids) if tid not in special_token_ids
        ]

        random.shuffle(candidate_indices)
        num_to_mask = max(1, int(len(candidate_indices) * self.mlm_probability))
        masked_indices = sorted(candidate_indices[:num_to_mask])

        for idx in masked_indices:
            labels[idx] = token_ids[idx]
            rand_val = random.random()
            if rand_val < 0.8:
                output_ids[idx] = self.tokenizer.mask_token_id
            elif rand_val < 0.9:
                output_ids[idx] = random.randint(0, self.tokenizer.vocab_size - 1)
            # else: keep original (10%)

        return output_ids, labels

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        instance = self.instances[idx]
        tokens_a = instance["tokens_a"]
        tokens_b = instance["tokens_b"]
        is_random_next = instance["is_random_next"]

        # Convert tokens to ids
        ids_a = self.tokenizer.convert_tokens_to_ids(tokens_a)
        ids_b = self.tokenizer.convert_tokens_to_ids(tokens_b)

        # Build input: [CLS] tokens_a [SEP] tokens_b [SEP]
        input_ids = (
            [self.tokenizer.cls_token_id]
            + ids_a
            + [self.tokenizer.sep_token_id]
            + ids_b
            + [self.tokenizer.sep_token_id]
        )

        # Token type ids: 0 for segment A, 1 for segment B
        token_type_ids = (
            [0] * (len(ids_a) + 2)  # [CLS] + tokens_a + [SEP]
            + [1] * (len(ids_b) + 1)  # tokens_b + [SEP]
        )

        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)

        # Pad to max_seq_length
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        token_type_ids += [0] * padding_length
        attention_mask += [0] * padding_length

        # Apply MLM
        input_ids, mlm_labels = self._apply_mlm(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(mlm_labels, dtype=torch.long),
            "next_sentence_label": torch.tensor(
                1 if is_random_next else 0, dtype=torch.long
            ),
        }
