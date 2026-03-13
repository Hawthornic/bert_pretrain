"""
BERT Demo Application — Gradio Web UI
Features:
  1. Fill-Mask: predict masked words in a sentence
  2. Semantic Similarity: compare two sentences using BERT embeddings
  3. Keyword Extraction: find the most "surprising" words in a text
"""

import argparse
import json
import os

import gradio as gr
import numpy as np
import torch
from transformers import BertConfig, BertForPreTraining, BertTokenizerFast, BertModel


class BertDemo:
    def __init__(self, model_dir: str, device: str = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Loading model from {model_dir} on {self.device}...")

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        # Load config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        config = BertConfig(**config_dict)

        # Load the pre-trained model
        self.model = BertForPreTraining.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Also create a BertModel (encoder only) for embeddings
        self.encoder = self.model.bert

        print("Model loaded successfully!")

    @torch.no_grad()
    def fill_mask(self, text: str, top_k: int = 5) -> str:
        """Predict masked tokens in text. Use [MASK] as placeholder."""
        if "[MASK]" not in text:
            return "Please include [MASK] in your sentence.\nExample: The capital of France is [MASK]."

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                                max_length=128).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.prediction_logits  # (1, seq_len, vocab_size)

        # Find [MASK] positions
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[1]

        results = []
        for i, pos in enumerate(mask_positions):
            token_logits = logits[0, pos]
            probs = torch.softmax(token_logits, dim=-1)
            top_probs, top_indices = probs.topk(top_k)

            results.append(f"[MASK] position {i + 1}:")
            for rank, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                token = self.tokenizer.decode([idx.item()]).strip()
                filled = text.replace("[MASK]", f"**{token}**", 1)
                results.append(f"  {rank + 1}. '{token}' (probability: {prob.item():.4f})")

            # Show the top-1 filled sentence
            best_token = self.tokenizer.decode([top_indices[0].item()]).strip()
            results.append(f"  → Best: {text.replace('[MASK]', best_token, 1)}")
            results.append("")

        return "\n".join(results)

    @torch.no_grad()
    def get_embedding(self, text: str) -> np.ndarray:
        """Get sentence embedding using [CLS] token."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                                max_length=128).to(self.device)
        outputs = self.encoder(**inputs)
        cls_emb = outputs.last_hidden_state[0, 0]  # [CLS] token
        return cls_emb.cpu().numpy()

    def semantic_similarity(self, text1: str, text2: str) -> str:
        """Compute cosine similarity between two sentences."""
        if not text1.strip() or not text2.strip():
            return "Please enter two sentences."

        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)

        # Interpret
        if cos_sim > 0.8:
            level = "Very Similar"
        elif cos_sim > 0.5:
            level = "Somewhat Similar"
        elif cos_sim > 0.2:
            level = "Slightly Related"
        else:
            level = "Not Similar"

        return (
            f"Cosine Similarity: {cos_sim:.4f}\n"
            f"Interpretation: {level}\n\n"
            f"Sentence A: {text1}\n"
            f"Sentence B: {text2}"
        )

    @torch.no_grad()
    def keyword_extraction(self, text: str, top_k: int = 8) -> str:
        """
        Extract keywords by masking each word and measuring prediction difficulty.
        Words that are hardest to predict = most informative = keywords.
        """
        if not text.strip():
            return "Please enter some text."

        tokens = self.tokenizer.tokenize(text)
        if len(tokens) < 3:
            return "Please enter a longer text (at least a few words)."

        word_scores = []

        for i, token in enumerate(tokens):
            if token.startswith("##"):
                continue

            # Create masked version
            masked_tokens = tokens.copy()
            masked_tokens[i] = "[MASK]"
            masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)

            inputs = self.tokenizer(masked_text, return_tensors="pt", padding=True,
                                    truncation=True, max_length=128).to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.prediction_logits

            mask_positions = (inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            if len(mask_positions) == 0:
                continue

            pos = mask_positions[0]
            probs = torch.softmax(logits[0, pos], dim=-1)
            original_id = self.tokenizer.convert_tokens_to_ids(token)
            original_prob = probs[original_id].item()

            # Lower probability = more surprising = more important
            surprise = -np.log(original_prob + 1e-10)
            clean_token = token.replace("##", "")
            if len(clean_token) > 1:
                word_scores.append((clean_token, surprise, original_prob))

        if not word_scores:
            return "Could not extract keywords from this text."

        word_scores.sort(key=lambda x: x[1], reverse=True)
        top_words = word_scores[:top_k]

        results = ["Keywords (ranked by importance):\n"]
        for rank, (word, surprise, prob) in enumerate(top_words):
            bar = "█" * int(surprise * 2) + "░" * max(0, 20 - int(surprise * 2))
            results.append(f"  {rank + 1}. {word:<15} surprise: {surprise:.2f}  [{bar}]")

        return "\n".join(results)


def create_ui(demo: BertDemo, share: bool = False):
    with gr.Blocks(title="BERT Pre-trained Model Demo", theme=gr.themes.Soft()) as app:
        gr.Markdown("# BERT Pre-trained Model Demo")
        gr.Markdown("Explore what your BERT model learned during pre-training.")

        with gr.Tab("Fill Mask"):
            gr.Markdown("Enter a sentence with `[MASK]` to see what the model predicts.")
            mask_input = gr.Textbox(
                label="Input",
                placeholder="The capital of France is [MASK].",
                lines=2,
            )
            mask_output = gr.Textbox(label="Predictions", lines=10)
            mask_btn = gr.Button("Predict", variant="primary")
            mask_btn.click(demo.fill_mask, inputs=mask_input, outputs=mask_output)

            gr.Examples(
                examples=[
                    "The [MASK] is the largest planet in the solar system.",
                    "She went to the [MASK] to buy some groceries.",
                    "Python is a popular [MASK] language.",
                    "The patient was taken to the [MASK] after the accident.",
                ],
                inputs=mask_input,
            )

        with gr.Tab("Semantic Similarity"):
            gr.Markdown("Compare two sentences to see how similar they are.")
            sim_text1 = gr.Textbox(label="Sentence A", placeholder="The cat sat on the mat.")
            sim_text2 = gr.Textbox(label="Sentence B", placeholder="A dog was lying on the rug.")
            sim_output = gr.Textbox(label="Result", lines=5)
            sim_btn = gr.Button("Compare", variant="primary")
            sim_btn.click(demo.semantic_similarity, inputs=[sim_text1, sim_text2], outputs=sim_output)

            gr.Examples(
                examples=[
                    ["The weather is nice today.", "It is a beautiful day outside."],
                    ["I love programming in Python.", "The snake slithered through the grass."],
                    ["Machine learning is a subset of AI.", "Deep learning uses neural networks."],
                ],
                inputs=[sim_text1, sim_text2],
            )

        with gr.Tab("Keyword Extraction"):
            gr.Markdown("Extract keywords from text using the model's understanding.")
            kw_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter a paragraph of text...",
                lines=4,
            )
            kw_output = gr.Textbox(label="Keywords", lines=12)
            kw_btn = gr.Button("Extract", variant="primary")
            kw_btn.click(demo.keyword_extraction, inputs=kw_input, outputs=kw_output)

    app.launch(server_name="0.0.0.0", server_port=7860, share=share)


def main():
    parser = argparse.ArgumentParser(description="BERT Demo App")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to pre-trained model checkpoint")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    demo = BertDemo(args.model_dir, device=args.device)
    create_ui(demo, share=args.share)


if __name__ == "__main__":
    main()
