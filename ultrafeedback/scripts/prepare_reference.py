from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from spo.utils.logging import setup_logging
from spo.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default="openbmb/UltraFeedback")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache_dir", default="data/cache")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", default="outputs/ref_sft")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    logger = setup_logging("spo.prepare_reference")
    seed_everything(args.seed)

    dataset = load_dataset(args.dataset_id, split=args.split, cache_dir=args.cache_dir)
    dataset = dataset.shuffle(seed=args.seed).select(range(min(args.num_samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def pick_response(ex):
        completions = ex.get("completions") or []
        if not completions:
            return None
        # Choose highest overall_score if present, else fine-grained score.
        def score(c):
            return c.get("overall_score", c.get("fine-grained_score", 0.0))
        best = max(completions, key=score)
        return best.get("response")

    def preprocess(ex):
        prompt = ex.get("instruction") or ex.get("prompt") or ex.get("query") or ex.get("question")
        response = pick_response(ex) or ex.get("response") or ex.get("output") or ex.get("answer") or ex.get("chosen")
        if not prompt or not response:
            return {"input_ids": None, "attention_mask": None}
        text = prompt + response
        tokenized = tokenizer(text, truncation=True, max_length=args.max_length)
        return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}

    tokenized = dataset.map(preprocess, remove_columns=dataset.column_names)
    tokenized = tokenized.filter(lambda x: x["input_ids"] is not None)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        save_total_limit=1,
        logging_steps=10,
        save_steps=200,
        bf16=True,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Saved SFT reference to %s", args.output_dir)


if __name__ == "__main__":
    main()
