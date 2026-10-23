import os
import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss

def main():
    FILE_PATH = "corpus_clean_uncased.txt"

    raw_dataset = load_dataset("text", data_files=FILE_PATH, split="train")

    train_dataset = raw_dataset.map(
        lambda x: {"sentence1": x["text"], "sentence2": x["text"]},
        batched=True,
        remove_columns=["text"]
    )

    model = SentenceTransformer("google/embeddinggemma-300m")
    model.max_seq_length = 128

    loss = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir="embeddinggemma-ancient-greek15iter",
        num_train_epochs=25,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=3e-5,
        bf16=True,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=2,
        dataloader_num_workers=0,
        optim="adamw_torch_fused",
        report_to="tensorboard",
        logging_dir="embeddinggemma-ancient-greek15iter/runs",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )

    trainer.train(resume_from_checkpoint=True)
    model.save_pretrained("embeddinggemma-300m-ancient-greek-final15iter")

if __name__ == "__main__":
    main()
