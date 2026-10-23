import os
import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import TripletDistanceMetric, TripletLoss


def main():
    FILE_PATH = "greek_triplets.json"

    raw_dataset = load_dataset("json", data_files=FILE_PATH, split="train")

    train_dataset = raw_dataset.remove_columns(["type", "meta"])

    model = SentenceTransformer("google/embeddinggemma-300m")
    model.max_seq_length = 128

    loss = TripletLoss(model=model, distance_metric=TripletDistanceMetric.COSINE, triplet_margin=0.3)

    training_args = SentenceTransformerTrainingArguments(
        output_dir="embeddinggemma-300m-ancient-greek-final10iter-tripleloss",
        num_train_epochs=10,
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
        logging_dir="embeddinggemma-300m-ancient-greek-final10iter-tripleloss/runs",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )

    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained("embeddinggemma-300m-ancient-greek-final10iter-tripleloss")

if __name__ == "__main__":
    main()
