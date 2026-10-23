import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

model_name = "answerdotai/ModernBERT-base"
data_path = "corpus_clean_cased2.txt"
output_dir = "./modern-bert-finetuned"
num_workers = 8

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )


if __name__ == "__main__":
    dataset = load_dataset("text", data_files=data_path)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=num_workers
    )

    model = AutoModelForMaskedLM.from_pretrained(model_name)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=15,
        per_device_train_batch_size=32,
        optim="adamw_torch_fused",
        fp16=False,
        bf16=True,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        logging_dir="./logs",
        report_to="tensorboard",
        tf32=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
    )

    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoints")
        trainer.train()


    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
