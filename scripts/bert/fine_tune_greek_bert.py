import torch
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import os
from transformers.trainer_utils import get_last_checkpoint

model_name = "princeton-logion/logion-bert-base"
data_path = "../pg_corpus_lemmatized_clean.txt"

num_workers = 8

tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

if __name__ == "__main__":


    dataset = load_dataset("text", data_files=data_path)
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=num_workers
    )

    # checkpoint = get_last_checkpoint("./bert-logion-finetuned")
    #
    # if checkpoint:
    #     model = BertForMaskedLM.from_pretrained(checkpoint)
    #     print("Checkpoint loaded")
    # else:
    #     model = BertForMaskedLM.from_pretrained(model_name)
    #     data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    # )

    model = BertForMaskedLM.from_pretrained(model_name)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./bert-logion-finetuned",
        num_train_epochs=200,
        per_device_train_batch_size=128,
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

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model("./bert-logion-finetunde")
