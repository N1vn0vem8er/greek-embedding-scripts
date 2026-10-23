import os
from pathlib import Path
import torch
from datasets import Dataset
from tokenizers import BertWordPieceTokenizer
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    corpus_file = Path("../pg_corpus_lemmatized_clean.txt")

    tokenizer = BertWordPieceTokenizer(clean_text=True, lowercase=True)

    tokenizer.train(
        files=[str(corpus_file)],
        vocab_size=30_000,
        min_frequency=1,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    tokenizer_dir = data_dir / "greek_bert_tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    tokenizer.save_model(str(tokenizer_dir))

    hf_tokenizer = BertTokenizerFast.from_pretrained(
        str(tokenizer_dir), do_lower_case=True
    )

    config = BertConfig(
        vocab_size=hf_tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=512,
        type_vocab_size=2,
    )

    model = BertForMaskedLM(config=config)
    print(f"Params: {model.num_parameters():,}")

    with open(corpus_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    raw_dataset = Dataset.from_dict({"text": lines})

    def tokenize_function(examples):
        return hf_tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding=False,
        )

    tokenized_dataset = raw_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=hf_tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./greek_bert_checkpoints",
        num_train_epochs=400,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=8,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    trainer.train(resume_from_checkpoint=True)

    final_model_dir = "./greek_bert_final_v2"
    trainer.save_model(final_model_dir)
    hf_tokenizer.save_pretrained(final_model_dir)


if __name__ == "__main__":
    main()
