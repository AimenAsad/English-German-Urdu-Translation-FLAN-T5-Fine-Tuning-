 # fine-tuning script
from datasets import load_dataset
from datasets import concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import torch

MODEL_NAME = "google/flan-t5-base"  # "google/mt5-small" for stronger multilingual

de_ds = load_dataset("wmt16", "de-en")   # WMT16 has English–German
ur_ds = load_dataset("opus100", "en-ur") # OPUS has English–Urdu (for simplicity, use "opus100")

train_de = de_ds["train"].shuffle(seed=42).select(range(5000))
val_de   = de_ds["validation"].shuffle(seed=42).select(range(500))
train_ur = ur_ds["train"].shuffle(seed=42).select(range(5000))
val_ur   = ur_ds["validation"].shuffle(seed=42).select(range(500))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

max_len = 128

def preprocess_de(batch):
    inputs = ["translate English to German: " + ex["en"] for ex in batch["translation"]]
    targets = [ex["de"] for ex in batch["translation"]]

    model_inputs = tokenizer(
        inputs, max_length=128, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        targets, max_length=128, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_ur(batch):
    inputs = ["translate English to Urdu: " + ex["en"] for ex in batch["translation"]]
    targets = [ex["ur"] for ex in batch["translation"]]

    model_inputs = tokenizer(
        inputs, max_length=128, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        targets, max_length=128, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_de = train_de.map(preprocess_de, batched=True, remove_columns=train_de.column_names)
val_de   = val_de.map(preprocess_de, batched=True, remove_columns=val_de.column_names)
train_ur = train_ur.map(preprocess_ur, batched=True, remove_columns=train_ur.column_names)
val_ur   = val_ur.map(preprocess_ur, batched=True, remove_columns=val_ur.column_names)

# Merge datasets
train_ds = concatenate_datasets([train_de, train_ur])
val_ds   = concatenate_datasets([val_de, val_ur])

# 2. Load model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 3. Training
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/translator",
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./models/translator")
