# evaluate.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "./models/translator"
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

# Load small sample
ds = load_dataset("wmt16", "de-en")["validation"].select(range(100))

inputs = ["translate English to German: " + ex["translation"]["en"] for ex in ds]
refs = [ex["translation"]["de"] for ex in ds]

# Batch evaluation for speed + stability
batch_size = 8
preds = []

for i in range(0, len(inputs), batch_size):
    enc = tokenizer(inputs[i:i+batch_size], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    outputs = model.generate(**enc, max_length=128)
    batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    preds.extend(batch_preds)

# Metrics
bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])
rouge_score = rouge.compute(predictions=preds, references=refs)

print("BLEU:", bleu_score["score"])
print("ROUGE-L:", rouge_score["rougeL"])

chrf = evaluate.load("chrf")
chrf_score = chrf.compute(predictions=preds, references=refs)
print("ChrF:", chrf_score["score"])
