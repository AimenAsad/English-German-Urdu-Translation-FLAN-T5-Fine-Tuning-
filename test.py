import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------
# Load Model & Tokenizer
# ------------------------
MODEL_PATH = "./models/translator"   # adjust if your model is saved somewhere else
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ------------------------
# Test Sentences (EN → DE)
# ------------------------
test_sentences = [
    "Good morning!",
    "How old are you?",
    "I am from Pakistan.",
    "Can you help me?",
    "This is my friend.",
    "The book is on the table.",
    "We are going to the university.",
    "I like to play football.",
    "She is drinking water.",
    "Tomorrow will be a sunny day.",
    "Where are you going?",
    "I love learning new languages.",
    "My father is a doctor.",
    "The train arrives at five o’clock.",
    "Please open the window.",
    "Do you understand English?",
    "He is watching television.",
    "They are playing in the park.",
    "I would like a cup of tea.",
    "The restaurant is very good."
]

# ------------------------
# Translation Function
# ------------------------
def translate(sentence):
    input_text = f"translate English to German: {sentence}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------------
# Run Tests
# ------------------------
print("🔹 Testing English → German Translation\n")
for en in test_sentences:
    de = translate(en)
    print(f"EN: {en}")
    print(f"DE: {de}\n")
