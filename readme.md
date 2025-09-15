## English ↔ German / Urdu Translation (FLAN-T5 Fine-Tuning)

This project focuses on fine-tuning the FLAN-T5 model for English → German and English → Urdu translation tasks.
The work demonstrates how transformer-based models can be adapted for low-resource setups while highlighting challenges and improvements needed.

### What I Did

Fine-tuned FLAN-T5 on:
English → German (small dataset).
English → Urdu using OPUS-100 dataset.

Created testing and evaluation scripts to check translations.

Used standard translation evaluation metrics:
BLEU (measures word overlap).
ROUGE (measures recall & coverage of meaning).

ChrF (character-level F-score, helpful for morphologically rich languages like German/Urdu).

### Results

The results are relatively low because:

No access to GPU (training was done on CPU).
Used small training datasets.
With larger datasets and GPU training, translation accuracy and grammar correctness can improve significantly.

### Future Improvements

Train on larger parallel datasets (e.g., OPUS-100 full, WMT).
Use GPU / TPU to speed up fine-tuning and improve results.
Explore transformer variants (MarianMT, M2M-100, NLLB).
Deploy as an API (Flask/FastAPI) or a Streamlit app for real-time translation.

### Project Structure
translation-project<br>
│── fine_tuning.py              # Script to fine-tune both English-German & English-Urdu<br>
│── test.py                    # Script to test translation with example sentences<br> 
│── evaluate.py                 # Script to evaluate results using BLEU, ROUGE, ChrF<br> 
│── requirements.txt            # Dependencies<br>
│── README.md<br> 

### Why BLEU, ROUGE, and ChrF?

BLEU: Good for measuring word-to-word overlap in translations.
ROUGE: Captures meaning recall (important for checking if the core message is translated).
ChrF: Works well for morphologically rich languages like German and Urdu, where word forms change often.
