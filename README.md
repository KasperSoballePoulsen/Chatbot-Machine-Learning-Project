# 🫀 Chatbot Machine Learning Project

Et simpelt Python-chatbotprojekt, der bruger machine learning og NLP til at forudsige risikoen for hjertekarsygdom baseret på brugerinput.  
Chatbotten kombinerer klassifikation med en samtaleoplevelse og sentimentanalyse for at vurdere brugerens tone og symptomer.

## 📊 Funktionalitet

- Interaktiv chatbot i terminalen
- Forudsigelse af hjertesygdom med Random Forest-model
- NLP-analyse af symptomer med HuggingFace Transformers (`distilbert-base-cased-distilled-squad`)
- Sentimentanalyse for hver brugerbesked
- Automatisk samtaleopsummering med:
  - Positiv/negativ stemning
  - Hyppigst brugte ord

## 🧠 Brugte teknologier

- `pandas`, `scikit-learn` – dataforberedelse og RandomForest-modellen
- `transformers` (Hugging Face) – spørgsmål/svar og sentimentanalyse
- `spaCy` – lemmatisering og keyword-analyse
- `heart.csv` – datasæt over hjertesygdomme

## 📁 Struktur

- `chatbot.py` – hele projektets kode
- `heart.csv` – datasæt med patientoplysninger
- `.gitignore` – ignorerer fx cache, venv osv.

## ▶️ Sådan kører du det

1. Sørg for at have installeret følgende pakker:

```bash
pip install pandas scikit-learn transformers spacy
python -m spacy download en_core_web_sm
```

2. Kør programmet:
```bash
python chatbot.py
```

3. Følg instruktionerne fra chatbotten i terminalen. Skriv exit for at afslutte.
