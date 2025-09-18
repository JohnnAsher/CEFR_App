# 🌍 CEFR Level Classifier (Streamlit App)
This is a Streamlit web application that predicts the CEFR (A1–C2) level of an English essay using:
- A fine-tuned DistilBERT model (`models/distilbert_clean/`)
- A hybrid logistic regression model with lexical features
- A CEFR wordlist for lexical profiling

## Project Structure
CEFR_App/
│   app.py                  # Main Streamlit application (UI + wiring)
│   requirements.txt        # Python dependencies
│   README.md               # Instructions
│
├── models/
│   distilbert_clean/       # Fine-tuned DistilBERT model + tokenizer
│   hybrid_lr.pkl           # Hybrid logistic regression model
│   hybrid_scaler.pkl       # Scaler for hybrid features
│   temperature_global.pt   # Global temperature calibration
│   temperature_vector.pt   # Vector temperature calibration
│
├── data/
│   cefr_wordlist_clean.pkl # CEFR wordlist for lexical profiling
│   feature_stats.json      # Feature averages by CEFR level
│   prompts.csv             # Practice prompts
│   predictions_log.csv     # Created automatically if missing (user logs)
│
├── helpers/
│   __init__.py             # Marks helpers as a package
│   config.py               # Central paths + LEVELS constant
│   features.py             # Tokenization + lexical feature extraction
│   predictions.py          # Hybrid model scoring + calibration prediction
│   feedback.py             # CEFR interpretation, grammar/style notes
│   logging_utils.py        # Submission logging to CSV
│   calibration_wrappers.py # Temperature scaling classes
├── requirements.txt        # Dependency list
└── README.md               # Usage instructions

## Installation
1. Create a new Python virtual environment (recommended: Python 3.11).
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows PowerShell

2. Install dependencies

pip install -r requirements.txt

## Running the App
Run in project root (CEFR_App):___ which will open the local URL in browser
streamlit run app/app.py

## Usage
1. Test an Essay
   - Paste or upload an essay
   - Get CEFR prediction (A1–C2), lexical profile, unknown words
   - Get feedback that categorised by concise, detailed, features and charts

2. Practice Mode
   - Select a writing prompt
   - Get feedback with detailed analysis and charts

3. Progress Tracking
   - Submissions logged automatically
   - View progress vs. your CEFR goal