from pathlib import Path
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Base paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models"
LOG_PATH = DATA_PATH / "predictions_log.csv"

# --- Specific model/data files ---
MODEL_NAME            = "JohnnAsher/cefr-distilbert" # Primary, load from Hugging Face Hub
LOCAL_MODEL_PATH      = MODEL_PATH / "distilbert_clean" # Fallback, load from local 'models/distilbert_clean/' folder

HYBRID_SCALER         = MODEL_PATH / "hybrid_scaler.pkl" 
HYBRID_LR             = MODEL_PATH / "hybrid_lr.pkl"
WLIST                 = DATA_PATH / "cefr_wordlist_clean.pkl"
FEATURE_STATS_PATH    = DATA_PATH / "feature_stats.json"
PROMPTS_PATH          = DATA_PATH / "prompts.csv"

# --- Constants ---
LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

# --- Torch device ---
device = torch.device("cpu")

# --- Wordlist + lexical weights ---
df_wlist = pd.read_pickle(WLIST)
word2level = dict(zip(df_wlist["headword"], df_wlist["CEFR"]))
weights = {lvl: i+1 for i, lvl in enumerate(LEVELS)}

# --- Load model + tokenizer ---
def load_model():
    try:
        # Try Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32
        ).to(device)
        print(f"Loaded model from Hugging Face Hub: {MODEL_NAME}")
    except Exception as e:
        # Fallback to local model folder
        print(f"⚠️ Falling back to local model due to error: {e}")
        tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_PATH), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(LOCAL_MODEL_PATH),
            local_files_only=True,
            torch_dtype=torch.float32
        ).to(device)
        print(f"Loaded model from local path: {LOCAL_MODEL_PATH}")
    return tokenizer, model