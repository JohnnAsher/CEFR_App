from pathlib import Path
import torch
import pandas as pd

# --- Base paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models"
LOG_PATH = DATA_PATH / "predictions_log.csv"

# --- Specific model/data files ---
MODEL_NAME            = "JohnnAsher/cefr-distilbert"
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