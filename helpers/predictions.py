import numpy as np
import torch
import re
from helpers.features import tokenize, lexical_profile
from helpers.config import LEVELS, device

def top_k_predictions(probs, k=2):
    """Return top-k CEFR levels and probabilities."""
    sorted_idx = np.argsort(probs)[::-1]
    return [(LEVELS[i], probs[i]) for i in sorted_idx[:k]]

def predict_with_calibration(text, tokenizer, base_model, calib_global, calib_vector, device):
    """Choose calibration dynamically: VectorT for B1–C2, GlobalT for A1/A2"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)

    # Quick uncalibrated check (fast)
    with torch.no_grad():
        logits = base_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # If top prediction looks like A1/A2 → use GlobalT
    top_idx = np.argmax(probs)
    if top_idx <= 1:  # 0 = A1, 1 = A2
        model = calib_global
    else:
        model = calib_vector

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    return probs

def hybrid_score_lr(text, tokenizer, base_model, scaler, hyb_lr, max_length=256):

    if not text.strip():
        return None

    # --- 1. DistilBERT embedding ---
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = base_model.distilbert(**inputs)   # encoder only
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # --- 2. Lexical features ---
    tokens = tokenize(text)
    n_tokens = len(tokens)
    props, unk = lexical_profile(text)
    avg_word_len = np.mean([len(tok) for tok in tokens]) if n_tokens > 0 else 0
    ttr = len(set(tokens)) / (n_tokens + 1e-6)
    lex_weighted = sum(props[lvl] * (i+1) for i, lvl in enumerate(LEVELS))
    avg_sent_len = np.mean([len(tokenize(s)) for s in re.split(r"[.!?]", text) if s.strip()]) if text.strip() else 0
    
    feat_vals = [props[lvl] for lvl in LEVELS] + [
        lex_weighted, unk, avg_word_len, ttr, n_tokens, avg_sent_len  
    ]
    lex_feat = np.array(feat_vals).reshape(1, -1)

    # --- 3. Combine (embedding + lexical) ---
    X_feat = np.hstack([cls_emb, lex_feat])  # (1, hidden_dim + lexical_dim)

    # --- 4. Scale & predict ---
    X_feat_s = scaler.transform(X_feat)
    raw_probs = hyb_lr.predict_proba(X_feat_s)[0]
    probs = align_proba_to_levels(raw_probs, hyb_lr.classes_, LEVELS)

    return probs

def align_proba_to_levels(probs, classes, levels):
    """
    Reorder sklearn predict_proba output to match `levels`.
    Works if `classes` are labels (e.g. "A1") OR numeric ids (e.g. 0..K-1).
    Falls back gracefully if it can't align.
    """
    probs = np.asarray(probs, dtype=float)
    classes = np.asarray(classes)

    aligned = np.zeros(len(levels), dtype=float)

    # Case 1: classes are strings like "A1","A2",...
    if classes.dtype.kind in ("U", "S", "O"):  # strings/object
        try:
            idx_map = {str(c): levels.index(str(c)) for c in classes}
            for j, c in enumerate(classes):
                aligned[idx_map[str(c)]] = probs[j]
        except ValueError:
            # Some label in classes not in levels → fallback (assume already aligned)
            aligned = probs.copy()

    # Case 2: classes are ints (label ids)
    else:
        # If ids are exactly 0..K-1, map by index → LEVELS order
        if set(classes.tolist()) == set(range(len(levels))):
            for j, c in enumerate(classes):
                aligned[int(c)] = probs[j]
        else:
            # Unrecognized id scheme → fallback (assume already aligned)
            aligned = probs.copy()

    s = aligned.sum()
    if s > 0:
        aligned /= s
    return aligned