import re
import numpy as np
from helpers.config import LEVELS, word2level, weights

token_re = re.compile(r"[a-z]+(?:'[a-z]+)?")

def tokenize(text):
    return token_re.findall(text.lower())

def lexical_profile(text, word2level=word2level, LEVELS=LEVELS, weights=weights):
    tokens = tokenize(text)
    n = len(tokens)
    counts = {lvl: 0 for lvl in LEVELS}
    unknown = 0
    wsum = 0
    for tok in tokens:
        lvl = word2level.get(tok)
        if lvl:
            counts[lvl] += weights[lvl]
            wsum += weights[lvl]
        else:
            unknown += 1
    props = {lvl: counts[lvl]/wsum if wsum else 0 for lvl in LEVELS}
    return props, unknown/n if n else 0

def text_profile(text):
    """Extract simple structural features: avg sentence length, TTR."""
    sents = re.split(r"[.!?]", text)
    avg_len = np.mean([len(tokenize(s)) for s in sents if s.strip()])
    ttr = len(set(tokenize(text))) / max(1, len(tokenize(text)))
    return np.array([avg_len, ttr])