import csv
import datetime
from helpers.config import LOG_PATH

def log_submission(username, text, preds):
    """Append a submission to the predictions log (CSV)."""
    header = ["Username", "Timestamp", "Essay", "Predicted_Level", "Confidence"]
    file_exists = LOG_PATH.exists()

    with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for lvl, score in preds:
            writer.writerow([
                username,
                datetime.datetime.now().isoformat(),
                text[:50] + "...",
                lvl,
                f"{score:.2f}"
            ])

def clear_log():
    """Clear the predictions log and reset with header."""
    header = ["Username", "Timestamp", "Essay", "Predicted_Level", "Confidence"]
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)