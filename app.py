import os
import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import docx
import csv
import joblib
import json
from pathlib import Path

# Helpers
from helpers.features import lexical_profile
from helpers.predictions import predict_with_calibration, hybrid_score_lr, top_k_predictions
from helpers.feedback import interpret_predictions, feature_based_feedback, word_count_feedback
from helpers.logging_utils import log_submission, clear_log
from helpers.calibration_wrappers import ModelWithTemperature, ModelWithVectorTemp
from helpers.config import (
    BASE_DIR,
    HYBRID_SCALER, HYBRID_LR,
    FEATURE_STATS_PATH,
    PROMPTS_PATH, LOG_PATH,
    LEVELS, device, word2level, weights, load_model
)

# Ensure predictions_log.csv exists with header
if not LOG_PATH.exists():
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Username", "Timestamp", "Essay", "Predicted_Level", "Confidence"])

with open(FEATURE_STATS_PATH, "r") as f:
    FEATURE_STATS = pd.DataFrame(json.load(f))

tokenizer, base_model = load_model()

# --- Load calibration wrappers ---
calib_global = ModelWithTemperature(base_model)
calib_global.temperature.data = torch.load(BASE_DIR / "models" / "temperature_global.pt")

calib_vector = ModelWithVectorTemp(base_model, num_classes=6)
calib_vector.temperatures.data = torch.load(BASE_DIR / "models" / "temperature_vector.pt")

# --- Safe loaders ---
def safe_load_joblib(path, name):
    if not path.exists():
        st.error(f"Missing {name}: {path}")
        st.stop()
    return joblib.load(path)

def safe_load_pickle(path, name):
    if not path.exists():
        st.error(f"Missing {name}: {path}")
        st.stop()
    return pd.read_pickle(path)

# --- Hybrid components ---
scaler = safe_load_joblib(HYBRID_SCALER, "Hybrid Scaler")
hyb_lr = safe_load_joblib(HYBRID_LR, "Hybrid Logistic Regression")

# --- Streamlit UI ---
st.set_page_config(page_title="CEFR Predictor", layout="wide")
st.title("🌍 CEFR Level Classifier (MVP)")

# --- Global user state ---
if "username" not in st.session_state:
    st.session_state.username = ""

if "goal" not in st.session_state:
    st.session_state.goal = None

# Input username once (keeps value after reruns)
st.session_state.username = st.text_input(
    "Enter your name (for tracking):",
    value=st.session_state.username
)

# Show current user/goal everywhere
if st.session_state.username:
    st.caption(f"👤 Current user: {st.session_state.username} | 🎯 Goal: {st.session_state.goal if st.session_state.goal else 'Not set'}")

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["📖 Test an Essay", "✍️ Practice Mode", "📈 My Progress"])

with tab1:
    st.subheader("Enter your essay")

    uploaded_file = st.file_uploader("Upload an essay (TXT or DOCX)", type=["txt", "docx"])

    text_input = ""
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            text_input = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            text_input = "\n".join([para.text for para in doc.paragraphs])

    # Pre-fill the textarea with uploaded text
    text_input = st.text_area("Paste or edit essay here:", value=text_input, height=200)
    word_count = word_count_feedback(text_input)

    if st.button("Predict CEFR"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            
            # DistilBERT prediction
            probs_bert = predict_with_calibration(text_input, tokenizer, base_model, calib_global, calib_vector, device)
            top2_bert = top_k_predictions(probs_bert, k=2)
            st.success("✍️ The Stylist (Fluency & Structure):") #Distillbert Prediction
            for lvl, score in top2_bert:
                st.write(f"- {lvl}: {score:.0%}")
            
            # Hybrid prediction
            probs_hybrid = hybrid_score_lr(text_input, tokenizer, base_model, scaler, hyb_lr)
            top2_hybrid = top_k_predictions(probs_hybrid, k=2)
            st.info("📚 The Word Analyst (Vocabulary & Features):") #Hybrid LR Prediction
            for lvl, score in top2_hybrid:
                st.write(f"- {lvl}: {score:.0%}")
            log_submission(st.session_state.username, text_input, top2_hybrid)

            # --- Feedback & Analysis ---
            st.subheader("📘 Feedback")

            # Precompute everything once
            top_level = top2_hybrid[0][0]
            feedback_concise = interpret_predictions(probs_hybrid, text_input, top_k=2, mode="Concise")
            feedback_detailed = interpret_predictions(probs_hybrid, text_input, top_k=2, mode="Detailed")
            feature_msgs = feature_based_feedback(text_input, top_level, FEATURE_STATS)

            tab_concise, tab_detailed, tab_features, tab_visuals = st.tabs(
                ["📘 Concise", "📖 Detailed", "🔍 Features", "📊 Charts", ]
            )

            with tab_concise:
                st.write("### 📘 Concise Feedback")
                for msg in feedback_concise:
                    st.markdown(msg)
                    st.write("")   

            with tab_detailed:
                st.write("### 📖 Detailed Feedback")
                for block in feedback_detailed:
                    st.markdown(block)
                    st.write("")   

            with tab_features:
                st.write(f"### 🔍 Feature-based Analysis (vs average writer at {top_level})")
                for msg in feature_msgs:
                    st.markdown(msg)


            with tab_visuals: 
                st.write("### 📊 Probability Distributions")

                fig, ax = plt.subplots(2, 1, figsize=(6, 6))

                # DistilBERT chart
                ax[0].bar(LEVELS, probs_bert)
                ax[0].set_title("DistilBERT Probabilities")
                ax[0].set_ylim(0, 1)

                # Hybrid chart
                ax[1].bar(LEVELS, probs_hybrid, color="orange")
                ax[1].set_title("Hybrid Probabilities")
                ax[1].set_ylim(0, 1)

                st.pyplot(fig)

                st.write("### 🔤 Lexical Profile")
                props, unk = lexical_profile(text_input, word2level, LEVELS, weights)
                fig, ax = plt.subplots()
                ax.bar(props.keys(), props.values())
                ax.set_ylim(0, 1)
                st.pyplot(fig)

                st.caption(f"Unknown proportion: {unk:.2f}")

        # Optional: Clear saved predictions
    if st.button("🗑️ Clear Saved Predictions"):
        clear_log()
        st.success("Prediction log cleared and reset!")


with tab2:
    st.subheader("Practice Mode")
    st.write("Pick a prompt and test yourself.")

    # Load prompts (CSV: id, cefr_min, cefr_max, topic, prompt_text)
    try:
        df_prompts = pd.read_csv(PROMPTS_PATH)
    except FileNotFoundError:
        st.warning("⚠️ prompts.csv not found in data/. Using fallback prompts.")
        df_prompts = pd.DataFrame([
            {"id": 1, "cefr_min": "A2", "cefr_max": "B1", "topic": "Daily routine",
            "prompt_text": "Describe your morning routine on weekdays."},
            {"id": 2, "cefr_min": "B1", "cefr_max": "B2", "topic": "Study habits",
            "prompt_text": "Do you prefer studying alone or with friends? Explain why."}
        ])
 
    # Determine current level (best of last N submissions)
    if LOG_PATH.exists() and st.session_state.username:
        df_log = pd.read_csv(LOG_PATH)
        df_user = df_log[df_log["Username"] == st.session_state.username]
        if not df_user.empty:
            N = 10
            df_recent = df_user.tail(N)
            current_level = df_recent.iloc[-1]["Predicted_Level"]
        else:
            current_level = "B1"  # default if no history
    else:
        current_level = "B1"

    # Filter prompts within ±1 CEFR band
    cur_idx = LEVELS.index(current_level)
    allowed_levels = LEVELS[max(0, cur_idx-1): min(len(LEVELS), cur_idx+2)]
    df_prompts_filtered = df_prompts[
        df_prompts["cefr_min"].isin(allowed_levels) | df_prompts["cefr_max"].isin(allowed_levels)
    ]

    topic = st.selectbox("Choose a prompt:", df_prompts_filtered["topic"])
    prompt_text = df_prompts_filtered.loc[df_prompts_filtered["topic"] == topic, "prompt_text"].values[0]
    st.info(prompt_text)

    practice_input = st.text_area("Write your answer:", height=150)
    word_count = word_count_feedback(practice_input)

    if st.button("Submit Practice"):
        if not practice_input.strip():
            st.warning("Please write something before submitting.")
        else:
            # DistilBERT prediction
            probs_bert = predict_with_calibration(practice_input, tokenizer, base_model, calib_global, calib_vector, device)
            top2_bert = top_k_predictions(probs_bert, k=2)

            st.success("✍️ The Stylist (Fluency & Structure):")
            for lvl, score in top2_bert:
                st.write(f"- {lvl}: {score:.0%}")

            # Hybrid prediction
            probs_hybrid = hybrid_score_lr(practice_input, tokenizer, base_model, scaler, hyb_lr)
            top2_hybrid = top_k_predictions(probs_hybrid, k=2)

            st.info("📚 The Word Analyst (Vocabulary & Features):")
            for lvl, score in top2_hybrid:
                st.write(f"- {lvl}: {score:.0%}")

            # Log submission
            log_submission(st.session_state.username, practice_input, top2_hybrid)

            # Detailed feedback only (always on in Practice Mode)
            st.subheader("📘 Feedback")
            feedback_detailed = interpret_predictions(probs_hybrid, practice_input, top_k=2, mode="Detailed")
            for block in feedback_detailed:
                st.markdown(block)
                st.write("")

with tab3:
    if not st.session_state.username:
        st.info("👤 Please enter a username above to view your progress.")
    else:
        # Show goal setting
        st.subheader(f"📈 Progress for {st.session_state.username}")
        st.session_state.goal = st.selectbox(
            "🎯 Set your CEFR goal:",
            LEVELS,
            index=LEVELS.index(st.session_state.goal) if st.session_state.goal else 0
        )

        # Load submission log
        if LOG_PATH.exists():
            df_log = pd.read_csv(LOG_PATH)
            df_user = df_log[df_log["Username"] == st.session_state.username]

            if not df_user.empty:
                # 🔹 Focus only on the most recent N submissions
                N = 10   # or 20 if you want a longer window
                df_recent = df_user.tail(N)

                # Current and best levels in the window
                current_level = df_recent.iloc[-1]["Predicted_Level"]
                best_idx = df_recent["Predicted_Level"].map(lambda x: LEVELS.index(x)).max()
                best_level = LEVELS[best_idx]

                st.info(f"📝 Latest submission: **{current_level}**")
                st.info(f"📈 Best level (last {N} essays): **{best_level}**")
                st.info(f"🎯 Goal: **{st.session_state.goal}**")

                # Progress bar (based on best recent)
                progress = (best_idx+1) / (LEVELS.index(st.session_state.goal)+1)
                st.progress(min(progress, 1.0))

                # Trendline (confidence over time)
                df_recent["Timestamp"] = pd.to_datetime(df_recent["Timestamp"])
                st.line_chart(df_recent.set_index("Timestamp")["Confidence"])

                # Bar chart (level counts in window)
                st.bar_chart(df_recent["Predicted_Level"].value_counts())

            else:
                st.info("No submissions found for this user yet.")
        else:
            st.info("No submissions logged yet.")