import numpy as np
import streamlit as st
from helpers.features import tokenize, lexical_profile, text_profile
from helpers.config import LEVELS, word2level, weights

def top_k_predictions(probs, k=2):
    sorted_idx = np.argsort(probs)[::-1]  # sort descending
    top_levels = [(LEVELS[i], probs[i]) for i in sorted_idx[:k]]
    return top_levels

def interpret_predictions(probs, text, top_k=2, mode="Concise"):
    """
    Interpret model predictions into human-readable feedback.

    Args:
        probs (list/np.array): Probabilities per CEFR level.
        top_k (int): Number of top levels to show.
        mode (str): "Concise" or "Detailed".
    """
    # Get top-k predictions
    top_preds = top_k_predictions(probs, k=top_k)
    main_level, main_score = top_preds[0]

    feedback = []

    descriptors = {
    "A1": "You can use very basic everyday expressions and simple sentences. Writing is limited to familiar topics with simple vocabulary.",
    "A2": "You can write short, simple texts on routine matters. Sentences may be linked with basic connectors like 'and', 'but', and 'because'.",
    "B1": "You can produce connected texts on familiar topics. Writing shows some ability to explain opinions, describe experiences, and link ideas with more varied connectors.",
    "B2": "You can write clear, detailed texts on a range of subjects. You can express viewpoints, argue for or against ideas, and maintain good cohesion across paragraphs.",
    "C1": "You can write well-structured texts on complex subjects. Writing is flexible, with effective use of organizational patterns, varied vocabulary, and clear style.",
    "C2": "You can write precisely, fluently, and persuasively in almost any context. Writing shows full control of language, including nuance, register, and idiomatic expression."
    }

    if mode == "Concise":
        # Only show best prediction
        feedback.append(f"🎯 **Likely Level: {main_level} ({main_score:.0%})**")
        feedback.append(descriptors[main_level])  # from your description dict
        feedback.append(f"🔜 *Next step: Focus on skills for {LEVELS[min(LEVELS.index(main_level)+1, len(LEVELS)-1)]}.*")

    elif mode == "Detailed":
        # Show top-3 if possible
        top_preds3 = top_k_predictions(probs, k=3)
        pred_lines = [f"- {lvl}: {score:.0%}" for lvl, score in top_preds3]
        pred_block = "📊 **Predicted Levels:**\n" + "\n".join(pred_lines)
        feedback.append(pred_block)

        next_level_idx = min(LEVELS.index(main_level)+1, len(LEVELS)-1)
        next_level = LEVELS[next_level_idx]
        # Add CEFR description
        feedback.append(f"\n📝 At **{main_level}**, {descriptors[main_level]}\n\n")
        feedback.append(f"\n📝 For **{next_level}**, {descriptors[next_level]}\n\n")

        # Add simple feature-based interpretation
        avg_len, ttr = text_profile(text)  # assumes you saved the essay input
        props, unk = lexical_profile(text)

        feature_block = (
            "🔎 **Feature Highlights:**\n"
            f"- Avg sentence length: {avg_len:.1f}\n"
            f"- TTR (vocab diversity): {ttr:.2f}\n"
            f"- Unknown word proportion: {unk:.2f}"
        )
        feedback.append(feature_block)

        # Strengths & Suggestions
        strengths, suggestions = grammar_style_notes(avg_len, ttr, unk)

        if strengths:
            strengths_block = "✅ **Strengths:**\n" + "\n".join([f"- {s}" for s in strengths])
            feedback.append(strengths_block)

        if suggestions:
            suggestions_block = "💡 **Suggestions:**\n" + "\n".join([f"- {s}" for s in suggestions])
            feedback.append(suggestions_block)

    return feedback

def grammar_style_notes(avg_len, ttr, unk):
    strengths = []
    suggestions = []

    # Sentence length
    if avg_len < 8:
        suggestions.append("Sentences are very short; try combining ideas for more fluency.")
    elif avg_len < 15:
        strengths.append("Sentence length is balanced for clear communication. Consider expanding on points with examples")
    elif avg_len > 24:
        suggestions.append("Sentences are quite long; consider splitting for clarity.")
    else: 
        strengths.append("Sentences are long, engaging and flow well")


    # Vocabulary diversity
    if ttr < 0.25:
        suggestions.append("Vocabulary range is limited; try using more varied words.")
    elif ttr < 0.4:
        suggestions.append("Vocabulary shows some variety, but could be expanded with less common words.")
    else:
        strengths.append("Vocabulary shows strong diversity for this level.")


    # Unknown words
    if unk >= 0.25:
        suggestions.append("High use of uncommon words may affect clarity.")
    elif unk < 0.15:
        strengths.append("Word choice is safe and familiar, supporting clarity.")

    return strengths, suggestions

def feature_based_feedback(text, top_level, feature_stats=None):
    """
    Compare user text features against CEFR-level benchmarks.
    
    Args:
        text (str): User essay
        top_level (str): Predicted CEFR level
        feature_stats (pd.DataFrame): Optional dataframe with averages per level
    """
    tokens = tokenize(text)
    n_tokens = len(tokens)
    avg_len, ttr = text_profile(text)
    props, unk = lexical_profile(text, word2level, LEVELS, weights)

    msgs = []

    # --- Raw metrics
    msgs.append("📊 **Your Writing Metrics**")
    msgs.append(f"- Word count: {n_tokens}")
    msgs.append(f"- Avg sentence length: {avg_len:.1f}")
    msgs.append(f"- Avg word length: {np.mean([len(tok) for tok in tokens]):.2f}")
    msgs.append(f"- Type–Token Ratio (TTR): {ttr:.2f}")
    msgs.append(f"- Unknown word proportion: {unk:.2f}")

    # --- Compare to level benchmarks (if available)
    if feature_stats is not None:
        row = feature_stats.loc[feature_stats["Level"] == top_level].iloc[0]
        msgs.append("")  # <-- this creates a blank line in Streamlit
        msgs.append(f"\n📐 **Compared to typical {top_level} learners:**")
        msgs.append(f"- Your avg sentence length: {avg_len:.1f} (Peers: {row['Avg_Sent_Len']:.1f})")
        msgs.append(f"- Your TTR: {ttr:.2f} (Peers: {row['TTR']:.2f})")
        msgs.append(f"- Unknown word rate: {unk:.2f} (Peers: {row['Unknown_Prop']:.2f})")
        msgs.append(f"- Lexical Weighted: {sum(props[l] * (i+1) for i, l in enumerate(LEVELS)):.2f} (Peers: {row['Lex_Weighted']:.2f})")

    # Keep this tab **objective/benchmark-style**, no narrative
    return msgs

def word_count_feedback(text, min_words=50, ideal_range=(120, 300)):
    """Show word count and warnings about essay length."""
    word_count = len(tokenize(text))
    st.caption(f"✍️ Word count: {word_count}")

    if word_count < min_words:
        st.warning(f"⚠️ Your essay is quite short. Predictions may not be accurate with fewer than {min_words} words.")
    elif not ideal_range[0] <= word_count <= ideal_range[1]:
        st.info(f"ℹ️ For best results, try writing between {ideal_range[0]} and {ideal_range[1]} words.")
    
    return word_count