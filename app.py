import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --- Load your trained model ---
try:
    model = pickle.load(open('rock_mine_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("❌ 'rock_mine_model.pkl' not found. Please place it in the same folder.")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="Rock vs Mine Predictor", layout="centered")
st.title("🪨 Rock vs 💣 Mine Prediction")
st.markdown("Upload a CSV file with **60 sonar readings** to predict if the object is a **Rock or a Mine**.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV (1 row × 60 columns, no headers)", type=["csv"])

# Instructions
st.info("""
📌 **CSV Format:**
- USE 1 row info
- 60 comma-separated values
- Avoid header row
""")

# Process the file
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None)

        if df.shape != (1, 60):
            st.error(f"❌ CSV must have exactly **1 row and 60 columns**. Found: {df.shape}")
            st.stop()

        input_data = df.values.reshape(1, -1)
        prediction = model.predict(input_data)
        
        # Try prediction probability if supported
        try:
            probs = model.predict_proba(input_data)[0]
            labels = model.classes_  # e.g., ['M', 'R']
            prediction_dict = dict(zip(labels, probs))
        except:
            # fallback for models without probability support
            prediction_dict = {'M': 0.5, 'R': 0.5}

        # Show prediction
        if prediction[0] == 'R':
            st.success("🎯 The object is predicted to be a **Rock**")
        else:
            st.success("💣 The object is predicted to be a **Mine**")

        # --- Bar Chart for Prediction ---
        st.subheader("🔢 Prediction Confidence")
        labels_map = {'R': 'Rock', 'M': 'Mine'}
        plot_labels = [labels_map.get(l, l) for l in prediction_dict.keys()]
        plot_values = list(prediction_dict.values())

        fig, ax = plt.subplots()
        bars = ax.bar(plot_labels, plot_values, color=['skyblue', 'salmon'], width=0.6)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probability (Rock vs Mine)")
        ax.bar_label(bars, fmt="%.2f", padding=5)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error reading the file: {e}")
