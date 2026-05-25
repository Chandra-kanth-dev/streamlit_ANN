# ============================================
# TITANIC SURVIVAL PREDICTION APP
# ============================================

# ============================================
# IMPORT LIBRARIES
# ============================================

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# ============================================
# LOAD MODEL
# ============================================

model = tf.keras.models.load_model("titanic_ann_model.h5")

# ============================================
# CREATE SCALER
# ============================================

# Training dataset min/max assumptions
# Replace with actual training values if available

scaler = MinMaxScaler()

sample_data = np.array([
    [1, 1, 0],
    [3, 80, 600]
])

scaler.fit(sample_data)

# ============================================
# HEADER SECTION
# ============================================

st.markdown("""
# 🚢 Titanic Survival Prediction System
### Deep Learning Based Passenger Survival Prediction
""")

# ============================================
# PROJECT DESCRIPTION
# ============================================

st.markdown("""
---
## 📌 Project Description

This application predicts whether a passenger would survive during the Titanic disaster using:

- Artificial Neural Networks (ANN)
- TensorFlow Deep Learning Model
- Passenger Information
- Real-time Prediction System

The model is trained using:
- Passenger Class
- Age
- Fare

The trained model is deployed using Streamlit.
---
""")

# ============================================
# INPUT SECTION
# ============================================

st.markdown("## 🎯 Passenger Input Form")

col1, col2, col3 = st.columns(3)

# Passenger Class
with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

# Age
with col2:
    age = st.slider(
        "Age",
        1,
        80,
        25
    )

# Fare
with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=50.0
    )

# ============================================
# PREDICTION BUTTON
# ============================================

if st.button("Predict Survival"):

    # ========================================
    # PREPROCESS INPUT
    # ========================================

    input_data = np.array([[pclass, age, fare]])

    # Normalize Input
    input_scaled = scaler.transform(input_data)

    # ========================================
    # PREDICTION
    # ========================================

    prediction = model.predict(input_scaled)

    probability = float(prediction[0][0])

    # ========================================
    # OUTPUT SECTION
    # ========================================

    st.markdown("---")
    st.markdown("## 📊 Prediction Output")

    col1, col2, col3 = st.columns(3)

    # Prediction Result
    with col1:

        if probability > 0.5:
            st.success("✅ SURVIVED")
        else:
            st.error("❌ NOT SURVIVED")

    # Survival Probability
    with col2:
        st.metric(
            label="Survival Probability",
            value=f"{probability*100:.2f}%"
        )

    # Confidence Score
    with col3:

        confidence = max(probability, 1 - probability)

        st.metric(
            label="Confidence Score",
            value=f"{confidence*100:.2f}%"
        )

    # ========================================
    # VISUALIZATION
    # ========================================

    st.markdown("---")
    st.markdown("## 📈 Probability Visualization")

    survive_prob = probability
    not_survive_prob = 1 - probability

    chart_data = pd.DataFrame({
        "Category": ["Survived", "Not Survived"],
        "Probability": [survive_prob, not_survive_prob]
    })

    st.bar_chart(
        chart_data.set_index("Category")
    )

    # ========================================
    # PIE CHART
    # ========================================

    fig, ax = plt.subplots()

    ax.pie(
        [survive_prob, not_survive_prob],
        labels=["Survived", "Not Survived"],
        autopct='%1.1f%%'
    )

    st.pyplot(fig)

# ============================================
# FOOTER
# ============================================

st.markdown("""
---
### Developed using TensorFlow + Streamlit
""")