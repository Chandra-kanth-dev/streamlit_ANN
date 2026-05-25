# ============================================
# TITANIC SURVIVAL PREDICTION APP
# WITHOUT TENSORFLOW
# USING MANUAL LOGIC
# ============================================

# ============================================
# IMPORT LIBRARIES
# ============================================

import streamlit as st
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
# CREATE SCALER
# ============================================

scaler = MinMaxScaler()

sample_data = np.array([
    [1, 1, 0],
    [3, 80, 600]
])

scaler.fit(sample_data)

# ============================================
# SIGMOID FUNCTION
# ============================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ============================================
# MANUAL WEIGHTS
# ============================================

# Input -> Hidden
w_input_hidden = np.array([
    [0.11, 0.21],
    [0.14, 0.24],
    [0.17, 0.27]
])

# Hidden Bias
b_hidden = np.array([0.1, 0.1])

# Hidden -> Output
w_hidden_output = np.array([
    [0.31],
    [0.34]
])

# Output Bias
b_output = 0.1

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
- Manual Forward Propagation
- Sigmoid Activation Function
- Real-time Prediction System

The prediction is based on:
- Passenger Class
- Age
- Fare

This version works WITHOUT TensorFlow.
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
    # FORWARD PROPAGATION
    # ========================================

    # Hidden Layer
    hidden_input = np.dot(input_scaled, w_input_hidden) + b_hidden

    hidden_output = sigmoid(hidden_input)

    # Output Layer
    final_input = np.dot(hidden_output, w_hidden_output) + b_output

    final_output = sigmoid(final_input)

    probability = float(final_output[0][0])

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

    # Bar Chart
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

    # ========================================
    # SHOW INTERNAL CALCULATIONS
    # ========================================

    st.markdown("---")
    st.markdown("## 🧠 Neural Network Calculations")

    st.write("### Hidden Layer Input")
    st.write(hidden_input)

    st.write("### Hidden Layer Output")
    st.write(hidden_output)

    st.write("### Final Output")
    st.write(final_output)

# ============================================
# FOOTER
# ============================================

st.markdown("""
---
### Developed using Streamlit +  ANN Logic
""")
