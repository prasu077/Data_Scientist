import streamlit as st
import tensorflow as tf
import pickle
import json
import random
import numpy as np
import pandas as pd
import csv
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Orchids University Chatbot",
    page_icon="🎓",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #1e3a8a;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: #64748b;
    margin-bottom: 20px;
}

.footer {
    text-align:center;
    color:gray;
    font-size:14px;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# CONSTANTS
# =====================================

MAX_LENGTH = 9
CONFIDENCE_THRESHOLD = 0.75

# =====================================
# LOAD FILES
# =====================================

@st.cache_resource
def load_model_files():

    model = tf.keras.models.load_model(
        "gru_chatbot_model.h5"
    )

    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    with open("label_encoder.pkl", "rb") as file:
        encoder = pickle.load(file)

    with open("intentsnew.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    return model, tokenizer, encoder, data


model, tokenizer, encoder, data = load_model_files()

# =====================================
# FUNCTIONS
# =====================================

def predict_intent(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post"
    )

    prediction = model.predict(
        padded,
        verbose=0
    )

    confidence = float(np.max(prediction))

    predicted_class = np.argmax(prediction)

    tag = encoder.inverse_transform(
        [predicted_class]
    )[0]

    return tag, confidence


def get_response(tag):

    for intent in data["intents"]:

        if intent["tag"] == tag:

            return random.choice(
                intent["responses"]
            )

    return "Sorry, I couldn't understand."


def log_unknown_question(question):

    filename = "unknown_questions.csv"

    file_exists = os.path.isfile(filename)

    with open(
        filename,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Question"])

        writer.writerow([question])


# =====================================
# SESSION STATE
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🎓 Chatbot Panel")

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Model Information")

st.sidebar.info(
    """
Model : GRU

Accuracy : 99.12%

Vocabulary : 333

Intents : 41
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("⚡ Quick Questions")

quick_questions = [
    "Admission process",
    "Hostel fee",
    "Courses offered",
    "Placement details",
    "Scholarship details"
]

for q in quick_questions:

    if st.sidebar.button(q):

        st.session_state.messages.append(
            {
                "role": "user",
                "content": q
            }
        )

        tag, confidence = predict_intent(q)

        if confidence >= CONFIDENCE_THRESHOLD:

            response = get_response(tag)

        else:

            response = (
                "Sorry, I couldn't understand your question."
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

st.sidebar.markdown("---")

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# =====================================
# HEADER
# =====================================

st.markdown(
    "<div class='main-title'>🎓 Orchids University</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Chatbot Assistant</div>",
    unsafe_allow_html=True
)

# =====================================
# DISPLAY CHAT HISTORY
# =====================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# =====================================
# USER INPUT
# =====================================

user_input = st.chat_input(
    "Ask your question here..."
)

if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    tag, confidence = predict_intent(
        user_input
    )

    if confidence >= CONFIDENCE_THRESHOLD:

        response = get_response(tag)

    else:

        response = (
            "Sorry, I couldn't understand your question. "
            "Please try asking in a different way."
        )

        log_unknown_question(
            user_input
        )

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

# =====================================
# FOOTER
# =====================================

st.markdown(
    """
    <div class='footer'>
    Developed for Orchids University | GRU Chatbot Project
    </div>
    """,
    unsafe_allow_html=True
)