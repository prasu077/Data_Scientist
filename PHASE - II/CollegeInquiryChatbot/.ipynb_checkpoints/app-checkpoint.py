import streamlit as st
import numpy as np
import pickle
import json
import random

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="College Inquiry Chatbot",
    page_icon="🎓",
    layout="centered"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stChatMessage {
    border-radius: 12px;
}

.title {
    text-align:center;
    color:#1E88E5;
    font-size:40px;
    font-weight:bold;
}

.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:20px;
}

.footer {
    text-align:center;
    color:gray;
    font-size:12px;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD RESOURCES
# ==================================================

@st.cache_resource
def load_resources():

    model = load_model("college_chatbot_gru.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("max_length.pkl", "rb") as f:
        max_length = pickle.load(f)

    with open("intentsnew.json", "r", encoding="utf-8") as f:
        intents = json.load(f)

    return model, tokenizer, encoder, max_length, intents


model, tokenizer, encoder, max_length, intents = load_resources()
def preprocess_query(text):

    text = text.lower()

    replacements = {
        "fees": "fee",
        "courses": "course",
        "admissions": "admission",
        "hostels": "hostel",
        "documents": "document",
        "facilities": "facility"
    }

    words = text.split()

    words = [replacements.get(w, w) for w in words]

    stop_words = {
        "what",
        "about",
        "can",
        "you",
        "please",
        "tell",
        "me",
        "is",
        "the",
        "a"
    }

    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# ==================================================
# PREDICT INTENT
# ==================================================

def predict_intent(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post",
        truncating="post"
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

# ==================================================
# RESPONSE
# ==================================================

def get_response(tag):

    for intent in intents["intents"]:

        if intent["tag"] == tag:

            return random.choice(intent["responses"])

    return "Sorry, I couldn't understand your question."


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="title">🎓 College Inquiry Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask about admissions, fees, hostel, placements, courses and more.</div>',
    unsafe_allow_html=True
)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("📚 About")

    st.info(
        """
        This AI-powered chatbot can answer:

        ✅ Admission Queries

        ✅ Hostel Information

        ✅ Fee Structure

        ✅ Placement Details

        ✅ Courses Offered

        ✅ Scholarships

        ✅ Contact Information
        """
    )

    st.markdown("---")

    st.success("Model: GRU Neural Network")

    st.success("Frontend: Streamlit")

    st.success("Backend: TensorFlow")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []
        st.rerun()

# ==================================================
# CHAT HISTORY
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ==================================================
# CHAT INPUT
# ==================================================

user_input = st.chat_input(
    "Type your question here..."
)

if user_input:

    # USER MESSAGE

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):

        st.markdown(user_input)

    # BOT PROCESSING

    with st.spinner("Thinking..."):

        tag, confidence = predict_intent(user_input)
        tag, confidence = predict_intent(user_input)

        st.sidebar.markdown("### Debug Info")
        st.sidebar.write("Predicted Tag:", tag)
        st.sidebar.write("Confidence:", f"{confidence:.2%}")
        if confidence < 0.60:

            response = (
                "I'm not sure I understood that. "
                "Could you please rephrase your question?"
            )

        else:

            response = get_response(tag)

    # BOT MESSAGE

    bot_message = f"""
{response}

**Confidence:** {confidence:.2%}
"""

    with st.chat_message("assistant"):

        st.markdown(bot_message)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": bot_message
        }
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div class='footer'>
    Developed using TensorFlow GRU & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)