import streamlit as st
import json
import random
import urllib.parse
from pathlib import Path

# Use relative path for compatibility on local and cloud environments
BASE_PATH = Path(__file__).parent
BENCHMARK_PATH = BASE_PATH / "data/benchmark.json"

# Load the benchmark.json file
def load_data():
    if not BENCHMARK_PATH.exists():
        st.error(f"File not found: {BENCHMARK_PATH}")
        return {}

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

# Function to select a random topic and filter based on user selection
def get_filtered_text(data, selected_background):
    if not data:  # Handle case where data couldn't be loaded
        return "No data available", "No original text", {}

    random_topic_key = random.choice(list(data.keys()))
    topic_data = data[random_topic_key]
    topic_name = topic_data.get("topic", "Unknown Topic")
    original_text = topic_data.get("original_text", "No original text available.")

    tailored_texts = topic_data.get("tailored_texts", {})
    selected_texts = {}

    for model, categories in tailored_texts.items():
        for category, text_variants in categories.items():
            # Apply filtering based on background selection (exclude the selected background)
            if selected_background == "L" and category in ["CS", "CL"]:
                if isinstance(text_variants, dict):
                    selected_prompt = random.choice(list(text_variants.keys()))
                    selected_text = text_variants[selected_prompt]
                    selected_texts[f"{model} - {category} (Prompt {selected_prompt})"] = selected_text
                else:
                    selected_texts[f"{model} - {category}"] = text_variants
            elif selected_background == "CS" and category in ["L", "CL"]:
                if isinstance(text_variants, dict):
                    selected_prompt = random.choice(list(text_variants.keys()))
                    selected_text = text_variants[selected_prompt]
                    selected_texts[f"{model} - {category} (Prompt {selected_prompt})"] = selected_text
                else:
                    selected_texts[f"{model} - {category}"] = text_variants
            elif selected_background == "CL" and category in ["L", "CS"]:
                if isinstance(text_variants, dict):
                    selected_prompt = random.choice(list(text_variants.keys()))
                    selected_text = text_variants[selected_prompt]
                    selected_texts[f"{model} - {category} (Prompt {selected_prompt})"] = selected_text
                else:
                    selected_texts[f"{model} - {category}"] = text_variants

    return random_topic_key, topic_name, original_text, selected_texts

# Generate a shareable URL with topic and tailored texts encoded
def generate_shareable_url(topic_key, topic_name, original_text, tailored_texts, background):
    base_url = "https://francescacarlon-thesis-project-app-xqi66e.streamlit.app/"
    
    params = {
        "topic": topic_name,
        "original": original_text,
        "tailored": json.dumps(tailored_texts),
        "background": background
    }
    
    encoded_params = urllib.parse.urlencode(params)
    return f"{base_url}?{encoded_params}"

# Streamlit UI
st.title("Adaptive Benchmark Display (Based on Background)")

# Load data
data = load_data()

# User selects background
selected_background = st.selectbox("Select your background:", ["General", "L", "CS", "CL"])  # "General" shows everything

# Check if the URL contains preloaded data
query_params = st.query_params.to_dict()

if "topic" in query_params:
    # Load from URL
    topic_name = query_params["topic"]
    original_text = query_params["original"]
    tailored_texts = json.loads(query_params["tailored"])

    st.subheader(f"📌 Topic: {topic_name}")
    st.markdown(f"### 📜 Original Text:")
    st.info(original_text)

    st.markdown("### 🎯 Tailored Texts (Filtered Based on Selection):")
    for category, text in tailored_texts.items():
        with st.expander(f"🔹 {category}"):
            st.write(text)

    st.success(f"This topic was loaded from a shared link (Background: {query_params.get('background', 'General')})!")

else:
    if "history" not in st.session_state:
        st.session_state.history = []  # Store generated topics history

    if st.button("Generate Random Topic"):
        topic_key, topic_name, original_text, selected_texts = get_filtered_text(data, selected_background)
        shareable_link = generate_shareable_url(topic_key, topic_name, original_text, selected_texts, selected_background)

        st.session_state.history.append({"topic_name": topic_name, "link": shareable_link, "background": selected_background})

        st.subheader(f"📌 Topic: {topic_name}")
        st.markdown(f"### 📜 Original Text:")
        st.info(original_text)

        st.markdown("### 🎯 Tailored Texts (Filtered Based on Selection):")
        for category, text in selected_texts.items():
            with st.expander(f"🔹 {category}"):
                st.write(text)

        # Show shareable link
        st.markdown(f"🔗 **[Share this topic]({shareable_link})**")

    # Display previously generated topics with links
    if st.session_state.history:
        st.markdown("## 🔄 Previously Generated Topics:")
        for entry in st.session_state.history:
            st.markdown(f"- 🔗 [{entry['topic_name']} (Background: {entry['background']})]({entry['link']})")