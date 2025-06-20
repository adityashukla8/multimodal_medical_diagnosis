import streamlit as st
import torch
from PIL import Image
from services.mongo import get_mongo_collection
from models.loaders import load_image_model, load_text_model
from services.embeddings import (
    preprocess_text,
    extract_text_features,
    extract_image_features,
    pad_embedding,
    combine_embeddings,
)
from services.search import vector_search
from utils.gcs_utils import read_image_from_gcs

# Set Streamlit page config
st.set_page_config(page_title="Search Similar Cases")

# Load MongoDB and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
collection = get_mongo_collection()

@st.cache_resource
def load_models():
    return load_image_model(), *load_text_model()

image_model, tokenizer, text_model = load_models()

# Function to show retrieved case results
def show_results(docs):
    for doc in docs:
        with st.expander(f"🧾 Case ID: {doc.get('case_id', '-')}", expanded=False):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(read_image_from_gcs(doc["image_path"]), use_container_width=True)
            with col2:
                st.markdown(f"**Score:** {doc.get('score', 0):.3f}")
                st.markdown(f"**Finding:** {doc.get('finding', '-')}")
                st.markdown(f"**Caption:** {doc.get('caption_x', '-')}")
                st.markdown(f"**GCS Path:** `{doc.get('image_path', '-')}`")

# Sample GCS test images
sample_images = {
    "Case 1": "gs://multicare-dataset/pmc4/pmc461/pmc4614622_emerg-2-48-g002_undivided_1_1.jpg",
    "Case 2": "gs://multicare-dataset/pmc7/pmc726/pmc7262379_gr3_undivided_1_1.jpg",
    "Case 3": "gs://multicare-dataset/pmc4/pmc453/pmc4533487_nmc-54-673-g1_undivided_1_1.jpg"
}

# App Title & Instructions
st.title("🔍 Find Similar Cases")
st.markdown("#### *For diagnostic ease and triage assistance.*")
st.markdown("""
**How it works**:
- **Text Search**: Enter clinical notes to find similar cases.
- **Image Search**: Upload or reference a CT scan image to retrieve similar patient cases.
- **Combined Search**: Use both text and image inputs for best results.
""")

# Search Option
option = st.radio("Search similar cases using:", ["Text + Image", "Image"])

# Text + Image
if option == "Text + Image":
    example_text = ("head ct scan image after stereotactic aspiration")

    text = st.text_area("Enter clinical text:", value=example_text)

    uploaded_image = st.text_input("Enter GCS path of image (gs://...) *(Find links in Sample Cases below)*", value='gs://multicare-dataset/pmc4/pmc461/pmc4614622_emerg-2-48-g002_undivided_1_1.jpg')

    if st.button("Search"):
        if text.strip() and uploaded_image is not None:
            text_emb = extract_text_features(preprocess_text(text), tokenizer, text_model, device)
            img_emb = extract_image_features(uploaded_image, image_model)
            combined = combine_embeddings(img_emb, text_emb)
            results = vector_search(collection, combined)
            show_results(results)
        else:
            st.warning("Please provide both clinical text and a valid image.")

# Image only
elif option == "Image":
    uploaded_image = st.text_input("Enter GCS path of image (gs://...)")
    if st.button("Search"):
        img_emb = extract_image_features(uploaded_image, image_model)
        results = vector_search(collection, pad_embedding(img_emb.tolist()))
        show_results(results)

# Sample Images Section
st.markdown("### 🧪 Sample GCS Images for Testing")
for label, path in sample_images.items():
    with st.expander(f"📁 {label}"):
        st.code(path, language="bash")
        try:
            img = read_image_from_gcs(path)
            st.image(img, caption=label, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load image: {e}")
