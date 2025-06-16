import streamlit as st
import torch
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

st.set_page_config(page_title="Search Similar Cases")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
collection = get_mongo_collection()

@st.cache_resource
def load_models():
    return load_image_model(), *load_text_model()

image_model, tokenizer, text_model = load_models()

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

st.title("🔍 Find Similar Cases")
st.markdown("#### *For diagnostic ease and triage assistance.*")
st.markdown("""**How it works**:
- **Text Search**: Enter clinical notes to find similar cases.
- **Image Search**: Provide a GCS path to an image to retrieve similar cases. 
- **Combined Search**: Use both text and image inputs for a more comprehensive search.

**Sample GCS image paths for testing:**
- `gs://multicare-dataset/pmc4/pmc461/pmc4614622_emerg-2-48-g002_undivided_1_1.jpg`
- `gs://multicare-dataset/pmc7/pmc726/pmc7262379_gr3_undivided_1_1.jpg`
- `gs://multicare-dataset/pmc4/pmc453/pmc4533487_nmc-54-673-g1_undivided_1_1.jpg`
""")

option = st.radio("Search similar cases using:", ["Text + Image", "Image", "Text"])

if option == "Text":
    query = st.text_area("Enter clinical text...")
    if st.button("Search"):
        cleaned = preprocess_text(query)
        text_emb = extract_text_features(cleaned, tokenizer, text_model, device)
        results = vector_search(collection, pad_embedding(text_emb.tolist()))
        show_results(results)

elif option == "Image":
    uploaded_image = st.text_input("Enter GCS path of image (gs://...)")
    if st.button("Search"):
        img_emb = extract_image_features(uploaded_image, image_model)
        results = vector_search(collection, pad_embedding(img_emb.tolist()))
        show_results(results)

elif option == "Text + Image":
    query = st.text_area("Enter clinical text...")
    uploaded_image = st.text_input("Enter GCS path of image (gs://...)")
    if st.button("Search"):
        text_emb = extract_text_features(preprocess_text(query), tokenizer, text_model, device)
        img_emb = extract_image_features(uploaded_image, image_model)
        combined = combine_embeddings(img_emb, text_emb)
        results = vector_search(collection, combined)
        show_results(results)
