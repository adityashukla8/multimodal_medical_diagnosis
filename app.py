import streamlit as st
from services.mongo import get_mongo_collection
from models.loaders import load_image_model, load_text_model
from services.embeddings import *
from services.search import vector_search
from utils.gcs_utils import read_image_from_gcs
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
collection = get_mongo_collection()

@st.cache_resource
def load_models():
    return load_image_model(), *load_text_model()

image_model, tokenizer, text_model = load_models()

def show_results(docs):
    for doc in docs:
        with st.expander(f"🧾 Case ID: {doc.get('case_id', '-')}", expanded=False):
            col1, col2 = st.columns([1, 2])  # Adjust ratio as needed

            with col1:
                st.image(read_image_from_gcs(doc["image_path"]), use_container_width=True)

            with col2:
                st.markdown(f"**Score:** {doc.get('score', 0):.3f}")
                st.markdown(f"**Finding:** {doc.get('finding', '-')}")
                st.markdown(f"**Caption:** {doc.get('caption_x', '-')}")
                st.markdown(f"**GCS Path:** `{doc.get('image_path', '-')}`")
                
st.title("🔍 MedFuse Multimodal Case Search")
option = st.radio("Search using:", ["Text", "Image", "Text + Image"])

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
