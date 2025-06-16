# app.py

import os
import streamlit as st
import cv2
import numpy as np
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
import torch
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
import re
import string
import nltk
from nltk.corpus import stopwords
from PIL import Image
import matplotlib.pyplot as plt
from math import ceil

nltk.download("stopwords")

# --- Setup ---
def get_env_variable(var_name):
    return os.environ.get(var_name).replace("\\x3a", ":")

MONGO_URI = get_env_variable("MONGO_URI")
DB_NAME = os.environ.get("DB_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
VECTOR_INDEX_NAME = "ct_vector_index"

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    image_model = DenseNet121(weights="imagenet", include_top=False, pooling="avg")
    image_model.trainable = False
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    text_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
    text_model.eval()
    return image_model, tokenizer, text_model

image_model, tokenizer, text_model = load_models()

stop_words = set(stopwords.words("english"))

# --- Utils ---
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

def extract_text_features(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

def extract_image_features(image_path):
    try:
        img_data = tf.io.read_file(image_path)
        img = tf.image.decode_image(img_data, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [224, 224])
        img_array = tf.expand_dims(img, axis=0)
        img_array = preprocess_input(img_array)
        features = image_model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"[ERROR] Image read failed: {e}")
        return None

def pad_embedding(embedding, target_dim=1792):
    return embedding + [0.0] * (target_dim - len(embedding))

def combine_embeddings(image_emb=None, text_emb=None):
    if image_emb is not None and text_emb is not None:
        return image_emb.tolist() + text_emb.tolist()
    elif image_emb is not None:
        return pad_embedding(image_emb.tolist(), 1792)
    elif text_emb is not None:
        return pad_embedding(text_emb.tolist(), 1792)
    else:
        return []

def vector_search(query_embedding, top_k=9, num_candidates=100):
    query_embedding = [float(x) for x in query_embedding]
    cursor = collection.aggregate([
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "combined_features",
                "queryVector": query_embedding,
                "numCandidates": num_candidates,
                "limit": top_k,
            }
        },
        {
            "$project": {
                "_id": 1,
                "image_path": 1,
                "caption_x": 1,
                "finding": 1,
                "case_id": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        }
    ])
    return list(cursor)

def read_image_from_gcs(image_path):
    try:
        with tf.io.gfile.GFile(image_path, 'rb') as f:
            img_bytes = f.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"[ERROR] GCS image read: {e}")
        return np.zeros((224, 224, 3), dtype=np.uint8)

def show_results(docs):
    cols = st.columns(3)
    for i, doc in enumerate(docs):
        col = cols[i % 3]
        with col:
            st.image(read_image_from_gcs(doc["image_path"]), caption=f"Score: {doc['score']:.3f}", use_column_width=True)
            st.markdown(f"**Case ID**: {doc.get('case_id', '-')}")
            st.markdown(f"**Finding**: {doc.get('finding', '-')}")
            st.markdown(f"**Caption**: {doc.get('caption_x', '-')}")
            st.markdown("---")

# --- UI ---
st.title("🔍 MedFuse Multimodal Case Search")
st.write("Search similar medical cases by image, text, or both using embeddings.")

option = st.radio("Search using:", ["Text", "Image", "Text + Image"])

if option == "Text":
    query = st.text_area("Enter clinical text...")
    if st.button("Search"):
        cleaned = preprocess_text(query)
        text_emb = extract_text_features(cleaned)
        results = vector_search(pad_embedding(text_emb.tolist()))
        show_results(results)

elif option == "Image":
    uploaded_image = st.text_input("Enter GCS path of image (gs://...)")
    if st.button("Search"):
        img_emb = extract_image_features(uploaded_image)
        results = vector_search(pad_embedding(img_emb.tolist()))
        show_results(results)

elif option == "Text + Image":
    query = st.text_area("Enter clinical text...")
    uploaded_image = st.text_input("Enter GCS path of image (gs://...)")
    if st.button("Search"):
        text_emb = extract_text_features(preprocess_text(query))
        img_emb = extract_image_features(uploaded_image)
        combined = combine_embeddings(img_emb, text_emb)
        results = vector_search(combined)
        show_results(results)
