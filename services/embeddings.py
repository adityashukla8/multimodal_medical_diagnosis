import re, string
import torch
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in text.split() if w not in stop_words)

def extract_text_features(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

def extract_image_features(image_path, model):
    try:
        img_data = tf.io.read_file(image_path)
        img = tf.image.decode_image(img_data, channels=3, expand_animations=False)
        img = tf.image.resize(tf.image.convert_image_dtype(img, tf.float32), [224, 224])
        img_array = preprocess_input(tf.expand_dims(img, axis=0))
        return model.predict(img_array, verbose=0).flatten()
    except Exception as e:
        print(f"[ERROR] Image feature extraction failed: {e}")
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
