from transformers import AutoTokenizer, AutoModel
import torch
from tensorflow.keras.applications import DenseNet121

def load_image_model():
    model = DenseNet121(weights="imagenet", include_top=False, pooling="avg")
    model.trainable = False
    return model

def load_text_model():
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model.eval()
    return tokenizer, model
