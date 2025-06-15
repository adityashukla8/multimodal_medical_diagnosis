import os
import csv
import logging
from io import StringIO

import pandas as pd
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

import torch
from transformers import AutoTokenizer, AutoModel
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

from ipdb import set_trace as ipdb

def get_env_variable(var_name):
    value = os.environ.get(var_name).replace("\\x3a", ":")
    return value

# Constants
CSV_PATH = str(get_env_variable("CSV_PATH"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 100))
DB_NAME = os.environ.get("DB_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stop_words = set(stopwords.words('english'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_headers(csv_path):
    """Read CSV headers from the first row."""
    print(csv_path)
    df = pd.read_csv(csv_path, nrows=1)
    return df.columns.tolist()

def parse_row_to_dict(row, header):
    """Parse a CSV row string into a dictionary using provided headers."""
    fields = next(csv.reader([row]))
    return dict(zip(header, fields))

class GenerateEmbeddingsDoFn(beam.DoFn):
    def setup(self):
        self.image_model = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
        self.image_model.trainable = False

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.text_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
        self.text_model.eval()

        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def extract_text_features(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = self.text_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    def process_image(self, gcs_image_path):
        try:
            img_data = tf.io.read_file(gcs_image_path)
            img = tf.image.decode_image(img_data, channels=3, expand_animations=False)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [224, 224])
            img_array = tf.expand_dims(img, axis=0)
            return preprocess_input(img_array)
        except Exception as e:
            logging.error(f"Image loading error: {e}")
            return None

    def process(self, element):
        image_path = element.get('image_path')
        case_text = element.get('case_text', "")

        # Process image
        image_embedding = None
        img_array = self.process_image(image_path)
        if img_array is not None:
            features = self.image_model.predict(img_array, verbose=0)
            image_embedding = features.flatten()

        # Process text
        text_embedding = None
        try:
            cleaned = self.preprocess_text(case_text)
            text_embedding = self.extract_text_features(cleaned)
        except Exception as e:
            logging.error(f"Text embedding error: {e}")

        element['image_features'] = image_embedding.tolist() if image_embedding is not None else None
        element['text_features'] = text_embedding.tolist() if text_embedding is not None else None

        if image_embedding is not None and text_embedding is not None:
            element['combined_features'] = image_embedding.tolist() + text_embedding.tolist()
        else:
            element['combined_features'] = None

        yield element

class WriteToMongoDB(beam.DoFn):
    """A Beam DoFn for writing documents to MongoDB in batches."""
    def __init__(self, mongo_uri, db_name, collection_name, batch_size=BATCH_SIZE):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.docs_batch = []

    def setup(self):
        logger.info(f"Connecting to MongoDB database: {self.db_name}, collection: {self.collection_name}")
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

    def process(self, element):
        doc = {
            "image_id": element.get('image_id', ""),
            "image_path": element.get('image_path', ""),
            "image_type": element.get('image_type', ""),
            "image_technique": element.get('image_technique', ""),
            "site": element.get('site', ""),
            "finding": element.get('finding', ""),
            "caption_x": element.get('caption_x', ""),
            "patient_id": element.get('patient_id', ""),
            "age": element.get('age', None),
            "gender": element.get('gender', ""),
            "case_id": element.get('case_id', ""),
            "case_text": element.get('case_text', ""),
            # "image_features": element.get('image_features', []),
            # "text_features": element.get('text_features', []),
            "combined_features": element.get('combined_features', []),
        }

        # self.collection.insert_one(doc)

        self.docs_batch.append(doc)
        if len(self.docs_batch) >= self.batch_size:
            self._flush_batch()
        # yield beam.pvalue.TaggedOutput('inserted', doc)

    def _flush_batch(self):
        try:
            self.collection.insert_many(self.docs_batch)
            logger.info(f"Inserted batch of {len(self.docs_batch)} records.")
        except Exception as e:
            logger.error(f"Error inserting batch: {e}")
        finally:
            self.docs_batch = []

    def finish_bundle(self):
        if self.docs_batch:
            self._flush_batch()

    def teardown(self):
        if self.docs_batch:
            self._flush_batch()
        self.client.close()

def run_pipeline(csv_path, mongo_uri, pipeline_options=None):
    """Run the Apache Beam pipeline for embedding extraction and MongoDB insertion."""
    headers = get_headers(csv_path)

    with beam.Pipeline(options=pipeline_options) as p:
        (
            p
            | "ReadCSV" >> beam.io.ReadFromText(csv_path, skip_header_lines=True)
            | "ParseCSV" >> beam.Map(lambda row: parse_row_to_dict(row, headers))
            # | "GenerateEmbeddings" >> beam.Map(generate_embeddings, image_model=image_model, text_model=text_model, tokenizer=text_tokenizer)
            | "GenerateEmbeddings" >> beam.ParDo(GenerateEmbeddingsDoFn())
            | "WriteToMongoDB" >> beam.ParDo(WriteToMongoDB(mongo_uri, DB_NAME, COLLECTION_NAME))
            # | beam.Map(print)
        )

if __name__ == "__main__":
    mongo_uri = get_env_variable("MONGO_URI")
    if not mongo_uri:
        logger.error("Environment variable 'mongo_uri' must be set.")
        exit(1)

    pipeline_options = PipelineOptions(
        runner=os.environ.get("RUNNER"),
        project=os.environ.get("PROJECT"),
        temp_location=get_env_variable("TEMP_LOCATION"),
        staging_location=get_env_variable("STAGING_LOCATION"),
        region=os.environ.get("REGION"),
        job_name=os.environ.get("JOB_NAME"),
        streaming=os.environ.get("STREAMING", False),
        requirements_file=os.environ.get("REQUIREMENTS_FILE"),
        # num_workers=5
    )

    run_pipeline(CSV_PATH, mongo_uri, pipeline_options)
