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

def process_image(gcs_image_path):
    """Load and preprocess an image from GCS path for DenseNet121."""
    try:
        img_data = tf.io.read_file(gcs_image_path)
        img = tf.image.decode_image(img_data, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [224, 224])
        img_array = tf.expand_dims(img, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        logger.error(f"Error loading image {gcs_image_path}: {e}")
        return None

def generate_image_embeddings(element, model):
    """Generate image embeddings using the provided model."""
    image_path = element.get('image_path')
    img_array = process_image(image_path)
    if img_array is not None:
        features = model.predict(img_array, verbose=0)
        element['image_features'] = features.flatten().tolist()
    else:
        element['image_features'] = None
    return element

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
            "image_features": element.get('image_features', []),
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
    model = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
    model.trainable = False

    with beam.Pipeline(options=pipeline_options) as p:
        (
            p
            | "ReadCSV" >> beam.io.ReadFromText(csv_path, skip_header_lines=True)
            | "ParseCSV" >> beam.Map(lambda row: parse_row_to_dict(row, headers))
            | "GenerateImageEmbeddings" >> beam.Map(generate_image_embeddings, model=model)
            | "WriteToMongoDB" >> beam.ParDo(WriteToMongoDB(mongo_uri, DB_NAME, COLLECTION_NAME))
            # | beam.Map(print)
        )

if __name__ == "__main__":
    mongo_uri = os.environ.get("MONGO_URI")
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
    )

    run_pipeline(CSV_PATH, mongo_uri, pipeline_options)
