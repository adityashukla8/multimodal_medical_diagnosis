from pymongo import MongoClient
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]
