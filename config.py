import os
from dotenv import load_dotenv
load_dotenv()

def get_env_variable(var_name, default=""):
    return os.environ.get(var_name, default).replace("\\x3a", ":")

MONGO_URI = get_env_variable("MONGO_URI")
DB_NAME = get_env_variable("DB_NAME")
COLLECTION_NAME = get_env_variable("COLLECTION_NAME")
VECTOR_INDEX_NAME = "ct_vector_index"
