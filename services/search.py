from config import VECTOR_INDEX_NAME

def vector_search(collection, query_embedding, top_k=9, num_candidates=100):
    cursor = collection.aggregate([
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "combined_features",
                "queryVector": [float(x) for x in query_embedding],
                "numCandidates": num_candidates,
                "limit": top_k,
            }
        },
        {
            "$project": {
                "_id": 1, "image_path": 1, "caption_x": 1,
                "finding": 1, "case_id": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ])
    return list(cursor)
