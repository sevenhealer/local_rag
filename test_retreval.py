import faiss
import json
import numpy as np
from create_embeddings import create_embeddings

def retrieve_chunks(query: str, top_k: int = 3) -> list[str]:
    try:
        with open("saved_chunks.json", "r") as f:
            chunks = json.load(f)
        
        index = faiss.read_index("faiss_index.index")
        
        query_embedding = create_embeddings([query])
        
        if query_embedding.size == 0:
            return [] 
        
        query_embedding = query_embedding.astype('float32')
        
        if len(query_embedding.shape) == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        
        distances, indices = index.search(query_embedding, top_k)
        
        return [chunks[i] for i in indices[0]]
    
    except Exception as e:
        print(f"Error in retrieve_chunks: {e}")
        return []

print(retrieve_chunks("AWS"))