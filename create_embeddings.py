from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(texts: list[str]) -> np.ndarray:
    try:
        if not texts:
            raise ValueError("Input list cannot be empty")
        
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings
    except Exception as e:
        print(f"Error in create_embeddings: {e}")
        return np.array([])
