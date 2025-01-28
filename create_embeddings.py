from sentence_transformers import SentenceTransformer
import numpy as np

def create_embeddings(texts: list[str]) -> np.ndarray:
    try:
        if not texts:
            raise ValueError("Input list cannot be empty")
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings
    except Exception as e:
        print(f"Error in create_embeddings: {e}")
        return np.array([])