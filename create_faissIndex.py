import faiss
import numpy as np

def create_faiss_index(embeddings: list[float]):
    try:
        if not embeddings.any():
            raise ValueError("Embeddings array is empty")
        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]
        print(dimension)
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, "faiss_index.index")
        print("FAISS index saved!")
    except ValueError as ve:
        print(f"ValueError in create_faiss_index: {ve}")
    except faiss.FaissException as fe:
        print(f"FAISS-specific error in create_faiss_index: {fe}")
    except Exception as e:
        print(f"Unexpected error in create_faiss_index: {e}")