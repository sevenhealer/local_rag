import faiss
import numpy as np

def create_faiss_index(embeddings: np.ndarray):
    try:
        if embeddings.size == 0:
            raise ValueError("Embeddings array is empty")
        
        embeddings = embeddings.astype('float32')
        dimension = embeddings.shape[1]
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
