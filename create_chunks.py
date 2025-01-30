from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(text: str, chunk_size=500, chunk_overlap=100) -> list[str]:
    try:
        if not text:
            raise ValueError("Text cannot be empty")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        return chunks
    except ValueError as ve:
        print(f"ValueError in create_chunks: {ve}")
        return []
    except Exception as e:
        print(f"Unexpected error in create_chunks: {e}")
        return []
