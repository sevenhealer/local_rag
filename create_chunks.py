from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(text: str) -> list[str]:
    try:
        if not text:
            raise ValueError("Text cannot be empty")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        print("Chunks created:", chunks)
        return chunks
    except ValueError as ve:
        print(f"ValueError in create_chunks: {ve}")
        return []
    except Exception as e:
        print(f"Unexpected error in create_chunks: {e}")
        return []