from docx import Document

def docx_extractor(file_path: str) -> str:
    text= ""
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        raise ValueError(f"Error Parsing DOCX: {str(e)}")

# docx_extractor('test/demo.docx')