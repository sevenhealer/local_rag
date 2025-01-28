from txt_extractor import txt_extractor
from docx_extractor import docx_extractor
from img_extractor import img_extractor
from pdf_extractor import pdf_extractor

def text_extractor(file_path: str, password: str = None) -> str:
    if file_path.endswith('.txt'):
        return txt_extractor(file_path)
    elif file_path.endswith('.docx'):
        return docx_extractor(file_path)
    elif file_path.endswith(('.png', '.jpg', '.jpeg')):
        return img_extractor(file_path)
    elif file_path.endswith('.pdf'):
        return pdf_extractor(file_path, password)
    else:
        return ValueError(f"Unsupported File Format.")