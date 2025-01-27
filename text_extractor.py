import fitz
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
from PIL import Image

def text_extractor(file_path: str) -> str:
    data = ""
    file = fitz.open(file_path)
    
    for page_num in range(len(file)):
        page = file.load_page(page_num)
        data += page.get_text()
        
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(file, xref)
            
            if pix.n < 5:
                pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            else:
                pix = fitz.Pixmap(fitz.csRGB, pix)
                pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            
            img_text = pytesseract.image_to_string(pil_img)
            data += "\n" + img_text
            
            pix = None

    return data

print(text_extractor('test/test4.pdf'))
