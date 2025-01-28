import fitz
from PIL import Image
from img_extractor import img_extractor

def pdf_extractor(file_path: str, password: str = None) -> str:
    data = ""
    file = fitz.open(file_path)
    if file.needs_pass:
        if not password:
            raise ValueError("Password Required")
        file.authenticate(password)
    
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
            
            img_text = img_extractor(pil_img)
            data += "\n" + img_text
            
            pix = None

    return data

# print(pdf_extractor('test/test4_protected.pdf', 'password'))