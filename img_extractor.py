# from ocr_preprocess import preprocess_image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def img_extractor(file_path: str) -> str:
    try:
        return pytesseract.image_to_string(file_path)
        # processed_img = preprocess_image(file_path)
        # return pytesseract.image_to_string(processed_img, config="--oem 1 --psm 3")
    except Exception as e:
        raise ValueError(f"Error Extracting IMG: {str(e)}")