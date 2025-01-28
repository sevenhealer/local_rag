def txt_extractor(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='UTF-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()
        
txt_extractor('test/demo.txt')