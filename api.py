from flask import Flask, request, jsonify
from text_extractor import text_extractor
import os
from create_chunks import create_chunks
from create_embeddings import create_embeddings
from create_faissIndex import create_faiss_index
import json

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        filename = file.filename

        if not filename:
            return jsonify({"error": "No file provided"}), 400

        temp_path = os.path.join('/tmp', filename)
        file.save(temp_path)

        data = text_extractor(temp_path)
        data = create_chunks(data)
        embeddings = create_embeddings(data)
        create_faiss_index(embeddings)
        with open("saved_chunks.json", "w") as f:
            json.dump(data, f)

        os.remove(temp_path)

        return jsonify({'text': data}), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
