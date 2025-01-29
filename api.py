from flask import Flask, request, jsonify
from text_extractor import text_extractor
import os
from create_chunks import create_chunks
from create_embeddings import create_embeddings
from create_faissIndex import create_faiss_index
import json
from test_retreval import retrieve_chunks
from generate_answer import generate_answer
import logging

logging.basicConfig(level=logging.INFO, filename="rag_pipeline")


app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        filename = file.filename

        if not filename:
            return jsonify({"error": "No file provided"}), 400

        temp_path = os.path.join(os.getcwd(), filename)
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
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/query', methods=['POST'])
def query():
    body = request.get_json()
    query = body['query']
    response_chunks = retrieve_chunks(query)
    if not response_chunks:
        return jsonify({"error": "No relevant chunks found for the query"}), 404

    answer = generate_answer(query, response_chunks)
    return jsonify({"query": query, "chunks": response_chunks, "answer": answer}), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
