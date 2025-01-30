import os
import json
import logging
import gradio as gr
from flask import Flask, request, jsonify
from text_extractor import text_extractor
from create_chunks import create_chunks
from create_embeddings import create_embeddings
from create_faissIndex import create_faiss_index
from test_retreval import retrieve_chunks
from generate_answer import generate_answer
import threading

logging.basicConfig(level=logging.INFO, filename="rag_pipeline.log")

app = Flask(__name__)

index = None

def load_faiss_index():
    global index
    try:
        index = faiss.read_index("faiss_index.index")
    except Exception as e:
        logging.error(f"Error loading FAISS index: {e}")
        index = None

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
        chunks = create_chunks(data)
        embeddings = create_embeddings(chunks)
        create_faiss_index(embeddings)

        with open("saved_chunks.json", "w") as f:
            json.dump(chunks, f)

        os.remove(temp_path)

        return jsonify({'text': chunks}), 200

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


def start_flask_app():
    app.run(host='0.0.0.0', port=5000)

def process_file(file, query):
    file_path = file.name
    data = text_extractor(file_path)
    chunks = create_chunks(data)
    embeddings = create_embeddings(chunks)
    create_faiss_index(embeddings)
    answer = generate_answer(query, chunks)
    return chunks, answer


def gradio_interface():
    iface = gr.Interface(
        fn=process_file,
        inputs=[
            gr.File(label="Upload PDF/DOCX/Text File"),
            gr.Textbox(label="Ask a Question", placeholder="Enter your question here")
        ],
        outputs=[
            gr.Textbox(label="Chunks"),
            gr.Textbox(label="Answer")
        ],
        title="RAG Pipeline",
        description="Upload a document (PDF, DOCX, or Text) and ask a question."
    )
    iface.launch(share=True, server_name="0.0.0.0", server_port=7860)

import threading

shutdown_event = threading.Event()

def start_flask_app():
    app.run(host='0.0.0.0', port=5000, use_reloader=False)

def stop_flask_app():
    shutdown_event.set()

def gradio_interface():
    iface = gr.Interface(
        fn=process_file,
        inputs=[gr.File(label="Upload PDF/DOCX/Text File"), gr.Textbox(label="Ask a Question")],
        outputs=[gr.Textbox(label="Chunks"), gr.Textbox(label="Answer")]
    )
    iface.launch(share=True, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=start_flask_app)
    flask_thread.start()

    gradio_interface()
