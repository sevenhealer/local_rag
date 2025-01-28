from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=True)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")


def generate_answer(query, retrieved_chunks):
    context = " ".join(retrieved_chunks)

    prompt = f"Answer the question based on the following context: {context} Question: {query}"
    # prompt = f"question: {query} context: {context}"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=4, early_stopping=True)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
