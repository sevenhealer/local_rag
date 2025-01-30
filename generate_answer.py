from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_answer(query: str, retrieved_chunks: list[str]) -> str:
    context = "\n".join(retrieved_chunks)
    prompt = (
        f"Answer the following question using the provided context:\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )

        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,
            num_beams=3,
            early_stopping=True,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

    except Exception as e:
        print(f"Generation error: {e}")
        return "[Error generating answer]"
