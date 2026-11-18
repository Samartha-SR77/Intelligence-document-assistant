import ollama
def generate_answer(question,context):
    prompt=f"""
    context:{context}
    question:{question}
    Answer using only the provided context.

    If the answer cannot be found in the context,
    return exactly: NULL

    Do not explain, elaborate, or add any other text
    when returning NULL.

    Do not guess or use outside knowledge.
    Maintain a friendly, professional tone.
    dont give whole context as answer just answer the questions briefly
    """
  
    response=ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )
    answer=response["message"]["content"]
    return answer
