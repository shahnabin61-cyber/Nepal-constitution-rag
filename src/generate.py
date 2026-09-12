from google import genai


def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    return f"""Answer the question using ONLY the context below, which is from the Constitution of Nepal (2015, as amended). If the context doesn't contain enough information to answer, say so clearly rather than guessing.

Context:
{context}

Question: {query}

Answer:"""


def generate_answer(query, retriever, gemini_api_key, k=5, model="gemini-3.6-flash"):
    client = genai.Client(api_key=gemini_api_key)
    retrieved_chunks = retriever.retrieve(query, k=k)
    prompt = build_prompt(query, retrieved_chunks)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text