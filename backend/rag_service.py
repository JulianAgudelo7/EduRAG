
import os

from dotenv import load_dotenv
from groq import Groq

from backend.vector_database import search_documents

# Cargar variables de entorno
load_dotenv()

# Cliente de Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Modelo de lenguaje
MODEL_NAME = "openai/gpt-oss-20b"


def generate_answer(question):
    """
    Recupera información de ChromaDB y genera
    una respuesta contextualizada utilizando Groq.
    """

    if not question or not question.strip():
        return "Debes escribir una pregunta."

    # Recuperar los tres fragmentos más relevantes
    results = search_documents(question, results_count=3)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        return "No hay documentos disponibles para responder."

    # Construir el contexto con las fuentes
    context_parts = []

    for document, metadata in zip(documents, metadatas):
        context_parts.append(
            f"Archivo: {metadata['filename']}\n"
            f"Página: {metadata['page']}\n"
            f"Contenido: {document}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Instrucciones para el modelo
    system_prompt = (
        "Eres EduRAG, un asistente académico inteligente. "
        "Responde en español utilizando únicamente la información "
        "proporcionada en el contexto. "
        "Si el contexto no contiene información suficiente, "
        "indica que no puedes responder con los documentos disponibles. "
        "No inventes información. "
        "Cuando sea posible, menciona el archivo y la página "
        "de donde obtuviste la información."
    )

    # Enviar pregunta y contexto al LLM
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    f"Contexto recuperado:\n{context}\n\n"
                    f"Pregunta del estudiante:\n{question}"
                )
            }
        ],
        temperature=0.2,
        max_completion_tokens=700
    )

    return response.choices[0].message.content
