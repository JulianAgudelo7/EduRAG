
import chromadb
from sentence_transformers import SentenceTransformer

# Modelo multilingüe para generar embeddings
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

embedding_model = SentenceTransformer(MODEL_NAME)

# Base de datos vectorial persistente
client = chromadb.PersistentClient(path="./chroma_db")

# Nueva colección para almacenar fragmentos
collection = client.get_or_create_collection(
    name="edurag_chunks",
    metadata={"hnsw:space": "cosine"}
)


def split_text(text, chunk_size=800, chunk_overlap=150):
    """
    Divide un texto en fragmentos con solapamiento.
    """

    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Los parámetros de fragmentación no son válidos.")

    text = text.strip()
    chunks = []

    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - chunk_overlap

    return chunks


def add_document_pages(filename, pages, document_id):
    """
    Divide las páginas de un PDF en fragmentos,
    genera embeddings y los almacena en ChromaDB.
    """

    ids = []
    documents = []
    metadatas = []

    for page in pages:
        page_chunks = split_text(page["content"])

        for chunk_number, chunk in enumerate(page_chunks, start=1):
            ids.append(
                f"{document_id}_pagina_{page['page']}_chunk_{chunk_number}"
            )

            documents.append(chunk)

            metadatas.append({
                "filename": filename,
                "document_id": document_id,
                "page": page["page"],
                "chunk": chunk_number
            })

    if not documents:
        return 0

    embeddings = embedding_model.encode(
        documents,
        convert_to_numpy=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return len(documents)


def search_documents(question, results_count=3):
    """
    Busca los fragmentos más relacionados con una pregunta.
    """

    total_documents = collection.count()

    if total_documents == 0:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    question_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(results_count, total_documents),
        include=["documents", "metadatas", "distances"]
    )

    return results