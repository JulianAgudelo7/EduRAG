
import chromadb
from sentence_transformers import SentenceTransformer

# Modelo multilingüe: permite trabajar con español
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Cargar modelo de embeddings
embedding_model = SentenceTransformer(MODEL_NAME)

# Base de datos vectorial persistente
client = chromadb.PersistentClient(path="./chroma_db")

# Colección de documentos de EduRAG
collection = client.get_or_create_collection(
    name="edurag_documents",
    metadata={"hnsw:space": "cosine"}
)


def add_document_pages(filename, pages, document_id):
    """
    Convierte las páginas de un PDF en embeddings
    y las almacena en ChromaDB.
    """

    ids = []
    documents = []
    metadatas = []

    for page in pages:
        content = page["content"].strip()

        if not content:
            continue

        ids.append(f"{document_id}_pagina_{page['page']}")
        documents.append(content)

        metadatas.append({
            "filename": filename,
            "document_id": document_id,
            "page": page["page"]
        })

    if not documents:
        return 0

    # Generar embeddings y convertirlos a listas
    embeddings = embedding_model.encode(
        documents,
        convert_to_numpy=True
    ).tolist()

    # Guardar o actualizar los registros
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return len(documents)


def search_documents(question, results_count=3):
    """
    Busca las páginas más relacionadas con una pregunta.
    """

    question_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(results_count, collection.count())
    )

    return results
