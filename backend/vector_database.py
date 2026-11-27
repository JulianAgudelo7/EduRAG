
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURACIÓN
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "chroma_db"
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_WORDS = 200
CHUNK_OVERLAP = 40
MIN_WORDS = 15
TOP_K = 4
RELEVANCE_THRESHOLD = 0.75

# ==========================================
# MODELO DE EMBEDDINGS
# ==========================================

embedding_model = SentenceTransformer(MODEL_NAME)

# ==========================================
# CHROMADB PERSISTENTE
# ==========================================

client = chromadb.PersistentClient(path=str(CHROMA_PATH))

collection = client.get_or_create_collection(
    name="edurag_chunks",
    metadata={"hnsw:space": "cosine"}
)

# ==========================================
# FRAGMENTACIÓN POR PALABRAS
# ==========================================

def split_text(text, chunk_size=CHUNK_WORDS,
               chunk_overlap=CHUNK_OVERLAP):

    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Los parámetros de fragmentación no son válidos.")

    words = text.split()
    chunks = []

    step = chunk_size - chunk_overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if len(chunk_words) >= MIN_WORDS:
            chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


# ==========================================
# INDEXACIÓN DE DOCUMENTOS
# ==========================================

def add_document_pages(filename, pages, document_id):
    """
    Fragmenta las páginas de un PDF, genera embeddings
    y almacena los fragmentos en ChromaDB.
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
        normalize_embeddings=True,
        convert_to_numpy=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return len(documents)


# ==========================================
# BÚSQUEDA SEMÁNTICA
# ==========================================

def search_documents(question, results_count=TOP_K):
    """
    Recupera los fragmentos más relacionados con la pregunta.
    Descarta los que superan la distancia máxima permitida.
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
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(results_count, total_documents),
        include=["documents", "metadatas", "distances"]
    )

    # Aplicar el umbral de relevancia
    filtered = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]]
    }

    for i, distance in enumerate(results["distances"][0]):

        if distance <= RELEVANCE_THRESHOLD:
            filtered["ids"][0].append(results["ids"][0][i])
            filtered["documents"][0].append(results["documents"][0][i])
            filtered["metadatas"][0].append(results["metadatas"][0][i])
            filtered["distances"][0].append(distance)

    return filtered


# ==========================================
# INFORMACIÓN DE LA BASE VECTORIAL
# ==========================================

def get_collection_stats():
    return {
        "total_chunks": collection.count(),
        "database_path": str(CHROMA_PATH),
        "embedding_model": MODEL_NAME,
        "chunk_words": CHUNK_WORDS,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "relevance_threshold": RELEVANCE_THRESHOLD
    }
