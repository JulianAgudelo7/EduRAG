
from backend.vector_database import search_documents

pregunta = input("Escribe una pregunta sobre tus documentos: ")

resultados = search_documents(pregunta, results_count=3)

documentos = resultados.get("documents", [[]])[0]
metadatos = resultados.get("metadatas", [[]])[0]
distancias = resultados.get("distances", [[]])[0]

if not documentos:
    print("\nNo se encontraron fragmentos.")
else:
    print("\n--- RESULTADOS DE LA BÚSQUEDA ---")

    for i, contenido in enumerate(documentos):
        print(f"\nResultado {i + 1}")
        print(f"Documento: {metadatos[i]['filename']}")
        print(f"Página: {metadatos[i]['page']}")
        print(f"Distancia: {distancias[i]:.4f}")
        print(f"Contenido:\n{contenido[:1000]}")
