
from backend.vector_database import (
    collection,
    embedding_model,
    TOP_K,
    RELEVANCE_THRESHOLD
)


def retrieve(question, k=TOP_K):

    total_chunks = collection.count()

    if total_chunks == 0:
        raise ValueError(
            "No hay fragmentos almacenados en ChromaDB."
        )

    embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).tolist()

    return collection.query(
        query_embeddings=embedding,
        n_results=min(k, total_chunks),
        include=["metadatas", "distances"]
    )


def evaluate_positive(test, k=TOP_K):

    results = retrieve(test["question"], k)

    expected_filename = test["filename"].casefold()
    expected_pages = set(test["pages"])

    ranked_results = []
    first_relevant_rank = None

    for index, metadata in enumerate(results["metadatas"][0]):

        filename = metadata.get("filename", "")
        page = metadata.get("page")
        distance = results["distances"][0][index]

        relevant = (
            filename.casefold() == expected_filename
            and page in expected_pages
        )

        rank = index + 1

        if relevant and first_relevant_rank is None:
            first_relevant_rank = rank

        ranked_results.append({
            "rank": rank,
            "filename": filename,
            "page": page,
            "distance": distance,
            "relevant": relevant
        })

    hit = int(first_relevant_rank is not None)

    reciprocal_rank = (
        1 / first_relevant_rank
        if first_relevant_rank is not None
        else 0
    )

    return {
        "question": test["question"],
        "hit": hit,
        "reciprocal_rank": reciprocal_rank,
        "first_relevant_rank": first_relevant_rank,
        "results": ranked_results
    }


def evaluate_negative(test, k=TOP_K):

    results = retrieve(test["question"], k)

    distances = results["distances"][0]

    closest_distance = min(distances) if distances else None

    rejected = (
        closest_distance is None
        or closest_distance > RELEVANCE_THRESHOLD
    )

    return {
        "question": test["question"],
        "rejected": int(rejected),
        "closest_distance": closest_distance
    }


def evaluate_dataset(test_cases, k=TOP_K):

    if not test_cases:
        raise ValueError("No hay preguntas de evaluación.")

    positive_cases = [
        test for test in test_cases
        if test["type"] == "positive"
    ]

    negative_cases = [
        test for test in test_cases
        if test["type"] == "negative"
    ]

    if not positive_cases or not negative_cases:
        raise ValueError(
            "Debe haber preguntas positivas y negativas."
        )

    positive_results = [
        evaluate_positive(test, k)
        for test in positive_cases
    ]

    negative_results = [
        evaluate_negative(test, k)
        for test in negative_cases
    ]

    hit_average = sum(
        item["hit"] for item in positive_results
    ) / len(positive_results)

    mrr_average = sum(
        item["reciprocal_rank"]
        for item in positive_results
    ) / len(positive_results)

    rejection_average = sum(
        item["rejected"]
        for item in negative_results
    ) / len(negative_results)

    return {
        "k": k,
        "threshold": RELEVANCE_THRESHOLD,
        "positive_count": len(positive_results),
        "negative_count": len(negative_results),
        "hit_at_k": hit_average,
        "mrr_at_k": mrr_average,
        "rejection_accuracy": rejection_average,
        "positive_results": positive_results,
        "negative_results": negative_results
    }


def print_evaluation_report(report):

    print("\n" + "=" * 60)
    print("             EVALUACIÓN DE EDURAG")
    print("=" * 60)

    print("\nRECUPERACIÓN DE INFORMACIÓN")
    print(f"Preguntas positivas: {report['positive_count']}")
    print(f"Hit@{report['k']}: {report['hit_at_k']:.2%}")
    print(f"MRR@{report['k']}: {report['mrr_at_k']:.4f}")

    for item in report["positive_results"]:

        print(f"\nPregunta: {item['question']}")
        print(f"Hit: {item['hit']}")
        print(f"MRR: {item['reciprocal_rank']:.4f}")

        if item["first_relevant_rank"] is not None:
            print(
                "Primer resultado relevante: posición "
                f"{item['first_relevant_rank']}"
            )
        else:
            print("No se encontró una página relevante.")

        for result in item["results"]:
            print(
                f"  {result['rank']}. "
                f"{result['filename']} | "
                f"Página {result['page']} | "
                f"Distancia: {result['distance']:.4f} | "
                f"Relevante: {result['relevant']}"
            )

    print("\n" + "-" * 60)
    print("RECHAZO DE PREGUNTAS FUERA DEL DOCUMENTO")

    print(f"Preguntas negativas: {report['negative_count']}")
    print(
        f"Rechazo correcto: "
        f"{report['rejection_accuracy']:.2%}"
    )
    print(f"Umbral: {report['threshold']}")

    for item in report["negative_results"]:

        distance = item["closest_distance"]

        status = (
            "RECHAZADA CORRECTAMENTE"
            if item["rejected"]
            else "RECUPERÓ INFORMACIÓN"
        )

        print(f"\nPregunta: {item['question']}")

        if distance is not None:
            print(f"Distancia más cercana: {distance:.4f}")
        else:
            print("No hay resultados recuperados.")

        print(f"Resultado: {status}")

    print("\n" + "=" * 60)
