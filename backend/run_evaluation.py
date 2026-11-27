
from backend.evaluation import (
    evaluate_dataset,
    print_evaluation_report
)


TEST_CASES = [

    # ==========================================
    # 12 PREGUNTAS POSITIVAS
    # ==========================================

    {
        "type": "positive",
        "question": "¿Qué pasos se siguen para crear una API Key en Google AI Studio?",
        "filename": "Guia_LLms_IAR02.pdf",
        "pages": [1]
    },
    {
        "type": "positive",
        "question": "¿Cómo se guardan las claves de API de forma segura en Google Colab?",
        "filename": "Guia_LLms_IAR02.pdf",
        "pages": [3]
    },
    {
        "type": "positive",
        "question": "¿Qué valores de temperatura se recomiendan para tareas técnicas y escritura creativa?",
        "filename": "Guia_LLms_IAR02.pdf",
        "pages": [4]
    },
    {
        "type": "positive",
        "question": "¿Qué es Max Output Tokens y qué sucede cuando se alcanza su límite?",
        "filename": "Guia_LLms_IAR02.pdf",
        "pages": [4]
    },
    {
        "type": "positive",
        "question": "¿Qué controla el parámetro Top P y cuáles son sus valores típicos?",
        "filename": "Guia_LLms_IAR02.pdf",
        "pages": [4]
    },
    {
        "type": "positive",
        "question": "¿Cómo se configura Thinking en Gemini 2.5 y Gemini 3.x?",
        "filename": "Guia_LLms_IAR02.pdf",
        "pages": [4]
    },
    {
        "type": "positive",
        "question": "¿Qué diferencias existen entre Robot Preciso, Asistente Creativo y Analista Profundo?",
        "filename": "Guia_LLms_IAR02.pdf",
        "pages": [9]
    },
    {
        "type": "positive",
        "question": "¿Cuál es la diferencia entre Zero-shot, One-shot y Few-shot?",
        "filename": "Guia_LLms_IAR02.pdf",
        "pages": [9]
    },
    {
        "type": "positive",
        "question": "En la Tarea A, ¿cuántas peticiones llegan por minuto y cuánto tarda cada petición en procesarse?",
        "filename": "Guia_LLms_IAR02.pdf",
        "pages": [9]
    },
    {
        "type": "positive",
        "question": "¿Qué columnas debe contener la matriz de evaluación de los modelos?",
        "filename": "Guia_LLms_IAR02.pdf",
        "pages": [10]
    },
    {
        "type": "positive",
        "question": "¿Qué apartados presenta la guía para implementar un sistema RAG?",
        "filename": "Guia_LLms_IAR02.pdf",
        "pages": [11]
    },
    {
        "type": "positive",
        "question": "¿Cuál es el objetivo del Taller 2 sobre implementación de RAG?",
        "filename": "Guia_LLms_IAR02.pdf",
        "pages": [12]
    },

    # ==========================================
    # 3 PREGUNTAS NEGATIVAS
    # ==========================================

    {
        "type": "negative",
        "question": "¿Cómo se prepara un sándwich de pollo?"
    },
    {
        "type": "negative",
        "question": "¿Cuál es el precio actual de un portátil ASUS en Colombia?"
    },
    {
        "type": "negative",
        "question": "¿Quién ganó el último campeonato mundial de fútbol?"
    }
]


def main():

    print("\nIniciando evaluación ampliada de EduRAG...")

    report = evaluate_dataset(
        TEST_CASES,
        k=4
    )

    print_evaluation_report(report)


if __name__ == "__main__":
    main()
