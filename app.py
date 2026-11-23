
import os
import uuid

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

from backend.document_processor import extract_text_from_pdf
from backend.vector_database import add_document_pages
from backend.rag_service import generate_answer
from backend.answer_generator import generate_answer as generate_answer_with_sources

app = Flask(__name__)

app.secret_key = "edurag-clave-local"

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_document():

    if "document" not in request.files:
        flash("No se seleccionó ningún documento.")
        return redirect(url_for("inicio"))

    file = request.files["document"]

    if file.filename == "":
        flash("Debes seleccionar un archivo.")
        return redirect(url_for("inicio"))

    if not allowed_file(file.filename):
        flash("Solo se permiten documentos PDF.")
        return redirect(url_for("inicio"))

    filename = secure_filename(file.filename)

    if not filename:
        flash("El nombre del archivo no es válido.")
        return redirect(url_for("inicio"))

    # Identificador único para cada carga
    document_id = uuid.uuid4().hex

    # Evitar sobrescribir archivos con el mismo nombre
    stored_filename = f"{document_id}_{filename}"
    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        stored_filename
    )

    try:
        # Guardar el PDF
        file.save(file_path)

        # Extraer el texto
        pages = extract_text_from_pdf(file_path)

        if not pages:
            os.remove(file_path)
            flash("No se encontró texto extraíble en el PDF.")
            return redirect(url_for("inicio"))

        # Almacenar páginas y embeddings en ChromaDB
        indexed_pages = add_document_pages(
            filename=filename,
            pages=pages,
            document_id=document_id
        )

        if indexed_pages == 0:
            flash("No se encontraron páginas para indexar.")
            return redirect(url_for("inicio"))

        total_characters = sum(
            len(page["content"]) for page in pages
        )

        return render_template(
            "index.html",
            filename=filename,
            total_pages=len(pages),
            total_characters=total_characters,
            indexed_pages=indexed_pages,
            extracted_pages=pages
        )

    except Exception as error:
        app.logger.exception("Error al procesar el documento")
        flash(f"No se pudo procesar el documento: {error}")
        return redirect(url_for("inicio"))

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")

    if not isinstance(question, str) or not question.strip():
        return jsonify({
            "error": "Debes escribir una pregunta."
        }), 400

    try:
        answer = generate_answer(question.strip())

        return jsonify({
            "question": question.strip(),
            "answer": answer
        })

    except Exception:
        app.logger.exception("Error al generar la respuesta")
        return jsonify({
            "error": "No fue posible generar la respuesta. Intenta nuevamente."
        }), 500
@app.route("/chat", methods=["GET"])
def chat():
    return render_template("chat.html")


@app.route("/chat/ask", methods=["POST"])
def chat_ask():
    question = request.form.get("question", "").strip()

    if not question:
        return render_template(
            "chat.html",
            answer="Debes escribir una pregunta.",
            sources=[]
        ), 400

    try:
        result = generate_answer_with_sources(question)

        return render_template(
            "chat.html",
            question=question,
            answer=result["answer"],
            sources=result["sources"]
        )

    except Exception:
        app.logger.exception("Error al generar la respuesta")

        return render_template(
            "chat.html",
            question=question,
            answer="No fue posible generar la respuesta. Intenta nuevamente.",
            sources=[]
        ), 500

if __name__ == "__main__":
    app.run(debug=True)
