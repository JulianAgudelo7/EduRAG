import os
import uuid

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify
)
from werkzeug.utils import secure_filename

from backend.document_processor import extract_text_from_pdf
from backend.vector_database import add_document_pages
from backend.answer_generator import generate_answer

# CONFIGURACION DE FLASK

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "edurag-clave-local"
)

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

# PAGINA PRINCIPAL

@app.route("/")
def inicio():
    return render_template("index.html")

# CARGA Y PROCESAMIENTO DE PDF

@app.route("/upload", methods=["POST"])
def upload_document():

    # Debe coincidir con name="file" en index.html
    if "file" not in request.files:
        flash("No se seleccionó ningún documento.")
        return redirect(url_for("inicio"))

    file = request.files["file"]

    if not file.filename:
        flash("Debes seleccionar un archivo PDF.")
        return redirect(url_for("inicio"))

    if not allowed_file(file.filename):
        flash("Solo se permiten documentos PDF.")
        return redirect(url_for("inicio"))

    filename = secure_filename(file.filename)

    if not filename:
        flash("El nombre del archivo no es válido.")
        return redirect(url_for("inicio"))

    # Generar identificador único
    document_id = uuid.uuid4().hex

    # Evitar sobrescribir archivos
    stored_filename = f"{document_id}_{filename}"

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        stored_filename
    )

    try:
        # Guardar PDF
        file.save(file_path)

        # Extraer texto del documento
        pages = extract_text_from_pdf(file_path)

        if not pages:
            os.remove(file_path)
            flash("No se encontró texto extraíble en el PDF.")
            return redirect(url_for("inicio"))

        # Indexar documento en ChromaDB
        indexed_pages = add_document_pages(
            filename=filename,
            pages=pages,
            document_id=document_id
        )

        if indexed_pages == 0:
            flash("No se encontraron páginas para indexar.")
            return redirect(url_for("inicio"))

        # Calcular estadísticas
        total_characters = sum(
            len(page["content"]) for page in pages
        )

        flash(
            f"Documento '{filename}' procesado correctamente. "
            f"Se indexaron {indexed_pages} fragmentos."
        )

        return render_template(
            "index.html",
            filename=filename,
            total_pages=len(pages),
            total_characters=total_characters,
            indexed_pages=indexed_pages,
            extracted_pages=pages
        )

    except Exception:
        app.logger.exception(
            "Error al procesar el documento"
        )

        flash(
            "No se pudo procesar el documento. "
            "Revisa la terminal para conocer el error."
        )

        return redirect(url_for("inicio"))

# CHAT UNICO DE EDURAG

@app.route("/ask", methods=["POST"])
def ask_question():

    data = request.get_json(silent=True) or {}

    question = data.get("question", "")

    if not isinstance(question, str) or not question.strip():
        return jsonify({
            "error": "Debes escribir una pregunta."
        }), 400

    question = question.strip()

    try:
        # Generar respuesta con fuentes
        result = generate_answer(question)

        return jsonify({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"]
        })

    except Exception:
        app.logger.exception(
            "Error al generar la respuesta"
        )

        return jsonify({
            "error": (
                "No fue posible generar la respuesta. "
                "Intenta nuevamente."
            )
        }), 500

# EJECUTAR APLICACION

if __name__ == "__main__":
    app.run(debug=True)
