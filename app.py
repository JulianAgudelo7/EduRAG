
import os
import uuid

from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

from backend.document_processor import extract_text_from_pdf
from backend.vector_database import add_document_pages

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


if __name__ == "__main__":
    app.run(debug=True)
