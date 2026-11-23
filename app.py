
import os
import uuid

from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

from backend.document_processor import extract_text_from_pdf
from backend.vector_database import add_document_pages
from backend.answer_generator import generate_answer


app = Flask(__name__)

app.secret_key = "edurag-clave-local"

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# Configuración de SQLite
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///chats.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# MODELOS DE BASE DE DATOS
# =========================

class Conversation(db.Model):

    id = db.Column(db.String(36), primary_key=True)

    title = db.Column(
        db.String(200),
        nullable=False,
        default="Nueva conversación"
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=db.func.now()
    )

    messages = db.relationship(
        "Message",
        backref="conversation",
        cascade="all, delete-orphan",
        order_by="Message.id"
    )


class Message(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    conversation_id = db.Column(
        db.String(36),
        db.ForeignKey("conversation.id"),
        nullable=False
    )

    role = db.Column(db.String(20), nullable=False)

    content = db.Column(db.Text, nullable=False)

    sources_json = db.Column(
        db.Text,
        nullable=False,
        default="[]"
    )


# Crear tablas automáticamente
with app.app_context():
    db.create_all()


# =========================
# FUNCIONES AUXILIARES
# =========================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_conversation(conversation_id):

    return Conversation.query.get_or_404(conversation_id)


def get_conversation_history(conversation):

    history = []

    for message in conversation.messages:

        history.append({
            "role": message.role,
            "content": message.content
        })

    return history


def save_message(conversation, role, content, sources=None):

    import json

    message = Message(
        conversation_id=conversation.id,
        role=role,
        content=content,
        sources_json=json.dumps(sources or [], ensure_ascii=False)
    )

    db.session.add(message)
    db.session.commit()

    return message


def get_messages_with_sources(conversation):

    import json

    history = []

    for message in conversation.messages:

        history.append({
            "role": message.role,
            "content": message.content,
            "sources": json.loads(message.sources_json or "[]")
        })

    return history


def generate_conversation_title(question):

    title = question.strip()

    if len(title) > 45:

        title = title[:45].rstrip() + "..."

    return title or "Nueva conversación"


# =========================
# INICIO
# =========================

@app.route("/")
def inicio():

    return render_template("index.html")


# =========================
# CARGAR DOCUMENTOS
# =========================

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

    document_id = uuid.uuid4().hex

    stored_filename = f"{document_id}_{filename}"

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        stored_filename
    )

    try:

        file.save(file_path)

        pages = extract_text_from_pdf(file_path)

        if not pages:

            os.remove(file_path)

            flash("No se encontró texto extraíble en el PDF.")

            return redirect(url_for("inicio"))

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


# =========================
# LISTA DE CONVERSACIONES
# =========================

@app.route("/chat")
def chat():

    conversations = Conversation.query.order_by(
        Conversation.created_at.desc()
    ).all()

    # Si no existe ninguna conversación, crear una
    if not conversations:

        return redirect(url_for("new_chat"))

    conversation_id = request.args.get("id")

    # Si no se especifica, seleccionar la más reciente
    if not conversation_id:

        conversation_id = conversations[0].id

    conversation = Conversation.query.get(conversation_id)

    # Si el ID no existe, regresar a la más reciente
    if conversation is None:

        return redirect(
            url_for("chat", id=conversations[0].id)
        )

    history = get_messages_with_sources(conversation)

    return render_template(
        "chat.html",
        history=history,
        conversations=conversations,
        current_conversation=conversation
    )


# =========================
# NUEVA CONVERSACIÓN
# =========================

@app.route("/new_chat")
def new_chat():

    conversation = Conversation(
        id=str(uuid.uuid4()),
        title="Nueva conversación"
    )

    db.session.add(conversation)
    db.session.commit()

    return redirect(
        url_for("chat", id=conversation.id)
    )


# =========================
# ELIMINAR CONVERSACIÓN
# =========================

@app.route("/delete_chat/<conversation_id>", methods=["POST"])
def delete_chat(conversation_id):

    conversation = Conversation.query.get_or_404(conversation_id)

    db.session.delete(conversation)
    db.session.commit()

    remaining = Conversation.query.order_by(
        Conversation.created_at.desc()
    ).first()

    if remaining:

        return redirect(
            url_for("chat", id=remaining.id)
        )

    return redirect(url_for("new_chat"))


# =========================
# ENVIAR PREGUNTAS
# =========================

@app.route("/ask", methods=["POST"])
def ask():

    question = request.form.get("question", "").strip()

    conversation_id = request.form.get("conversation_id")

    if not question or not conversation_id:

        return redirect(url_for("chat"))

    conversation = Conversation.query.get_or_404(conversation_id)

    history = get_conversation_history(conversation)

    try:

        # Generar respuesta usando historial previo
        result = generate_answer(
            question,
            history=history
        )

        answer = result["answer"]

        sources = result["sources"]

        # Guardar pregunta
        save_message(
            conversation,
            "user",
            question
        )

        # Guardar respuesta
        save_message(
            conversation,
            "assistant",
            answer,
            sources
        )

        # Asignar título a la primera pregunta
        if conversation.title == "Nueva conversación":

            conversation.title = generate_conversation_title(question)

            db.session.commit()

        return redirect(
            url_for("chat", id=conversation.id)
        )

    except Exception:

        app.logger.exception("Error al generar respuesta")

        save_message(
            conversation,
            "user",
            question
        )

        save_message(
            conversation,
            "assistant",
            "Ocurrió un error al generar la respuesta. "
            "Revisa la terminal para conocer el detalle."
        )

        return redirect(
            url_for("chat", id=conversation.id)
        )


# =========================
# EJECUCIÓN
# =========================

if __name__ == "__main__":

    app.run(debug=True)
