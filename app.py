import os
import uuid
import json

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash
)

from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

from backend.document_processor import extract_text_from_pdf
from backend.vector_database import add_document_pages
from backend.answer_generator import generate_answer


# ==========================================
# CONFIGURACIÓN DE FLASK
# ==========================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "edurag-clave-local"
)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# Configuración de SQLite
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///chats.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# MODELOS DE BASE DE DATOS
# ==========================================

class Conversation(db.Model):

    id = db.Column(
        db.String(36),
        primary_key=True
    )

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

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    conversation_id = db.Column(
        db.String(36),
        db.ForeignKey("conversation.id"),
        nullable=False
    )

    role = db.Column(
        db.String(20),
        nullable=False
    )

    content = db.Column(
        db.Text,
        nullable=False
    )

    sources_json = db.Column(
        db.Text,
        nullable=False,
        default="[]"
    )


# Crear tablas si no existen
with app.app_context():
    db.create_all()


# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def get_conversation_history(conversation):

    return [
        {
            "role": message.role,
            "content": message.content
        }
        for message in conversation.messages
    ]


def save_message(conversation, role, content, sources=None):

    message = Message(
        conversation_id=conversation.id,
        role=role,
        content=content,
        sources_json=json.dumps(
            sources or [],
            ensure_ascii=False
        )
    )

    # Agregar el mensaje a la transacción.
    # El commit se realiza desde la ruta correspondiente.
    db.session.add(message)

    return message


def get_messages_with_sources(conversation):

    history = []

    for message in conversation.messages:

        try:

            sources = json.loads(
                message.sources_json or "[]"
            )

            if not isinstance(sources, list):
                sources = []

            sources = [
                source
                for source in sources
                if isinstance(source, dict)
                and source.get("filename")
                and source.get("page") is not None
            ]

        except (json.JSONDecodeError, TypeError):

            app.logger.warning(
                "Fuentes inválidas en el mensaje %s",
                message.id
            )

            sources = []

        history.append({
            "role": message.role,
            "content": message.content,
            "sources": sources
        })

    return history


def generate_conversation_title(question):

    title = question.strip()

    if len(title) > 45:
        title = title[:45].rstrip() + "..."

    return title or "Nueva conversación"


# ==========================================
# PÁGINA PRINCIPAL
# ==========================================

@app.route("/")
def inicio():

    return render_template("index.html")


# ==========================================
# CARGAR DOCUMENTOS PDF
# ==========================================

@app.route("/upload", methods=["POST"])
def upload_document():

    if "document" not in request.files:

        flash("No se seleccionó ningún documento.")

        return redirect(url_for("inicio"))

    file = request.files["document"]

    if not file.filename:

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

        # Guardar PDF
        file.save(file_path)

        # Extraer texto
        pages = extract_text_from_pdf(file_path)

        if not pages:

            os.remove(file_path)

            flash(
                "No se encontró texto extraíble en el PDF."
            )

            return redirect(url_for("inicio"))

        # Indexar documento en ChromaDB
        indexed_pages = add_document_pages(
            filename=filename,
            pages=pages,
            document_id=document_id
        )

        if indexed_pages == 0:

            os.remove(file_path)

            flash(
                "No se encontraron fragmentos para indexar."
            )

            return redirect(url_for("inicio"))

        # Calcular estadísticas
        total_characters = sum(
            len(page["content"])
            for page in pages
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
            "Error al procesar el documento %s",
            filename
        )

        flash(
            "No se pudo procesar el documento. "
            "Comprueba que sea un PDF válido e intenta nuevamente."
        )

        return redirect(url_for("inicio"))


# ==========================================
# LISTA DE CONVERSACIONES
# ==========================================

@app.route("/chat")
def chat():

    try:

        conversations = Conversation.query.order_by(
            Conversation.created_at.desc()
        ).all()

        # Crear una conversación si no existen
        if not conversations:

            return redirect(url_for("new_chat"))

        conversation_id = request.args.get("id")

        # Seleccionar la conversación más reciente
        if not conversation_id:

            conversation_id = conversations[0].id

        conversation = db.session.get(
            Conversation,
            conversation_id
        )

        # Seleccionar una conversación válida
        if conversation is None:

            return redirect(
                url_for("chat", id=conversations[0].id)
            )

        history = get_messages_with_sources(
            conversation
        )

        return render_template(
            "chat.html",
            history=history,
            conversations=conversations,
            current_conversation=conversation
        )

    except Exception:

        db.session.rollback()

        app.logger.exception(
            "Error al consultar las conversaciones"
        )

        flash(
            "No fue posible cargar las conversaciones. "
            "Intenta nuevamente."
        )

        return redirect(url_for("inicio"))


# ==========================================
# NUEVA CONVERSACIÓN
# ==========================================

@app.route("/new_chat")
def new_chat():

    conversation_id = str(uuid.uuid4())

    try:

        conversation = Conversation(
            id=conversation_id,
            title="Nueva conversación"
        )

        db.session.add(conversation)
        db.session.commit()

    except Exception:

        db.session.rollback()

        app.logger.exception(
            "Error al crear una conversación"
        )

        flash(
            "No fue posible crear una conversación. "
            "Intenta nuevamente."
        )

        return redirect(url_for("inicio"))

    return redirect(
        url_for("chat", id=conversation_id)
    )


# ==========================================
# ELIMINAR CONVERSACIÓN
# ==========================================

@app.route(
    "/delete_chat/<conversation_id>",
    methods=["POST"]
)
def delete_chat(conversation_id):

    try:

        conversation = db.session.get(
            Conversation,
            conversation_id
        )

        if conversation is None:

            return redirect(url_for("chat"))

        # Eliminar conversación y sus mensajes
        db.session.delete(conversation)
        db.session.commit()

    except Exception:

        db.session.rollback()

        app.logger.exception(
            "Error al eliminar la conversación %s",
            conversation_id
        )

        flash(
            "No fue posible eliminar la conversación. "
            "Intenta nuevamente."
        )

        return redirect(url_for("inicio"))

    return redirect(url_for("chat"))


# ==========================================
# ENVIAR PREGUNTAS
# ==========================================

@app.route("/ask", methods=["POST"])
def ask():

    question = request.form.get(
        "question",
        ""
    ).strip()

    conversation_id = request.form.get(
        "conversation_id"
    )

    # Validar datos recibidos
    if not question or not conversation_id:

        return redirect(
            url_for("chat", id=conversation_id)
        )

    # ======================================
    # CONSULTAR CONVERSACIÓN E HISTORIAL
    # ======================================

    try:

        conversation = db.session.get(
            Conversation,
            conversation_id
        )

        if conversation is None:

            return redirect(url_for("chat"))

        history = get_conversation_history(
            conversation
        )

        # Cerrar la transacción de lectura antes
        # de realizar la consulta externa a Groq.
        db.session.rollback()

    except Exception:

        db.session.rollback()

        app.logger.exception(
            "Error al consultar el historial"
        )

        flash(
            "No fue posible consultar la conversación."
        )

        return redirect(url_for("inicio"))

    # ======================================
    # GENERAR RESPUESTA
    # ======================================

    try:

        result = generate_answer(
            question,
            history=history
        )

        if not isinstance(result, dict):

            raise ValueError(
                "Formato de respuesta inválido"
            )

        answer = result.get("answer")

        sources = result.get("sources", [])

        if not isinstance(answer, str) or not answer.strip():

            raise ValueError(
                "El generador devolvió una respuesta vacía"
            )

        answer = answer.strip()

        if not isinstance(sources, list):

            sources = []

    except Exception:

        app.logger.exception(
            "Error al generar la respuesta"
        )

        answer = (
            "Ocurrió un error al generar la respuesta. "
            "Intenta nuevamente en unos instantes."
        )

        sources = []

    # ======================================
    # GUARDAR CONVERSACIÓN
    # ======================================

    try:

        # Consultar nuevamente la conversación
        # después de generar la respuesta.
        conversation = db.session.get(
            Conversation,
            conversation_id
        )

        if conversation is None:

            flash(
                "La conversación ya no está disponible."
            )

            return redirect(url_for("chat"))

        # Guardar pregunta
        save_message(
            conversation,
            "user",
            question
        )

        # Guardar respuesta y fuentes
        save_message(
            conversation,
            "assistant",
            answer,
            sources
        )

        # Asignar título automático
        if conversation.title == "Nueva conversación":

            conversation.title = generate_conversation_title(
                question
            )

        # Confirmar todos los cambios juntos
        db.session.commit()

    except Exception:

        # Revertir cualquier cambio incompleto
        db.session.rollback()

        app.logger.exception(
            "Error al guardar los mensajes de la conversación %s",
            conversation_id
        )

        flash(
            "No fue posible guardar los mensajes. "
            "Intenta enviar tu pregunta nuevamente."
        )

        return redirect(url_for("inicio"))

    return redirect(
        url_for("chat", id=conversation_id)
    )


# ==========================================
# EJECUCIÓN
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)
