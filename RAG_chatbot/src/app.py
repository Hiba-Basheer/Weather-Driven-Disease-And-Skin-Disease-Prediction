"""
FastAPI interface for the RAG Chatbot.
Supports structured ML input, unstructured DL queries, skin image classification, and image + note hybrid input.
"""

import os
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.rag_pipeline import (
    detect_module,
    handle_ml_input,
    handle_dl_input,
    handle_skin_image_input,
    handle_skin_image_with_text,
    save_user_data,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/")
def read_root() -> dict:
    """Root endpoint for health check."""
    return {"message": "Welcome to the RAG Chatbot API"}

@app.get("/chat-ui", response_class=HTMLResponse)
def chat_ui(request: Request) -> HTMLResponse:
    """Serve the chatbot UI."""
    return templates.TemplateResponse("chat.html", {"request": request})

class TextQuery(BaseModel):
    query: str

@app.post("/chat/")
def chat_endpoint(query: TextQuery) -> dict:
    """
    Handle text-based queries and route to appropriate module.

    Parameters:
        query (TextQuery): User input string.

    Returns:
        dict: Module used and chatbot response.
    """
    module, normalized_input = detect_module(query.query)

    if module == "ml":
        response = handle_ml_input(normalized_input)
    elif module == "dl":
        response = handle_dl_input(normalized_input)
    elif module == "skin_image":
        response = handle_skin_image_input(normalized_input)
    else:
        response = "Could not route your input."

    save_user_data(module, query.query, response)
    return {"module": module, "response": response}

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)) -> dict:
    """
    Handle image-only upload and run skin disease prediction.

    Parameters:
        file (UploadFile): Uploaded image file.

    Returns:
        dict: Prediction response.
    """
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    response = handle_skin_image_input(file_path)
    save_user_data("skin_image", file.filename, response)
    return {"response": response}

@app.post("/upload-image-with-note/")
async def upload_image_with_note(
    file: UploadFile = File(...),
    note: Optional[str] = Form(None)
) -> dict:
    """
    Handle image + optional user note input for hybrid prediction.

    Parameters:
        file (UploadFile): Uploaded image file.
        note (Optional[str]): User-provided description.

    Returns:
        dict: Combined prediction and explanation.
    """
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    if note:
        response = handle_skin_image_with_text(file_path, note)
    else:
        response = handle_skin_image_input(file_path)

    save_user_data("skin_image_with_text", f"{file.filename} + note", response)
    return {"response": response}