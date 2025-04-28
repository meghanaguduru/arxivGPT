from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import RAGChatbot

app = FastAPI()

# Initialize chatbot once
chatbot = RAGChatbot(
    pdf_dir = "../data",
    vector_db_path = "../data/processed/first_vector_db",
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.question
    answer = chatbot.query(question)
    return {"response": answer}