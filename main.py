from langchain.vectorstores import VectorStore

from pathlib import Path
import subprocess
import tempfile
import pickle
from typing import Optional

import requests
import logging

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from chain_builder import get_chat_chain, get_qa_chain
from schemas import ChatResponse
from ingest import ingest_docs

from pydantic import BaseModel

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
vectorstore: Optional[VectorStore] = None

# Set CORS middleware to allow requests from the frontend
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

# Build the vector space at startup if it doesn't exist
@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    if not Path("vectorstore.pkl").exists():
        ingest_docs("meilisearch", "documentation")
    with open("vectorstore.pkl", "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)

@app.get("/chat")
def chat():
    return "Open a websocket connection on /chat to use the chatbot chain."

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    chat_chain = get_chat_chain(vectorstore, question_handler, stream_handler)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await chat_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Uh oh, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())

@app.get("/qa")
def qa(question: str):
    qa_chain = get_qa_chain(vectorstore)

    try:
        return qa_chain({"question": question}, return_only_outputs=True)
    except Exception as e:
        logging.error(e)
        return e