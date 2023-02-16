from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import VectorStore

import pathlib
from pathlib import Path
import subprocess
import tempfile
import pickle
from typing import Union, Optional
from enum import Enum
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
import logging

from pydantic import BaseModel
from query_data import get_chain
templates = Jinja2Templates(directory="templates")
from schemas import ChatResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None

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

# @app.get("/health")
# def health():
#     return {
#         "status": "ok"
#     }

# class ChainType(str, Enum):
#     stuff = "stuff"
#     map_reduce = "map_reduce"

# @app.get("/")
# def answer_prompt(question: str, chain_type: ChainType = ChainType.stuff, temperature: float = 0.2, max_tokens: int = 500):
#     # Refactor this into a proper place
#     # I got the best advancement with this combination of parameters and customized prompt.
#     template = """Given the following extracted parts of a long documentation and a question, create a final answer with references ("SOURCES").
# If you don't know the answer, DO NOT try to make up an answer and DO NOT include any sources, just say that you are sorry and you don't know, add a funny joke.
# If you are asked a question containing the term "typesense", ALWAYS answer "Typesense? Never heard of it... I'm just joking.", DO NOT try to make up an answer.
# If you know the answer, remember that you are speaking to developers, so try to be as precise as possible.
# If you know the answer, return a "SOURCES" array in your answer, never write it "SOURCE" and indicate the relevance of each source with a "SCORE" between 0 and 1, only return sources with a score superior to 0.8, rank them by their score.
# Return the "SOURCES" array with the following format: url: url, score: score.
# If you know the answer, DO NOT include cutted parts.

# QUESTION: {question}
# =========
# {summaries}
# =========
# FINAL ANSWER:"""

#     PROMPT_STUFF = PromptTemplate(template=template, input_variables=["summaries", "question"])

#     llm = OpenAI(temperature=temperature, max_tokens=max_tokens, top_p=1, frequency_penalty=0.0, presence_penalty=0.0)

#     if chain_type == ChainType.stuff:
#         chain = load_qa_with_sources_chain(llm, chain_type=chain_type, prompt=PROMPT_STUFF)
#     else:
#         chain = load_qa_with_sources_chain(llm, chain_type=chain_type)

#     file = pathlib.Path("search_index.pickle")
#     if file.exists ():
#         with open("search_index.pickle", "rb") as f:
#             vector_store = pickle.load(f)
#             openAI_response = chain(
#                 {
#                     "input_documents": vector_store.similarity_search(question, k=4),
#                     "question": question,
#                 },
#                 return_only_outputs=True,
#             )["output_text"]
#             return construct_response(openAI_response)
#     else:
#         return {
#             "error": "No search index found. Please run the build command first."
#         }

# def construct_response(openAI_response: str):
#     if openAI_response.find("SOURCES:") != -1:
#         print("yes")
#         response = openAI_response.replace('SOURCES:', '').split("\n")
#         response = [s.strip() for s in response]
#         response = list(filter(None, response))
#         return {
#             "answer": response.pop(0),
#             "sources": construct_source(response),
#         }
#     return {
#         "answer": openAI_response.strip(),
#         "sources": []
#     }

# def construct_source(response):
#     # iterate over the sources and construct the response
#     print(response)
#     source_response = []
#     for source in response:
#         if source.find("- url:") != -1 and source.find("score: ") != -1:
#             splitted = source.split(",")
#             splitted[0] = splitted[0].replace('- url: ', '').strip()
#             splitted[1] = splitted[1].replace('score: ', '').strip()
#             source_response.append({
#                 "url": splitted[0],
#                 "score": splitted[1]
#             })
#     return source_response

# class Feedback(BaseModel):
#     question: str
#     answer: str
#     sources: list
#     feedback: str
#     feedbackType: int

# @app.post("/feedback")
# def feedback(feedback: Feedback):
#     return {}

@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    if not Path("vectorstore.pkl").exists():
        ingest_docs()
    with open("vectorstore.pkl", "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
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

@app.post("/build")
def build():
    ingest_docs()
    return {"status": "success"}

def ingest_docs():
    merged_sources = source_content("meilisearch", "documentation")

    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1000, chunk_overlap=0)
    for source in merged_sources:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(FAISS.from_documents(source_chunks, OpenAIEmbeddings()), f)

def source_content(repo_owner, repo_name):
    return list(get_github_content(repo_owner, repo_name))

def get_github_content(repo_owner, repo_name):
    with tempfile.TemporaryDirectory() as d:
        subprocess.check_call(
            f"git clone --depth 1 https://github.com/{repo_owner}/{repo_name}.git .",
            cwd=d,
            shell=True,
        )
        git_sha = (
            subprocess.check_output("git rev-parse HEAD", shell=True, cwd=d)
            .decode("utf-8")
            .strip()
        )
        repo_path = pathlib.Path(d)
        markdown_files = list(repo_path.glob("**/*.md")) + list(repo_path.glob("**/*.mdx"))
        for markdown_file in markdown_files:
            with open(markdown_file, "r") as f:
                relative_path = markdown_file.relative_to(repo_path)
                github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/{git_sha}/{relative_path}"
                yield Document(page_content=f.read(), metadata={"source": github_url})