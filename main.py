from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
import pathlib
import subprocess
import tempfile
import pickle
from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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

@app.get("/")
def home():
    return {
        "Hey, I'm Blazy and I'm here to help you find the answer to your question about Meilisearch. Use the POST route to ask me something."
    }

@app.post("/")
def answer_prompt(question: str):
    chain = load_qa_with_sources_chain(OpenAI(temperature=0))
    file = pathlib.Path("search_index.pickle")
    if file.exists ():
        with open("search_index.pickle", "rb") as f:
            search_file = pickle.load(f)
            openAI_response = chain(
                {
                    "input_documents": search_file.similarity_search(question, k=4),
                    "question": question,
                },
                return_only_outputs=True,
            )["output_text"]
            return construct_response(openAI_response)
    else:
        return {
            "error": "No search index found. Please run the build command first."
        }

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

def construct_response(openAI_response: str):
    response = openAI_response.split("\n")
    print(response) # DEBUG LOG
    # response.remove('SOURCES:')
    text_answer = response.pop(0)
    return {
        "answer": text_answer.strip()
        # "sources": response,
    }