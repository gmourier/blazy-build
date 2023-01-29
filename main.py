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
from pydantic import BaseModel
# chain = load_qa_with_sources_chain(OpenAI(temperature=0))

# def prompt(question):
#     file = pathlib.Path("search_index.pickle")
#     if file.exists ():
#         with open("search_index.pickle", "rb") as f:
#             search_file = pickle.load(f)
#         print(
#             # Call OpenAI to get the answer
#             chain(
#                 {
#                     "input_documents": search_file.similarity_search(question, k=4),
#                     "question": question,
#                 },
#                 return_only_outputs=True,
#             )["output_text"]
#         )
#     else:
#         print("No search index found. Please run the build command first.")



app = FastAPI()


@app.get("/")
def home():
    return {"Hey, I'm Blazy and I'm here to help you find the answer to your question about Meilisearch. Use the POST route to ask me something."}

@app.post("/")
def answer_prompt(question: str):
    chain = load_qa_with_sources_chain(OpenAI(temperature=0))
    file = pathlib.Path("search_index.pickle")
    if file.exists ():
        with open("search_index.pickle", "rb") as f:
            search_file = pickle.load(f)
        return {
            "answer": chain(
                {
                    "input_documents": search_file.similarity_search(question, k=4),
                    "question": question,
                },
                return_only_outputs=True,
            )["output_text"]
        }
    else:
        return {
            "answer": "No search index found. Please run the build command first."
        }