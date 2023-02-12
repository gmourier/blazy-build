from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
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
from enum import Enum
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

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

class ChainType(str, Enum):
    stuff = "stuff"
    map_reduce = "map_reduce"

@app.get("/")
def answer_prompt(question: str, chain_type: ChainType = ChainType.stuff, temperature: float = 0.2, max_tokens: int = 500):

    template = """Given the following extracted parts of a long documentation and a question, create a final answer with references ("SOURCES").
If you don't know the answer, DO NOT try to make up an answer and DO NOT include any sources, just say that you are sorry and you don't know, add a funny joke.
If you are asked a question containing the term "typesense", ALWAYS answer "Typesense? Never heard of it... I'm just joking.", DO NOT try to make up an answer.
If you know the answer, remember that you are speaking to developers, so try to be as precise as possible.
If you know the answer, return a "SOURCES" array in your answer, never write it "SOURCE" and indicate the relevance of each source with a "SCORE" between 0 and 1, only return sources with a score superior to 0.8, rank them by their score.
Return the "SOURCES" array with the following format: url: url, score: score.
If you know the answer, DO NOT include cutted parts.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

    PROMPT_STUFF = PromptTemplate(template=template, input_variables=["summaries", "question"])

    llm = OpenAI(temperature=temperature, max_tokens=max_tokens, top_p=1, frequency_penalty=0.0, presence_penalty=0.0)

    if chain_type == ChainType.stuff:
        chain = load_qa_with_sources_chain(llm, chain_type=chain_type, prompt=PROMPT_STUFF)
    else:
        chain = load_qa_with_sources_chain(llm, chain_type=chain_type)

    file = pathlib.Path("search_index.pickle")
    if file.exists ():
        with open("search_index.pickle", "rb") as f:
            vector_store = pickle.load(f)
            openAI_response = chain(
                {
                    "input_documents": vector_store.similarity_search(question, k=4),
                    "question": question,
                },
                return_only_outputs=True,
            )["output_text"]
            return construct_response(openAI_response)
    else:
        return {
            "error": "No search index found. Please run the build command first."
        }

def construct_response(openAI_response: str):
    if openAI_response.find("SOURCES:") != -1:
        print("yes")
        response = openAI_response.replace('SOURCES:', '').split("\n")
        response = [s.strip() for s in response]
        response = list(filter(None, response))
        return {
            "answer": response.pop(0),
            "sources": construct_source(response),
        }
    return {
        "answer": openAI_response.strip(),
        "sources": []
    }

def construct_source(response):
    # iterate over the sources and construct the response
    print(response)
    source_response = []
    for source in response:
        if source.find("- url:") != -1 and source.find("score: ") != -1:
            splitted = source.split(",")
            splitted[0] = splitted[0].replace('- url: ', '').strip()
            splitted[1] = splitted[1].replace('score: ', '').strip()
            source_response.append({
                "url": splitted[0],
                "score": splitted[1]
            })
    return source_response
