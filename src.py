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

def get_github_docs(repo_owner, repo_name):
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
        markdown_files = list(repo_path.glob("**/*.md")) + list(
            repo_path.glob("**/*.mdx")
        )
        for markdown_file in markdown_files:
            with open(markdown_file, "r") as f:
                relative_path = markdown_file.relative_to(repo_path)
                github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/{git_sha}/{relative_path}"
                yield Document(page_content=f.read(), metadata={"source": github_url})

def source_docs():
    return list(get_github_docs("meilisearch", "documentation"))

def search_index(source_docs):
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in source_docs:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))
    with open("search_index.pickle", "wb") as f:
        pickle.dump(FAISS.from_documents(source_chunks, OpenAIEmbeddings()), f)

chain = load_qa_with_sources_chain(OpenAI(temperature=0))

def prompt(question):
    file = pathlib.Path("search_index.pickle")
    if file.exists ():
        with open("search_index.pickle", "rb") as f:
            search_file = pickle.load(f)
        print(
            # Call OpenAI to get the answer
            chain(
                {
                    "input_documents": search_file.similarity_search(question, k=4),
                    "question": question,
                },
                return_only_outputs=True,
            )["output_text"]
        )
    else:
        # No search index found. Creating one now... TODO: Move that job elsewhere to rebuild it every day?
        search_index(source_docs())
