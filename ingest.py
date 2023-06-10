"""Parse documentation sources, compute embeddings and store them in a vector space."""
import pathlib
import subprocess
import tempfile
import pickle

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores.meilisearch import Meilisearch
import meilisearch

import logging
logger = logging.getLogger(__name__)

def ingest_docs(org_name: str, repo_name: str):
    merged_sources = source_content(org_name, repo_name)
    source_chunks = []
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    for source in merged_sources:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

    client = meilisearch.Client('http://127.0.0.1:7700')
    index = client.index('langchain_demo')
    index.delete() # delete the index if it already exists to start from fresh data
    embeddings = OpenAIEmbeddings()

    print("Compute and add documents embeddings to Meilisearch vectorstore...")
    vectorstore = Meilisearch(index, embeddings.embed_query, "text")
    vectorstore.add_documents(source_chunks)
    print("Done.")

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

        markdown_files = list(repo_path.glob("**/resources/**/*.mdx")) + list(repo_path.glob("**/learn/**/*.mdx")) + list(repo_path.glob("**/reference/**/*.mdx"))

        for markdown_file in markdown_files:
            with open(markdown_file, "r") as f:
                relative_path = markdown_file.relative_to(repo_path)
                github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/{git_sha}/{relative_path}"
                yield Document(page_content=f.read(), metadata={"source": github_url})

if __name__ == "__main__":
    ingest_docs("meilisearch", "documentation")