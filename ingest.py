"""Parse documentation sources, compute embeddings and store them in a vector space."""
import pathlib
import subprocess
import tempfile
import pickle

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS

def ingest_docs(org_name: str, repo_name: str):
    merged_sources = source_content(org_name, repo_name)

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

if __name__ == "__main__":
    ingest_docs("meilisearch", "documentation")