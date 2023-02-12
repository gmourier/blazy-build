from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import pathlib
import subprocess
import tempfile
import pickle

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

def source_content(repo_owner, repo_name):
    return list(get_github_content(repo_owner, repo_name))

# Use NLTKTextSplitter to split the text into chunks of 1000 characters
# It seems to be slightly better than CharacterTextSplitter
# TODO: Test SpacyTextSplitter, https://langchain.readthedocs.io/en/latest/reference/modules/text_splitter.html
def build_index():
    merged_sources = source_content("meilisearch", "documentation")

    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1000, chunk_overlap=0)
    for source in merged_sources:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))
    with open("search_index.pickle", "wb") as f:
        pickle.dump(FAISS.from_documents(source_chunks, OpenAIEmbeddings()), f)
def main():
    build_index()

main()
