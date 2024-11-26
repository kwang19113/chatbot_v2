import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Process documents per subcategory.
    subcategories = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    for subcategory in subcategories:
        print(f"Processing subcategory: {subcategory}")
        documents = load_documents(subcategory)
        chunks = split_documents(documents)
        add_to_chroma(chunks, subcategory)


def load_documents(subcategory):
    documents = []
    subcategory_path = os.path.join(DATA_PATH, subcategory)

    # Load PDF documents from subcategory.
    pdf_loader = PyPDFDirectoryLoader(subcategory_path)
    documents.extend(pdf_loader.load())

    # Load TXT documents individually with UTF-8 encoding.
    for filename in os.listdir(subcategory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(subcategory_path, filename)
            text_loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(text_loader.load())

    # Update metadata to include the subcategory.
    for doc in documents:
        doc.metadata["subcategory"] = subcategory

    return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document], subcategory):
    # Define the Chroma path for the subcategory.
    subcategory_chroma_path = os.path.join(CHROMA_PATH, subcategory)
    os.makedirs(subcategory_chroma_path, exist_ok=True)

    # Load the existing database for the subcategory.
    db = Chroma(
        persist_directory=subcategory_chroma_path, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB for {subcategory}: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents in {subcategory}: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print(f"✅ No new documents to add in {subcategory}")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
