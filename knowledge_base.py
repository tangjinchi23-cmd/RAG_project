"""
Knowledge base service.
"""
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime


def check_md5(md5_str: str):
    """Check whether the given md5 string has already been processed.
        return False (md5 not processed yet)  True (already processed, record exists)
    """
    if not os.path.exists(config.md5_path):
        # Entering the if means the file does not exist, so this md5 has never been processed.
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    else:
        with open(config.md5_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()     # Strip leading/trailing spaces and newlines
                if line == md5_str:
                    return True         # Already processed

        return False


def save_md5(md5_str: str):
    """Append the given md5 string to the record file for persistence."""
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str: str, encoding='utf-8'):
    """Convert the given string into an md5 hex string."""

    # Encode the string into a bytes array
    str_bytes = input_str.encode(encoding=encoding)

    # Create the md5 object
    md5_obj = hashlib.md5()     # md5 object
    md5_obj.update(str_bytes)   # Feed in the bytes to hash
    md5_hex = md5_obj.hexdigest()       # Get the md5 hex string

    return md5_hex


class KnowledgeBaseService(object):
    def __init__(self):
        # Create the folder if it does not exist, skip if it does
        os.makedirs(config.persist_directory, exist_ok=True)

        self.chroma = Chroma(
            collection_name=config.collection_name,     # Collection (table) name
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory,     # Local storage folder for the database
        )     # Chroma vector store instance

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,       # Maximum length of each split text chunk
            chunk_overlap=config.chunk_overlap,     # Number of overlapping characters between consecutive chunks
            separators=config.separators,       # Symbols used to split by natural paragraphs
            length_function=len,                # Use Python's built-in len as the length metric
        )     # Text splitter instance

    def upload_by_str(self, data: str, filename):
        """Vectorize the given string and store it into the vector database."""
        # Compute the md5 of the incoming string
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[Skipped] Content already exists in the knowledge base"

        if len(data) > config.max_split_char_number:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        metadata = {
            "source": filename,
            # 2025-01-01 10:00:00
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "admin",
        }

        self.chroma.add_texts(      # Load the content into the vector store
            # iterable -> list / tuple
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        #
        save_md5(md5_hex)

        return "[Success] Content has been loaded into the vector store"


if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str("sample text", "testfile")
    print(r)
