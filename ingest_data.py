"""
One-off script to (re)build the Chroma vector store from the txt files in ./data.
Run:  python ingest_data.py
"""
import os
from knowledge_base import KnowledgeBaseService

DATA_DIR = "./data"


def main():
    service = KnowledgeBaseService()

    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        result = service.upload_by_str(text, filename)
        print(f"{filename}: {result}")


if __name__ == "__main__":
    main()
