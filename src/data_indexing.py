import faiss
import pickle
import json
import os
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict


def load_processed_data(data_path: str = "processed_data/chunks_dataset.json") -> List[Dict]:
    """
    Load the preprocessed chunks dataset
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f"Loaded {len(documents)} chunks")
    return documents


def create_faiss_index(documents: List[Dict]) -> faiss.Index:
    """
    Create FAISS vector index
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    document_texts = [doc["content"] for doc in documents]
    document_embeddings = model.encode(document_texts, show_progress_bar=True)

    # Create FAISS index
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    faiss.normalize_L2(document_embeddings)
    index.add(document_embeddings.astype('float32'))

    return index


def create_bm25_index(documents: List[Dict]) -> BM25Okapi:
    """
    Create BM25 index
    """
    # Tokenize documents
    tokenized_corpus = [doc["content"].lower().split() for doc in documents]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


def save_indexes(faiss_index, bm25, output_dir: str = "podcast_index"):
    """
    Save indexes to disk
    """
    os.makedirs(output_dir, exist_ok=True)

    faiss.write_index(faiss_index, f"{output_dir}/faiss_index.bin")

    with open(f"{output_dir}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print("BM25 index saved")


def load_indexes(index_dir: str = "podcast_index"):
    """
    Load saved indexes
    Returns: (faiss_index, bm25)
    """
    faiss_index = faiss.read_index(f"{index_dir}/faiss_index.bin")

    with open(f"{index_dir}/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    return faiss_index, bm25


def build_indexes():
    """
    Main function to build both indexes
    """

    documents = load_processed_data()

    faiss_index = create_faiss_index(documents)
    bm25 = create_bm25_index(documents)

    save_indexes(faiss_index, bm25)


if __name__ == "__main__":
    build_indexes()