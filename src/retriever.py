import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import faiss
import pickle


def load_data_and_indexes():
    """
    Load preprocessed data and indexes
    """
    with open("processed_data/chunks_dataset.json", 'r', encoding='utf-8') as f:
        documents = json.load(f)

    faiss_index = faiss.read_index("podcast_index/faiss_index.bin")

    with open("podcast_index/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    return documents, faiss_index, bm25, model


def search_bm25(query: str, bm25, documents: List[Dict], top_k: int = 20) -> List[Tuple[int, float]]:
    """
    Search using BM25 and return document indices with scores
    """
    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [(idx, scores[idx]) for idx in top_indices]

    return results


def search_faiss(query: str, faiss_index, model, top_k: int = 20) -> List[Tuple[int, float]]:
    """
    Search using FAISS and return document indices with scores
    """
    query_embedding = model.encode([query])

    faiss.normalize_L2(query_embedding)

    scores, indices = faiss_index.search(query_embedding.astype('float32'), top_k)

    results = [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0]))]

    return results


def reciprocal_rank_fusion(bm25_results: List[Tuple[int, float]],
                           faiss_results: List[Tuple[int, float]],
                           k: int = 60) -> List[Tuple[int, float]]:
    """
    Combine results using Reciprocal Rank Fusion
    """
    rrf_scores = {}

    # BM25
    for rank, (doc_id, score) in enumerate(bm25_results, 1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

    # FAISS
    for rank, (doc_id, score) in enumerate(faiss_results, 1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

    # Sort by RRF score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_results


def expand_chunks(relevant_chunks: List[Tuple[int, float]],
                  documents: List[Dict],
                  context_size: int = 2) -> List[Dict]:
    """
    Expand each relevant chunk to include surrounding chunks > parent retrieval
    context_size: number of chunks before and after to include
    """
    expanded_results = []
    processed_chunks = set()

    for chunk_idx, rrf_score in relevant_chunks:
        episode_num = documents[chunk_idx]['episode_number']
        episode_title = documents[chunk_idx]['episode_title']

        start_idx = max(0, chunk_idx - context_size)
        end_idx = min(len(documents), chunk_idx + context_size + 1)

        episode_chunks = []
        for i in range(start_idx, end_idx):
            if (documents[i]['episode_number'] == episode_num and
                    i not in processed_chunks):
                episode_chunks.append({
                    'chunk_idx': i,
                    'episode_number': documents[i]['episode_number'],
                    'episode_title': documents[i]['episode_title'],
                    'chunk_id': documents[i]['chunk_id'],
                    'content': documents[i]['content'],
                    'is_primary': i == chunk_idx,
                    'rrf_score': rrf_score if i == chunk_idx else 0.0
                })
                processed_chunks.add(i)

        if episode_chunks:
            combined_content = '\n\n'.join([chunk['content'] for chunk in episode_chunks])

            expanded_results.append({
                'primary_chunk_idx': chunk_idx,
                'episode_number': episode_num,
                'episode_title': episode_title,
                'rrf_score': rrf_score,
                'combined_content': combined_content,
                'chunk_details': episode_chunks,
                'total_chunks': len(episode_chunks)
            })

    return expanded_results


def hybrid_retrieve(query: str, top_k: int = 10, context_size: int = 2) -> List[Dict]:
    """
    Main retrieval function
    """
    documents, faiss_index, bm25, model = load_data_and_indexes()

    bm25_results = search_bm25(query, bm25, documents, top_k=20)
    faiss_results = search_faiss(query, faiss_index, model, top_k=20)

    rrf_results = reciprocal_rank_fusion(bm25_results, faiss_results)

    top_results = rrf_results[:top_k]

    expanded_results = expand_chunks(top_results, documents, context_size)

    return expanded_results


def print_retrieval_results(results: List[Dict], max_content_length: int = 200):
    """
    Print retrieval results for testing
    """
    print("\n" + "=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Episode: EP{result['episode_number']} - {result['episode_title']}")
        print(f"RRF Score: {result['rrf_score']:.4f}")
        print(f"Total chunks included: {result['total_chunks']}")
        print(f"Content preview: {result['combined_content'][:max_content_length]}...")

        chunk_info = [f"#{chunk['chunk_id']}{'*' if chunk['is_primary'] else ''}"
                      for chunk in result['chunk_details']]
        print(f"Chunks: {', '.join(chunk_info)} (* = primary match)")


if __name__ == "__main__":

   test_queries = [
       "How does sleep affect dopamine levels?",
       "What does Andrew Huberman say about caffeine and sleep?",
       "What is the best time to exercise for optimal performance?"
   ]

   for query in test_queries:
       results = hybrid_retrieve(query, top_k=5, context_size=2)
       print_retrieval_results(results, max_content_length=150)