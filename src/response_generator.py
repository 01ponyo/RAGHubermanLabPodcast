import json
import requests
from typing import List, Dict
from prompts import ResponseGenerationPrompt
import os
from dotenv import load_dotenv

load_dotenv()


class ResponseGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize response generator with OpenAI
        Using gpt-4o-mini
        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.max_context_tokens = 20000


    def create_system_prompt(self) -> str:
        return ResponseGenerationPrompt.system_prompt

    def format_context(self, chunks: List[Dict], max_tokens: int = 8000) -> str:
        """
        Format retrieved chunks into context for the LLM with token limit
        """
        if not chunks:
            return "No relevant context found."

        context_parts = []
        context_parts.append("=== RETRIEVED PODCAST CONTENT ===\n")

        current_tokens = 100

        for i, chunk in enumerate(chunks, 1):
            episode_info = f"Episode {chunk['episode_number']}: {chunk['episode_title']}"
            llm_score = chunk.get('llm_score', 'N/A')

            chunk_content = chunk['combined_content']
            chunk_tokens = len(chunk_content) // 4

            # Check if adding this chunk would exceed limit
            if current_tokens + chunk_tokens > max_tokens:
                # Truncate this chunk to fit
                remaining_tokens = max_tokens - current_tokens - 200
                if remaining_tokens > 500:
                    truncated_content = chunk_content[:remaining_tokens * 4] + "... [truncated]"
                    context_parts.append(f"--- Source {i} ---")
                    context_parts.append(f"Episode: {episode_info}")
                    context_parts.append(f"Relevance Score: {llm_score}")
                    context_parts.append(f"Content: {truncated_content}")
                    context_parts.append("")
                break

            # Add full chunk
            context_parts.append(f"--- Source {i} ---")
            context_parts.append(f"Episode: {episode_info}")
            context_parts.append(f"Relevance Score: {llm_score}")
            context_parts.append(f"Content: {chunk_content}")
            context_parts.append("")  # Empty line for separation

            current_tokens += chunk_tokens

        return "\n".join(context_parts)

    def create_user_prompt(self, query: str, chunks: List[Dict]) -> str:
        context = self.format_context(chunks, self.max_context_tokens)

        prompt = f"""Based on the following Huberman Lab podcast content, please answer this question:

QUESTION: {query}

{context}

Please provide a comprehensive answer based on the retrieved content above. Make sure to:
- Reference specific episodes when citing information
- Provide actionable protocols if relevant
- Explain scientific mechanisms clearly
- Acknowledge if information is incomplete

ANSWER:"""

        return prompt

    def call_openai(self, system_prompt: str, user_prompt: str) -> str:
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500,
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data,
                timeout=45
            )

            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")

    def generate_response(self, query: str, chunks: List[Dict]) -> Dict[str, any]:

        system_prompt = self.create_system_prompt()
        user_prompt = self.create_user_prompt(query, chunks)

        response = self.call_openai(system_prompt, user_prompt)

        if not response:
            return {
                "query": query,
                "answer": "Sorry, I couldn't generate a response at this time. Please try again.",
                "sources": [],
                "error": "API call failed"
            }

        # Extract source information
        sources = []
        for chunk in chunks:
            source_info = {
                "episode_number": chunk['episode_number'],
                "episode_title": chunk['episode_title'],
                "llm_score": chunk.get('llm_score', 'N/A'),
                "rrf_score": chunk.get('rrf_score', 'N/A')
            }
            if source_info not in sources:  # Avoid duplicates
                sources.append(source_info)

        return {
            "query": query,
            "answer": response,
            "sources": sources,
            "num_chunks_used": len(chunks)
        }


def generate_rag_response(query: str, chunks: List[Dict], api_key: str, model: str = "gpt-4o-mini") -> Dict[str, any]:
    generator = ResponseGenerator(api_key, model)
    return generator.generate_response(query, chunks)


def print_rag_response(response: Dict[str, any]):
    print(f"\nQuery: {response['query']}")
    print(f"\nAnswer:\n{response['answer']}")

    sources = response.get('sources', [])
    if sources:
        print(f"\nSources ({len(sources)} episodes):")
        for i, source in enumerate(sources, 1):
            score_info = ""
            if source.get('llm_score', 'N/A') != 'N/A':
                score_info = f" (Score: {source['llm_score']:.1f})"
            print(f"  {i}. Episode {source['episode_number']}: {source['episode_title']}{score_info}")

    if 'error' in response:
        print(f"\nError: {response['error']}")


def complete_rag_pipeline(query: str, api_key: str, top_k: int = 5, context_size: int = 2) -> Dict[str, any]:
    """
    Complete RAG pipeline: Retrieve -> Rerank -> Generate
    """
    try:
        from retriever import hybrid_retrieve
        from LLM_reranker import rerank_with_llm

        chunks = hybrid_retrieve(query, top_k=top_k, context_size=context_size)

        reranked_chunks = rerank_with_llm(query, chunks, api_key)

        top_chunks = reranked_chunks[:2]
        response = generate_rag_response(query, top_chunks, api_key, model="gpt-4o-mini")

        return response

    except ImportError as e:
        return {
            "query": query,
            "answer": f"Pipeline error: Missing module {e}",
            "sources": [],
            "error": "Import error"
        }
    except Exception as e:
        return {
            "query": query,
            "answer": f"Pipeline error: {e}",
            "sources": [],
            "error": str(e)
        }


if __name__ == "__main__":

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Test queries
    test_queries = [
        "How does sleep affect dopamine levels?",
        "What are Huberman's recommendations for optimizing morning light exposure?",
    ]

    if OPENAI_API_KEY:
        for query in test_queries:
            print(f"\n{'=' * 100}")
            print(f"TESTING QUERY: {query}")
            print('=' * 100)

            response = complete_rag_pipeline(query, OPENAI_API_KEY)
            print_rag_response(response)

            print("\n" + "=" * 100 + "\n")
    else:
        print("Failed.")
