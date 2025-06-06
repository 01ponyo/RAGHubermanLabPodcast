import json
import requests
import re
from typing import List, Dict
from prompts import PodcastRerankingPrompt
import os
from dotenv import load_dotenv

load_dotenv()


class OpenAIReranker:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def create_reranking_prompt(self, query: str, chunks: List[Dict]) -> str:
        """
        Prompt for reranking multiple chunks
        """
        prompt = f"Query: {query}\n\n"
        prompt += "Evaluate the relevance of each podcast transcript block to the query:\n\n"

        for i, chunk in enumerate(chunks, 1):
            episode_info = f"Episode {chunk['episode_number']}: {chunk['episode_title']}"
            content_preview = chunk['combined_content'][:800] + "..." if len(chunk['combined_content']) > 800 else \
            chunk['combined_content']

            prompt += f"Block {i}:\n"
            prompt += f"Source: {episode_info}\n"
            prompt += f"Content: {content_preview}\n\n"

        prompt += "Provide your evaluation in the specified JSON format."
        return prompt

    def call_openai_api(self, prompt: str) -> str:
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": PodcastRerankingPrompt.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"}  # Force JSON output
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data,
                timeout=30
            )

            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")


    def parse_reranking_response(self, response: str) -> List[Dict]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                return result.get('results', [])
            else:
                print("No JSON found in response")
                return self._parse_simple_format(response)

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response preview: {response[:200]}...")
            return self._parse_simple_format(response)

    def parse_simple_format(self, response: str) -> List[Dict]:
        results = []
        lines = response.split('\n')

        current_block = None
        current_score = None
        current_reasoning = ""

        for line in lines:
            line = line.strip()

            if line.startswith('Block') and ':' in line:
                if current_block is not None:
                    results.append({
                        'block_id': current_block,
                        'score': current_score or 0.5,
                        'reasoning': current_reasoning.strip()
                    })

                try:
                    current_block = int(line.split()[1].rstrip(':'))
                    current_score = None
                    current_reasoning = ""
                except:
                    pass

            elif 'score' in line.lower() and any(char.isdigit() for char in line):
                try:
                    score_match = re.search(r'(\d*\.?\d+)', line)
                    if score_match:
                        current_score = float(score_match.group(1))
                        if current_score > 1:
                            current_score = current_score / 10  # Convert 8 to 0.8
                except:
                    pass

            elif line and not line.startswith('Block') and 'score' not in line.lower():
                current_reasoning += line + " "

        if current_block is not None:
            results.append({
                'block_id': current_block,
                'score': current_score or 0.5,
                'reasoning': current_reasoning.strip()
            })

        return results

    def rerank_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Main reranking function
        """
        prompt = self.create_reranking_prompt(query, chunks)

        response = self.call_openai_api(prompt)
        if not response:
            print("Failed.")
            return chunks

        rankings = self.parse_reranking_response(response)
        if not rankings:
            print("Failed.")
            return chunks

        reranked_chunks = []
        for ranking in rankings:
            block_id = ranking.get('block_id', 0)
            score = ranking.get('score', 0.0)
            reasoning = ranking.get('reasoning', '')

            if 1 <= block_id <= len(chunks):
                chunk = chunks[block_id - 1].copy()  # block_id is 1-indexed
                chunk['llm_score'] = score
                chunk['llm_reasoning'] = reasoning
                reranked_chunks.append(chunk)

        reranked_chunks.sort(key=lambda x: x.get('llm_score', 0), reverse=True)

        scores_list = [f"{chunk.get('llm_score', 0):.1f}" for chunk in reranked_chunks]
        print(f"Reranking complete. Scores: {scores_list}")

        return reranked_chunks


def rerank_with_llm(query: str, chunks: List[Dict], api_key: str = None) -> List[Dict]:

    reranker = OpenAIReranker(api_key=api_key)
    return reranker.rerank_chunks(query, chunks)


def print_reranked_results(results: List[Dict], max_content_length: int = 200):
    """
    Print reranked results with LLM scores
    """
    print("\n" + "=" * 80)
    print("LLM RERANKED RESULTS (OpenAI GPT-3.5-turbo)")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Episode: EP{result['episode_number']} - {result['episode_title']}")
        print(f"RRF Score: {result['rrf_score']:.4f}")

        if 'llm_score' in result:
            print(f"LLM Score: {result['llm_score']:.1f}")
            print(f"LLM Reasoning: {result['llm_reasoning']}")

        print(f"Content preview: {result['combined_content'][:max_content_length]}...")


# Test the system
if __name__ == "__main__":

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    try:
        from retriever import hybrid_retrieve

        # Test query
        query = "How does sleep affect dopamine levels?"

        # Getting results from hybrid retriever
        chunks = hybrid_retrieve(query, top_k=5, context_size=2)

        # Reranking with OpenAI GPT
        if OPENAI_API_KEY:
            reranked_chunks = rerank_with_llm(query, chunks, OPENAI_API_KEY)
            print_reranked_results(reranked_chunks)

            # Show cost estimate
            estimated_tokens = len(' '.join([chunk['combined_content'] for chunk in chunks])) // 4
            estimated_cost = estimated_tokens * 0.002 / 1000
            print(f"\nEstimated cost: ~${estimated_cost:.4f}")
        else:
            print("Failed.")

    except Exception as e:
        print(f"Error: {e}")