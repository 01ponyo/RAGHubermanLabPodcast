import os
import re
from pathlib import Path
from docx import Document
import json
import pandas as pd
from typing import Dict, List


def extract_episode_info(filename: str) -> Dict[str, str]:
    """
    Extract episode number and title from filename
    Expected format: "EP001 - Dr. Lex Fridman.docx"
    """
    # Remove .docx extension
    name = filename.replace('.docx', '')

    # Extract episode number and title
    match = re.match(r'EP(\d+)\s*-\s*(.+)', name)

    if match:
        episode_num = match.group(1).zfill(3)  # Pad with zeros (001, 002, etc.)
        title = match.group(2).strip()
        return {
            'episode_number': episode_num,
            'title': title,
            'filename': filename
        }
    else:
        # If pattern doesn't match
        return {
            'episode_number': 'unknown',
            'title': name,
            'filename': filename
        }


def read_docx_content(file_path: str) -> str:
    """
    Extract text content from podcast scripts
    """
    try:
        doc = Document(file_path)
        content = []

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text.strip())

        return '\n'.join(content)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def clean_text(text: str) -> str:

    # Remove multiple whitespaces and normalize spacing
    text = re.sub(r'\s+', ' ', text)

    # Remove extra newlines while preserving paragraph structure
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Keep double newlines for paragraphs
    text = re.sub(r'\n{3,}', '\n\n', text)  # Limit to max 2 newlines

    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Split text into overlapping chunks by words
    """
    words = text.split()
    chunks = []

    if len(words) <= chunk_size:
        return [text]

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = ' '.join(chunk_words)
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


def process_episode_file(file_path: str) -> Dict:
    """
    Process a single episode file and return structured data
    """
    filename = os.path.basename(file_path)
    metadata = extract_episode_info(filename)

    raw_content = read_docx_content(file_path)
    cleaned_content = clean_text(raw_content)

    chunks = chunk_text(cleaned_content)

    return {
        'metadata': metadata,
        'full_content': cleaned_content,
        'chunks': chunks,
        'num_chunks': len(chunks),
        'content_length': len(cleaned_content)
    }


def process_all_episodes(data_folder: str) -> List[Dict]:
    """
    Process all .docx files in the specified folder
    """
    data_path = Path(data_folder)
    processed_data = []

    docx_files = list(data_path.glob("*.docx"))

    for file_path in sorted(docx_files):  # Sort to process in order
        print(f"Processing: {file_path.name}")

        processed_file = process_episode_file(str(file_path))
        processed_data.append(processed_file)

    return processed_data


def create_chunks_dataset(processed_data: List[Dict]) -> List[Dict]:
    """
    Create a dataset of chunks for indexing
    """
    chunks_data = []

    for episode in processed_data:
        metadata = episode['metadata']

        for i, chunk in enumerate(episode['chunks']):
            chunks_data.append({
                'episode_number': metadata['episode_number'],
                'episode_title': metadata['title'],
                'filename': metadata['filename'],
                'chunk_id': i,
                'content': chunk,
                'chunk_length': len(chunk)
            })

    return chunks_data


def save_processed_data(processed_data: List[Dict], output_folder: str = "processed_data"):
    """
    Save processed data
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    chunks_data = create_chunks_dataset(processed_data)

    # Save full processed data as JSON
    with open(output_path / "processed_episodes.json", 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    # Save chunks as JSON (for indexing!)
    with open(output_path / "chunks_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    # Save chunks as CSV to view
    chunks_df = pd.DataFrame(chunks_data)
    chunks_df.to_csv(output_path / "chunks_dataset.csv", index=False)

    # Create summary
    summary = {
        'total_episodes': len(processed_data),
        'total_chunks': len(chunks_data),
        'avg_chunks_per_episode': len(chunks_data) / len(processed_data) if processed_data else 0,
        'episodes_info': [
            {
                'episode': ep['metadata']['episode_number'],
                'title': ep['metadata']['title'],
                'chunks': ep['num_chunks'],
                'content_length': ep['content_length']
            }
            for ep in processed_data
        ]
    }

    with open(output_path / "processing_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return chunks_data


def main():

    DATA_FOLDER = "/Users/polina/PycharmProjects/RAGHubermanLabPodcast/Huberman Lab Podcast Episode Files"
    OUTPUT_FOLDER = "processed_data"

    processed_data = process_all_episodes(DATA_FOLDER)

    chunks_data = save_processed_data(processed_data, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()