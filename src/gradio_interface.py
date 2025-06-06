import gradio as gr
import os
from typing import Tuple
from dotenv import load_dotenv
load_dotenv()


from response_generator import complete_rag_pipeline


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def process_query(query: str) -> Tuple[str, str]:
    """
    Process user query
    """
    try:
        response = complete_rag_pipeline(query, OPENAI_API_KEY, top_k=5, context_size=2)

        answer = response.get('answer', 'No answer generated.')

        sources = response.get('sources', [])
        if sources:
            sources_text = "**Sources:**\n"
            for i, source in enumerate(sources, 1):
                episode_num = source.get('episode_number', 'Unknown')
                episode_title = source.get('episode_title', 'Unknown Title')
                llm_score = source.get('llm_score', 'N/A')

                if llm_score != 'N/A':
                    sources_text += f"{i}. Episode {episode_num}: {episode_title} (Relevance: {llm_score:.1f})\n"
                else:
                    sources_text += f"{i}. Episode {episode_num}: {episode_title}\n"
        else:
            sources_text = "No sources found."

        return answer, sources_text

    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        return error_msg, ""


def create_interface():
    """
    Create and configure the Gradio interface
    """

    css = """
    .gradio-container {
        max-width: 900px !important;
        margin: auto !important;
    }
    .title {
        text-align: center;
        color: #2563eb;
        margin-bottom: 20px;
    }
    .description {
        text-align: center;
        font-size: 16px;
        color: #374151;
        margin-bottom: 25px;
        line-height: 1.5;
    }
    .sources-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
    }
    """

    with gr.Blocks(css=css, title="Huberman Lab RAG") as interface:
        # Title
        gr.Markdown(
            "# 🧠 The Huberman Lab Podcast RAG",
            elem_classes=["title"]
        )

        # Description
        gr.Markdown(
            "Ask whatever you want to know about science and science-based tools for everyday life, as discussed by Andrew Huberman and his guests",
            elem_classes=["title"]
        )

        # Input section
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., How does sleep affect dopamine levels?",
                    lines=2,
                    max_lines=5
                )

                submit_btn = gr.Button(
                    "🔍 Ask Huberman Lab",
                    variant="primary",
                    size="lg"
                )

        # Output section
        with gr.Row():
            with gr.Column():
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=10,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True
                )

                sources_output = gr.Textbox(
                    label="Episode Sources",
                    lines=5,
                    max_lines=10,
                    interactive=False,
                    show_copy_button=True
                )

        # Example queries
        gr.Markdown("### 💡 Example Questions:")

        example_queries = [
            "How does sleep affect dopamine levels?",
            "What are Huberman's recommendations for morning light exposure?",
            "What supplements does Huberman recommend for sleep?",
            "How does cold exposure affect the immune system?",
            "What protocols does Huberman suggest for focus and concentration?"
        ]

        gr.Examples(
            examples=[[query] for query in example_queries],
            inputs=[query_input],
            label="Click on any example to try it:"
        )

        # Connect the interface
        submit_btn.click(
            fn=process_query,
            inputs=[query_input],
            outputs=[answer_output, sources_output],
            show_progress=True
        )

        # Allow Enter key to submit
        query_input.submit(
            fn=process_query,
            inputs=[query_input],
            outputs=[answer_output, sources_output],
            show_progress=True
        )

        # Footer
        gr.Markdown(
            """
            ---
            **Note:** This system retrieves information from Huberman Lab podcast transcripts and uses AI to generate responses. 
            Always verify important health information with qualified professionals.
            """
        )

    return interface


def main():
    interface = create_interface()
    interface.launch(
        share=True,  # Creates public link for sharing
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Default Gradio port
        show_error=True,  # Show errors in interface
        quiet=False  # Show startup logs
    )


if __name__ == "__main__":
    main()