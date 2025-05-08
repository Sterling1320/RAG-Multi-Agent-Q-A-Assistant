RAG + Multi-Agent Q&A Assistant

Objective:
A knowledge assistant that retrieves relevant information from a document collection using Retrieval-Augmented Generation (RAG) and generates natural-language answers using the Gemini LLM.

Features:
- Document Ingestion & Chunking: Loads documents, chunks them, and generates embeddings for efficient retrieval.
- RAG Retrieval: Uses cosine similarity to find the top 3 most relevant document chunks.
- LLM Integration: Utilizes the Gemini LLM to generate natural language answers based on the retrieved context.
- Tool Routing: Routes queries containing math or definition requests to specific tools (calculator or dictionary) and other queries to the RAG → LLM pipeline.
- Web Interface: Built with Streamlit to interact with the assistant via a simple UI.

Installation:
1. Install required dependencies:
    pip install -r requirements.txt

2. Set up your API keys:
    - Obtain a Gemini API key from Google and set it as an environment variable:
        export GEMINI_API_KEY="your-api-key"
    - Alternatively, replace the API key directly in the code (not recommended for security reasons).

Usage:
1. To run the Streamlit web interface:
    streamlit run interface.py

2. To run from the command line (CLI):
    python main.py

Key Components:
- RAG Engine: Manages document chunking and retrieval using embeddings.
- Agent: Orchestrates the logic of routing queries to either the RAG engine, Gemini model, or specific tools (calculator, dictionary).
- Streamlit UI: Provides a text input box for user queries, displaying the answer, tool used, and retrieved context.

Known Issues:
- Latency: Initial loading of the Gemini model, document ingestion, and API calls may introduce delays, especially on the first run. This is due to model configuration and document embedding, which can take time. Subsequent queries should experience reduced latency due to caching of the resources.

Future Improvements:
- Implement more efficient document indexing and retrieval techniques (e.g., using FAISS).
- Add more external tool integrations (e.g., dictionary API for word definitions).
