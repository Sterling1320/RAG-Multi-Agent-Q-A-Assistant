RAG + Multi-Agent Q&A Assistant

This project is a question-answering assistant that uses a combination of multi-agent logic and retrieval-augmented generation (RAG). It can handle simple math, look up definitions (placeholder), or answer general questions using document context and Gemini.

Architecture:
- The Agent class decides which tool to use based on the query: a calculator, dictionary (not fully implemented), or a RAG pipeline.
- RAGEngine loads text documents from the docs folder, splits them into chunks, and encodes them using sentence-transformers. It retrieves the most relevant chunks for a given query.
- The selected context is passed to Gemini (gemini-2.0-flash) for answer generation.
- The interface is built using Streamlit. A CLI is also available for testing.

Key Design Choices:
- Modular agent-based structure for easy extensibility.
- Lightweight embedding model (all-MiniLM-L6-v2) for fast retrieval.
- Fast LLM (Gemini Flash) for quick responses.
- Simple fallback logic in case of unsupported queries.

How to Run:
1. Install dependencies:
   pip install -r requirements.txt

2. Place .txt documents in a folder named docs.

3. Add your Gemini API key in .streamlit/secrets.toml:
   GEMINI_API_KEY = "your_api_key_here"

4. Launch the Streamlit app:
   streamlit run interface.py

5. (Optional) Run from terminal using:
   python main.py
