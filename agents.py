import os
from dotenv import load_dotenv
import google.generativeai as genai
from rag import RAGEngine

load_dotenv()

try:
    import streamlit as st
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except ImportError:
    st = None  # In case you're not using Streamlit locally

# Retrieve the key safely
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("❌ GEMINI_API_KEY not found. Set it in .env or Streamlit secrets.")

# Configure Gemini
genai.configure(api_key=api_key)

class Agent:
    def __init__(self):
        try:
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize Gemini model: {e}")

        self.rag = RAGEngine()
        self.rag.load_documents("docs")

    def run(self, query):
        log = {"used_tool": "None", "answer": "No answer", "context": "No context found"}

        if any(kw in query.lower() for kw in ["calculate", "+", "-", "*", "/", "math"]):
            try:
                answer = str(eval(query))
                log["used_tool"] = "calculator"
                log["context"] = "N/A"
            except Exception:
                answer = "Invalid calculation."
        elif "define" in query.lower():
            word = query.lower().replace("define", "").strip()
            answer = f"{word}: [definition lookup not implemented]"
            log["used_tool"] = "dictionary"
            log["context"] = "N/A"
        else:
            context_chunks = self.rag.retrieve(query)
            context_texts = [c['text'] for c in context_chunks]
            log["used_tool"] = "RAG"
            log["context"] = context_texts

            context = "\n".join(context_texts)
            prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""

            try:
                response = self.model.generate_content(prompt)
                answer = response.text.strip()
            except Exception as e:
                answer = f"❌ Failed to generate content: {e}"

        log["answer"] = answer
        return log
