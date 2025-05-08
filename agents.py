import os
import streamlit as st
import google.generativeai as genai
from rag import RAGEngine

# Configure Gemini with the API key from Streamlit secrets
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

class Agent:
    def __init__(self):
        try:
            # Use the latest stable model
            self.model = genai.GenerativeModel("gemini-2.0-flash")
            self.rag = RAGEngine()
            self.rag.load_documents("docs")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize Gemini model: {e}")

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

            response = self.model.generate_content(prompt)
            answer = response.text.strip()

        log["answer"] = answer
        return log
