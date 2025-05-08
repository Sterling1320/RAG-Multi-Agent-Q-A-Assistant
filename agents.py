import os
import streamlit as st
import google.generativeai as genai
from rag import RAGEngine

# Configure Gemini with the API key from Streamlit secrets
import streamlit as st

if "GEMINI_API_KEY" not in st.secrets:
    raise RuntimeError("❌ GEMINI_API_KEY is missing in Streamlit secrets")

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

        # Check for math-related query
        if any(kw in query.lower() for kw in ["calculate", "+", "-", "*", "/", "math"]):
            try:
                # Clean the query to isolate the math expression
                expression = ''.join(c if c.isdigit() or c in '+-*/.()' else ' ' for c in query)
                # Attempt to evaluate the expression
                answer = str(eval(expression))
                log["used_tool"] = "calculator"
                log["context"] = "N/A"
            except Exception as e:
                answer = f"Invalid calculation: {str(e)}"

        # Check for "define" related query
        elif "define" in query.lower():
            word = query.lower().replace("define", "").strip()
            # Simulated definition response (replace with actual lookup in the future)
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
