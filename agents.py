import os
import google.generativeai as genai
from rag import RAGEngine

# Set the Gemini API key
os.environ["GEMINI_API_KEY"] = "AIzaSyC5RWLt9_CXdAJBWcJTKD0ziQ74uCL4maM"  # Replace with your actual API key

class Agent:
    def __init__(self):
        # Configure the Gemini API with your API key
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.rag = RAGEngine()
        self.rag.load_documents("docs")  # Folder containing your .txt files

    def run(self, query):
        log = {"used_tool": "None", "answer": "No answer", "context": "No context found"}

        # Tool routing logic
        if any(kw in query.lower() for kw in ["calculate", "+", "-", "*", "/", "math"]):
            try:
                answer = str(eval(query))
                log["used_tool"] = "calculator"
                log["context"] = "N/A"
            except Exception:
                answer = "Invalid calculation."
        elif "define" in query.lower():
            word = query.lower().replace("define", "").strip()
            answer = f"{word}: [definition lookup not implemented]"  # Placeholder for actual dictionary API
            log["used_tool"] = "dictionary"
            log["context"] = "N/A"
        else:
            # RAG retrieval
            context_chunks = self.rag.retrieve(query)
            context_texts = [c['text'] for c in context_chunks]
            log["used_tool"] = "RAG"
            log["context"] = context_texts

            # Prepare the prompt for Gemini
            context = "\n".join(context_texts)
            prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""

            # Get response from Gemini model
            response = self.model.generate_content(prompt)
            answer = response.text.strip()

        log["answer"] = answer
        return log
