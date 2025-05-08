from sentence_transformers import SentenceTransformer
import torch
import os

class RAGEngine:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = []
        self.embeddings = []

    def load_documents(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    self._add_chunks(text, filename)

    def _add_chunks(self, text, source):
        chunk_size = 300
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            embedding = self.model.encode(chunk, convert_to_tensor=True)
            self.chunks.append({"text": chunk, "source": source})
            self.embeddings.append(embedding)

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(query_embedding, torch.stack(self.embeddings))
        top_indices = torch.topk(similarities, top_k).indices
        return [self.chunks[i] for i in top_indices]
