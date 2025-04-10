from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import os
import faiss

class VectorDBLangchain:
    def __init__(self, dim, index_path = "../data/processed/first_vector_db"):
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.index_path = index_path
        self.embedding_dim = dim
        self.db = None

    def create_db(self, embeddings, texts):
        print("Creating new index...")
        text_embeddings = list(zip(texts, embeddings))
        # Create FAISS index
        self.db = FAISS.from_embeddings(text_embeddings, self.embedding_model)
        self.db.save_local(self.index_path)
        print(f"Saved index with {self.db.index.ntotal} embeddings")

    def load_db(self):
        # Load FAISS index
        self.db = FAISS.load_local(self.index_path, self.embedding_model, allow_dangerous_deserialization=True)
        print(f"Loaded index with {self.db.index.ntotal} embeddings")
    
    def get_retriever(self):
        return self.db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
