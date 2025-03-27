import faiss
from sentence_transformers import SentenceTransformer

class vectorDB:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatL2(self.dim)

    def add(self, embeddings):
        self.index.add(embeddings)
        print(self.index.ntotal)

    def retrieve(self, query, k):
        model_name = 'all-MiniLM-L6-v2' #22M param sentence embedding model, embeddings are 384 dim
        model = SentenceTransformer(model_name)
        query_embedding = model.encode(query)
        return self.index.search(query_embedding, k)
    
    def save(self, embeddings, file_path):
        faiss.write_index(self.index, file_path)

    def load(self, file_path):
        faiss.read_index(file_path)


