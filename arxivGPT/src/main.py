from preprocess import PDFTextExtractor
from vector_db import vectorDB
import os

embeddings_dim = 384 # hardcoded default (temp hack)
sentences = None

if __name__ == "__main__":
    vector_db_path = "../data/processed/first_vector_db.index"
    if not os.path.exists(vector_db_path):
        pdf_extractor = PDFTextExtractor("../data")
        sentences, embeddings = pdf_extractor.process_pdfs(pdf_extractor.input_dir)

        embeddings_dim = embeddings.shape[1]
        print(f"Embeddings shape: {embeddings.shape}")

        vector_db = vectorDB(embeddings_dim)
        vector_db.add(embeddings)
        vector_db.save(vector_db_path)
        print(f"Saved index with {vector_db.index.ntotal} embeddings")
    else:  
        print(f"Loading index from {vector_db_path}, embeddings dim : {embeddings_dim}")
        vector_db = vectorDB(embeddings_dim)
        vector_db.load(vector_db_path)
        print(f"Loaded index with {vector_db.index.ntotal} embeddings")
        print(vector_db.index.is_trained)

    # Retrieve similar documents
    distances, indices = vector_db.retrieve("deep learning in nlp", 5)  
    # Smaller distances are closer matches for IndexFlatIP
    print(f"Distances : {distances}, indices : {indices}")
    if sentences : 
        for index in indices[0]:
            print(sentences[index]+" \n")
    else:
        print("No sentences stored in the database")