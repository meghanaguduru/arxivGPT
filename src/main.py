from preprocess import PDFTextExtractor
from vector_db import vectorDB

if __name__ == "__main__":
    pdf_extractor = PDFTextExtractor("../data")
    embeddings = pdf_extractor.process_pdfs(pdf_extractor.input_dir)

    embeddings_dim = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape}")

    vector_db = vectorDB(embeddings_dim)
    vector_db.add(embeddings)
    vector_db.retrieve("deep learning", 2)
    
    vector_db.save(embeddings, "data/processed/first_vector_db.index")
    
