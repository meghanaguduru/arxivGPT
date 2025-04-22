from preprocess import PDFTextExtractor
from llm_inference import LLMModel
from vector_db import vectorDB
from vector_db_langchain import VectorDBLangchain
from prompt_template import rag_prompt
import os

embeddings_dim = 384 # hardcoded default (temp hack)
sentences = None

if __name__ == "__main__":
    vector_db_path = "../data/processed/first_vector_db"

    # Initialize LLM
    # model_name = "mistralai/Mistral-7B-v0.1"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    llm = LLMModel(model_name=model_name)
    vector_db = VectorDBLangchain(embeddings_dim, index_path=vector_db_path)

    pdf_extractor = PDFTextExtractor("../data")
    if not os.path.exists(vector_db_path):
        sentences, embeddings = pdf_extractor.process_pdfs(pdf_extractor.input_dir)
        vector_db.create_db(embeddings, sentences)
    else:
        print("Loading existing vector database...")
        vector_db.load_db()

    # Create a retrieval chain
    retriever = vector_db.get_retriever()
    # Query
    query = "How do birds fly?"
    docs_and_scores = vector_db.similarity_search_with_score(query)

    for doc, score in docs_and_scores:
        print("Score: ", score)
        print("Content : ", doc.page_content)
        
    threshold = 0.5 # FAISS IndexFlatL2 scores -> the lower the score, closer the embedding
    filtered_docs = [doc for doc, score in docs_and_scores if score <= threshold]
    
    context = "\n".join([doc.page_content for doc in filtered_docs])

    rag_chain = rag_prompt | llm.pipeline
    # Get answer from LLM
    response = rag_chain.invoke({"question": query, "context": context})
    
    print("[RAG based Answer:]", response)
    
    # Raw LLM answer
    raw_prompt = (
        "You are an AI assistant answering questions.\n"
        "Answer the following question briefly and accurately.\n"
        f"Question: {query} \n"
        "Answer: "
    )
    raw_response = llm.pipeline.invoke(raw_prompt)
    print("[Raw LLM Answer:]", raw_response)
  