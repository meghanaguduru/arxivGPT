from preprocess import PDFTextExtractor
from llm_inference import LLMModel
from langchain.chains import create_retrieval_chain, LLMChain
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
    query = "Explain transformer architecture"
    docs = retriever.get_relevant_documents(query)
    print("Retrieved documents:", docs)
    context = "\n".join([doc.page_content for doc in docs])

    for doc in docs:
        print("Document:", doc.page_content)

    rag_chain = LLMChain(llm = llm.pipeline, prompt = rag_prompt)
    # Get answer from LLM
    response = rag_chain.invoke({"question": query, "context": context})
    
    # retrieval_chain = create_retrieval_chain(retriever, llm.pipeline)
    # response = retrieval_chain.invoke({"input": query, "context": context})  
    
    print("Answer:", response)
  