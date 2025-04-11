
from llm_inference import LLMModel
from vector_db_langchain import VectorDBLangchain
from preprocess import PDFTextExtractor
from langchain.chains import LLMChain
import os
from prompt_template import rag_prompt


class RAGChatbot:
    def __init__(self, pdf_dir:str, vector_db_path:str, model_name:str):
        self.vector_db_path = vector_db_path
        self.embeddings_dim = 384 # hardcoded default (temp hack)
        self.llm = LLMModel(model_name=model_name)
        self.vector_db = VectorDBLangchain(self.embeddings_dim, index_path=vector_db_path)
        self.pdf_extractor = PDFTextExtractor(pdf_dir)

        if not os.path.exists(vector_db_path):
            sentences, embeddings = self.pdf_extractor.process_pdfs(self.pdf_extractor.input_dir)
            self.vector_db.create_db(embeddings, sentences)
        else:
            print("Loading existing vector database...")
            self.vector_db.load_db()

        self.retriever = self.vector_db.get_retriever()
        self.rag_chain = LLMChain(llm=self.llm.pipeline, prompt=rag_prompt)

    def query(self, user_question:str) -> str:
        docs = self.retriever.get_relevant_documents(user_question)
        context = "\n".join([doc.page_content for doc in docs])
        response = self.rag_chain.invoke({"question": user_question, "context": context})
        return response
