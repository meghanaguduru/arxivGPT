import os
import fitz as pymupdf # fitz is the PyMuPDF library older versions
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np

class PDFTextExtractor:
    model_name = 'all-MiniLM-L6-v2' #22M param sentence embedding model, embeddings are 384 dim
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/"+model_name)
    model = SentenceTransformer(model_name)
    # Max token limit for MiniLM
    MAX_TOKENS = 256
    
    def __init__(self, input_dir= "../data/input_papers", output_dir="../data/processed/"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_text(self, pdf_path):
        doc = pymupdf.open(pdf_path)
        extracted_text = []
        for page in doc:
            extracted_text.append(page.get_text())
        full_extracted_text = "\n".join(extracted_text)
        return full_extracted_text

    def clean_text(self, text):
        # remove special characters
        # urls not managed too well
        text = re.sub(r"[^a-zA-Z0-9.]", " ", text)
        # make lowercase
        text = text.lower()
        # remove extra spaces
        text = re.sub(r"\s+", " ", text)
        # some journal numbers mishandled but moving on as those may not be v. important
        return text

    '''
    # chunking should balance retrieval accuracy and computational efficiency
    def chunk_text(self, text, chunk_size=256, overlap=30):
        # basic chunking with overlap (fixed size)
        # todo : explore semantic chunking
        chunks = []
        for i in range(0, len(text), chunk_size-overlap):
            chunks.append(text[i: i+chunk_size])
        return chunks
    '''
    
    def chunk_text(self, text):
        # Split text into chunks by paragraph
        # If paragraph is > MAX_TOKEN, cut it
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        final_chunks = []
        
        for para in paragraphs:
            tokenized = tokenizer.encode(para, add_special_tokens = False)
            if len(tokenized) <= MAX_TOKENS:
                final_chunks.append(para)
            else:
                # Fallback : Split by sentences or brute force token chunks
                words = para.split()
                chunk = []
                curr_len = 0
                
                for word in words:
                    tokens = tokenizer.encode(word, add_special_tokens = False)
                    if curr_len + len(tokens) > MAX_TOKENS:
                        final_chunks.append(" ".join(chunk))
                        chunk = []
                        tokens = []
                        curr_len = 0
                    chunk.append(word)
                    curr_len += len(tokens)
                    
                if chunk:
                    final_chunks.append(" ".join(chunk))
        print(final_chunks + "\n")
        return final_chunks
    
    def text_embeddings(self, chunks):
        embeddings = model.encode(chunks)   
        return embeddings

    def process_pdfs(self, pdf_dir):
        """
        Extract text from all PDFs in a directory and save as .txt files
        """
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        all_embeddings = [] # Collect all embeddings
        all_chunks = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            text = self.extract_text(pdf_path)
            # cleanup
            text = self.clean_text(text)
            # chunk text into sections
            chunks = self.chunk_text(text)
            # print(f"Chunks : {chunks}")
            all_chunks.extend(chunks)
            # get embeddings
            # takes sometime, even though we use an ultrafast model
            embeddings = self.text_embeddings(chunks)
            all_embeddings.append(embeddings)
            # save to output_dir
        all_embeddings = np.vstack(all_embeddings)
        print(f"Final embeddings shape {all_embeddings.shape}")
        print(f"Total chunks : {len(all_chunks)}")
        return all_chunks, np.array(all_embeddings)