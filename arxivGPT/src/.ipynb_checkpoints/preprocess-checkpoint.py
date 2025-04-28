import os
import fitz as pymupdf # fitz is the PyMuPDF library older versions
import re
from sentence_transformers import SentenceTransformer
import numpy as np

class PDFTextExtractor:
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

    # chunking should balance retrieval accuracy and computational efficiency
    def chunk_text(self, text, chunk_size=256, overlap=30):
        # basic chunking with overlap (fixed size)
        # todo : explore semantic chunking
        chunks = []
        for i in range(0, len(text), chunk_size-overlap):
            chunks.append(text[i: i+chunk_size])
        return chunks
    
    def text_embeddings(self, chunks):
        model_name = 'all-MiniLM-L6-v2' #22M param sentence embedding model, embeddings are 384 dim
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks)
        return embeddings

    def process_pdfs(self, pdf_dir):
        """
        Extract text from all PDFs in a directory and save as .txt files
        """
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        all_embeddings = [] # Collect all embeddings
        all_chunks = []
        # i = 0
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            text = self.extract_text(pdf_path)
            # cleanup
            text = self.clean_text(text)
            # chunk text into sections
            chunks = self.chunk_text(text)
            # i = i + 1
            # print("Printing all chunks to gen GT")
            # print(f"count : {i}")
            # print(f"Chunks : {chunks}")
            all_chunks.extend(chunks)
            print(len(chunks))
            # get embeddings
            # takes sometime, even though we use an ultrafast model
            embeddings = self.text_embeddings(chunks)
            all_embeddings.append(embeddings)
            # save to output_dir
        all_embeddings = np.vstack(all_embeddings)
        print(f"Final embeddings shape {all_embeddings.shape}")
        print(f"Total chunks : {len(all_chunks)}")
        return all_chunks, np.array(all_embeddings)