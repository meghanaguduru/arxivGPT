import os
import fitz as pymupdf # fitz is the PyMuPDF library older versions
import re
import nltk

nltk.download('punkt')

class PDFTextExtractor:
    def __init__(self, input_dir= "../data/", output_dir="data/processed/"):
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
    def chunk_text(self, text, chunk_size=524, stride=256):
        sentences = nltk.sent_tokenize(text)
        return sentences[0]

    def process_pdfs(self, pdf_dir):
        """
        Extract text from all PDFs in a directory and save as .txt files
        """
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            text = self.extract_text(pdf_path)
            # cleanup
            text = self.clean_text(text)
            # chunk text into sections
            text = self.chunk_text(text)
            print(text)
            # save to output_dir
        

if __name__ == "__main__":
    pdf_extractor = PDFTextExtractor("../data")
    pdf_extractor.process_pdfs(pdf_extractor.input_dir)