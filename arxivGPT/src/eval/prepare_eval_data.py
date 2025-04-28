import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # or as many `..` as needed

import jsonlines
import os

from preprocess import PDFTextExtractor
from llm_inference import LLMModel
from vector_db_langchain import VectorDBLangchain
from prompts.prompt_tuned import prompt
from google import genai

# Settings
vector_db_path = "../../data/processed/first_vector_db"
output_path = "../../data/evaluation/eval_data.jsonl"
embedding_dim = 384
num_sentences_to_use = 50  # Adjust as needed

# Init LLM and vector DB
llm = LLMModel(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0") # Use a larger model via API
vector_db = VectorDBLangchain(dim=embedding_dim, index_path=vector_db_path)
vector_db.load_db()

# Google API to use larger models
client = genai.Client(api_key="AIzaSyCkWPHToVcuhqiaoTJrKIK8NtI5WL3fkTw")

# Get all stored docs
all_docs = vector_db.db.docstore._dict.values()
sentences = [doc.page_content for doc in all_docs]

# Limit to a subset (or shuffle and take a sample if needed)
sentences = sentences[:num_sentences_to_use]

# Output file
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with jsonlines.open(output_path, mode='w') as writer:
    # TODO : Make i = paper number by adding paper number in metadata
    for i, sentence in enumerate(sentences):
        try:
            # Prompt the LLM to generate a QA pair
            qa_prompt = (
                "You are given a sentence from a research paper.\n"
                "Generate a question someone might ask based on it, "
                "and also provide a brief, correct answer using only the given sentence.\n\n"
                f"Sentence: {sentence}\n\n"
                "Question:"
            )

            # Get full QA text
            # qa_response = client.models.generate_content(                 #40B param model
            #     model="gemini-2.0-flash", contents=qa_prompt 
            # )
            # print(qa_response.text)
            qa_response = llm.pipeline.invoke(qa_prompt)

            # Split into question and answer
            if "Answer:" in qa_response:
                question_part, answer = qa_response.split("Answer:", 1)
                question = question_part.replace("Question:", "").strip()
                answer = answer.strip()
            else:
                print(f"[Warning] Skipping badly formatted response:\n{qa_response}")
                continue

            # Write to file
            writer.write({
                "paper_id": f"{i}",
                "question": question,
                "answer": answer,
                "ground_context": sentence
            })

            print(f"[✓] Generated QA pair #{i + 1}")
        except Exception as e:
            print(f"[Error] QA generation failed for sentence #{i + 1}: {e}")
