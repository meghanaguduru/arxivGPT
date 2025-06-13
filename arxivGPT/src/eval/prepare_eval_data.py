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
api_key = os.getenv("GOOGLE_API_KEY")
vector_db_path = "../../data/processed/first_vector_db"
output_path = "../../data/evaluation/eval_data.jsonl"
embedding_dim = 384
num_sentences_to_use = 50  # Adjust as needed

# Init LLM and vector DB
llm = LLMModel(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0") # Use a larger model via API
vector_db = VectorDBLangchain(dim=embedding_dim, index_path=vector_db_path)
vector_db.load_db()

# Google API to use larger models
client = genai.Client(api_key=api_key)

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
                "and provide a brief, correct answer using ONLY the given sentence.\n\n"
                f"Sentence: {sentence}\n\n"
                "Question:"
            )

            # Get full QA text
            # qa_response = client.models.generate_content(                 #40B param model
            #     model="gemini-2.0-flash", contents=qa_prompt 
            # )
            # print(qa_response.text)
            qa_response = llm.pipeline.invoke(qa_prompt)

            # Clean prompt leakage if it repeats the full prompt
            if "Sentence:" in qa_response and "Question:" in qa_response:
                qa_response = qa_response.split("Sentence:", 1)[-1].strip()

            # Split properly into question and answer
            if "Question:" in qa_response and "Answer:" in qa_response:
                print(f"Writing data sample from response")
                question_part = qa_response.split("Question:", 1)[-1].split("Answer:")[0].strip()
                answer_part = qa_response.split("Answer:", 1)[-1].strip()
                question = question_part
                answer = answer_part
            else:
                print(f"[Warning] Skipping badly formatted response:\n{qa_response}")
                continue
                
            print(f"""
                paper_id: {i},
                question: {question},
                answer: {answer},
                ground_context: {sentence}
            """)

            # Write to file
            writer.write({
                "paper_id": f"{i}",
                "question": question,
                "answer": answer,
                "ground_context": sentence
            })
p
            print(f"[✓] Generated QA pair #{i + 1}")
        except Exception as e:
            print(f"[Error] QA generation failed for sentence #{i + 1}: {e}")
