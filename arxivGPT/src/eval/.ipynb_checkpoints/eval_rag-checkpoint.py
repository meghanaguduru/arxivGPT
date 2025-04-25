import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # or as many `..` as needed

import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import precision_score
import torch
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import jsonlines
from prompts.prompt_tuned import prompt
from llm_inference import LLMModel
from vector_db_langchain import VectorDBLangchain

# Paths and settings
eval_file_path = '../../data/eval/eval_data.jsonl'
vector_db_path = "../../data/processed/first_vector_db"
embedding_dim = 384
threshold = 0.9  # Lower distance = higher similarity for FAISS IndexFlatL2

# Use a list comprehension to read all data into a list
with jsonlines.open(eval_file_path) as reader:
    eval_data = [obj for obj in reader]
    
print(eval_data)

# Load the model for embeddings and evaluation
model = SentenceTransformer('all-MiniLM-L6-v2')  # Example, change based on your model
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Function for RAG retrieval
def retrieve_chunks(question, paper_id):
    # This should implement the chunk retrieval logic (embedding-based search etc.)
    return ["Some chunk of text related to the question and paper."]  # Dummy return

# Function to generate the answer (based on the retrieval)
def generate_answer(retrieved, question):
    # This should implement your RAG-based answer generation
    return "Generated answer based on retrieval."

# BLEU scoring
smooth_fn = SmoothingFunction().method1

# Initialize components
llm = LLMModel(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
vector_db = VectorDBLangchain(dim=embedding_dim, index_path=vector_db_path)
vector_db.load_db()
retriever = vector_db.get_retriever()
rag_chain = prompt | llm.pipeline

# Evaluation
results = []
for qa in eval_data:
    # Retrieve relevant chunks and generate the answer
    # retrieved = retrieve_chunks(qa["question"], qa["paper_id"])
    # generated = generate_answer(retrieved, qa["question"])
    # Retrieve relevant chunks using FAISS
    docs_and_scores = vector_db.similarity_search_with_score(question)
    filtered_docs = [doc for doc, score in docs_and_scores if score <= threshold]
    context = "\n\n".join([doc.page_content for doc in filtered_docs]) if filtered_docs else ""

    # Generate answer
    generated = rag_chain.invoke({"question": question, "context": context})

    # Embeddings for Recall@5
    retrieved_embeds = model.encode(retrieved, convert_to_tensor=True)
    ground_embed = model.encode(qa["ground_context"], convert_to_tensor=True)
    recall_at_5 = any(torch.cosine_similarity(ground_embed, r, dim=0).item() > 0.7 for r in retrieved_embeds)

    # ROUGE score
    rouge_score = rouge.score(generated, qa["answer"])['rougeL'].fmeasure

    # BERTScore
    _, _, f1 = bert_score([generated], [qa["answer"]], lang="en")
    bert_f1 = f1[0].item()

    # BLEU score
    reference_tokens = qa["answer"].split()
    candidate_tokens = generated.split()
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smooth_fn, weights=(0.5, 0.5))

    # Precision: Assuming we have a binary ground truth for relevance (1 for relevant, 0 for irrelevant)
    # For simplicity, let's assume precision here is simply the proportion of relevant chunks
    # returned from the RAG retrieval.
    # In your case, you could have a more sophisticated method for computing Precision.

    retrieved_text = " ".join(retrieved)  # Convert list of retrieved chunks to a string
    precision = precision_score([1], [1] if qa["ground_context"] in retrieved_text else [0], average='binary')

    # Append to results
    results.append({
        "paper_id": qa["paper_id"],
        "question": qa["question"],
        "generated": generated,
        "reference": qa["answer"],
        "recall@5": recall_at_5,
        "rougeL": round(rouge_score, 3),
        "bert_f1": round(bert_f1, 3),
        "bleu": round(bleu_score, 3),
        "precision": round(precision, 3),
    })

# Show results
for r in results:
    print(json.dumps(r, indent=2))
