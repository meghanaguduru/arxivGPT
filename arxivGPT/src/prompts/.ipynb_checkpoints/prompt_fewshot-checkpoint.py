from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an AI assistant helping summarize and understand NLP research papers.\n\n"
        
        "Example 1:\n"
        "Context:\n"
        "Large Language Models (LLMs) such as GPT-3 have shown remarkable abilities in few-shot learning. "
        "However, they are prone to hallucinations and struggle with long context retention.\n\n"
        "Question: What are the limitations of LLMs discussed in the paper?\n"
        "Answer: The paper highlights that LLMs can hallucinate facts and have difficulty retaining long-range dependencies.\n\n"
        
        "Example 2:\n"
        "Context:\n"
        "This work introduces a dataset for instruction tuning in multiple languages, enabling models to follow prompts across diverse linguistic contexts.\n\n"
        "Question: What is the main contribution of the paper?\n"
        "Answer: The main contribution is a multilingual instruction-tuning dataset that improves prompt-following performance across languages.\n\n"
        
        "Example 3:\n"
        "Context:\n"
        "The proposed method leverages reinforcement learning from human feedback (RLHF) to align model behavior with human preferences, using a reward model trained on ranking comparisons.\n\n"
        "Question: How is reinforcement learning used in this approach?\n"
        "Answer: The model is fine-tuned using RLHF, where a reward model guides the learning process based on human preferences.\n\n"
        
        "Now your turn:\n\n"
        "Context:\n"
        "{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
)
