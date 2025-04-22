from langchain.prompts import PromptTemplate

# Prompt tuning for a more refined answer
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a highly skilled AI assistant with expertise in research papers. "
        "Your job is to extract information from academic research papers to answer specific questions. "
        "Answer the question based on the context from the paper. Make sure to refer directly to the paper's findings or methodology.\n"
        "Context: {context} \n"
        "Question: {question} \n"
        "Answer:"
    )
)