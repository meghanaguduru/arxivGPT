from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an AI assistant summarizing and analyzing research papers step-by-step.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Let's think step by step.\n"
        "Answer:"
    )
)