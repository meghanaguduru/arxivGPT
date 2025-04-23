from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an AI assistant summarizing and analyzing research papers step-by-step.\n"
        "Context:\n{context}\n"
        "Question: {question}\n"
        "Let's think step by step.\n"
        "Answer:"
    )
)