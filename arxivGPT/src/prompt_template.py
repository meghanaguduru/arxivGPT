# unused because decided to use Langchain which internally gives a prompt to LLM
from langchain.prompts import PromptTemplate

rag_prompt = PromptTemplate(
    input_variables = ["context", "question"],
    template = (
        "You are an AI assistant answering questions based on research papers. \n"
        "Context:\n {context} \n"
        "Question: {question} \n"
        "Answer:"
    )
)