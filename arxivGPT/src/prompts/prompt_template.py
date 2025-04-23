from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables = ["context", "question"],
    template = (
        "You are an AI assistant answering questions based on research papers. \n"
        "Context:\n {context} \n"
        "Question: {question} \n"
        "Answer:"
    )
)