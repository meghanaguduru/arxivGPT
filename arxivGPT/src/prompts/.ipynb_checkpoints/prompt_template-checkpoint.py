from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables = ["context", "question"],
    template = (
        "You are an AI assistant answering questions based on research papers. \n\n"
        "Context:\n {context} \n\n"
        "Question: {question} \n"
        "Answer:"
    )
)