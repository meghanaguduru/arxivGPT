from langchain.prompts import PromptTemplate

# Prompt tuning for a more refined answer
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a HIGHLY SKILLED AI ASSISTANT with EXPERTISE IN RESEARCH PAPERS. "
        "Your job is to EXTRACT INFORMATION from academic research papers to answer specific questions. \n"
        "Your task is to provide CONCISE, ACCURATE ANSWERS based EXCLUSIVELY on the RELEVANT EXCERPTS from VARIOUS RESEARCH PAPERS provided below. "
        "The context consists of DISCONNECTED, PIECEWISE INFORMATION from MULTIPLE PAPERS. "
        "Ensure that your answers are GROUNDED in the FINDINGS or METHODOLOGIES of the papers, WITHOUT adding any PERSONAL INTERPRETATION or EXTRA DETAILS.\n\n"
        "Context: {context} \n\n"
        "Question: {question} \n"
        "Answer:"
    )
)