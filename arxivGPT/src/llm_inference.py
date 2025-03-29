# Use Mistral-7B model via HuggingFace Transformers library

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# run a simple query
query = "What is the capital of France?"
inputs = tokenizer(query, return_tensors="pt") # return Pytorch tensor
outputs = model.generate(**inputs, max_length=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# todo : Pass user query + retrieved context to the model for inference
# todo : Implement a simple web interface for the model