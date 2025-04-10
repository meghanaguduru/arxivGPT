# Use Mistral-7B model via HuggingFace Transformers library

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline


class LLMModel:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        hf_pipeline = pipeline(task="text-generation",model=self.model, tokenizer=self.tokenizer)
        self.pipeline = HuggingFacePipeline(pipeline=hf_pipeline)