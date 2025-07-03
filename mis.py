from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
