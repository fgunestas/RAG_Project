import os
from dotenv import load_dotenv

import warnings

warnings.filterwarnings("ignore")

load_dotenv()

auth_token = os.getenv("YOUR_HUGGING_FACE_API_KEY")



model_name = "EleutherAI/gpt-neo-1.3B"

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B",token="auth_token")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B",token="auth_token")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Kullanıcı sorgusu

input_text = "What is recursion in programming?"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

outputs = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=50,
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))