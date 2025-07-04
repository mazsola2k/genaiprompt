#this code runs well from RTX 3090 and 4090 GPUs 24GB RAM

# Import required libraries - you can use the setup_env.py script to install the required libraries
#https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
#>nvcc --version
#Cuda compilation tools, release 12.8, V12.8.93
#Build cuda_12.8.r12.8/compiler.35583870_0
#https://pytorch.org/get-started/locally/#windows-pip
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#pip install accelerate  huggingface-hub
#huggingface-cli login

import os

# Use D:/HF_CACHE if D: exists, else use C:/HF_CACHE
cache_drive = "D" if os.path.exists("D:/") else "C"
cache_dir = f"{cache_drive}:/HF_CACHE"
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir  # Use HF_HOME as TRANSFORMERS_CACHE is deprecated

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Prompt for Hugging Face token
hf_token = input("Enter your Hugging Face token (leave blank to use default login): ").strip() or None

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, token=hf_token).to(device)

while True:
    prompt = input("\nEnter your prompt (or type 'exit' to quit):\n> ")
    if prompt.strip().lower() == "exit":
        break

    inputs = tokenizer(f"Question: {prompt}\nAnswer:", return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        top_p=0.8,
        top_k=10,
        temperature=0.7,
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = generated_text.split("Answer:")[-1].strip() if "Answer:" in generated_text else generated_text.strip()
    print("\n" + answer)