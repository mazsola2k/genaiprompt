#https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.ggu

#pip install llama-cpp-python

from llama_cpp import Llama
import sys
import time

llm = Llama(
    model_path="./meta-llama-3-8b.Q4_K_M.gguf",  # Update to your downloaded file
    n_ctx=4096,
    n_gpu_layers=-1,  # Full GPU offload
    verbose=False,
)

while True:
    prompt = input("\nEnter your prompt (or type 'exit' to quit):\n> ")
    if prompt.strip().lower() == "exit":
        break

    print("Calculating response...", end="", flush=True)
    start = time.time()
    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
    )
    elapsed = time.time() - start
    print(f"\rResponse ready in {elapsed:.1f} seconds.\n")
    print(output["choices"][0]["text"].strip())