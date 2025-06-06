
"""
+-------------------------------------------------------------+
|                  Ollama Llama4 CLI Chatbot                  |
+-------------------------------------------------------------+
| 1. On start, check if 'llama4' model exists in Ollama.      |
|    - If not, pull it from Ollama Hub and show progress.     |
| 2. Print welcome message.                                   |
| 3. Loop:                                                    |
|    a. Prompt user for input.                                |
|    b. If input is 'exit', quit.                             |
|    c. Else, send prompt to Ollama API, stream response.     |
|    d. Print tokens and elapsed time as response streams in. |
|    e. Show full response at the end.                        |
+-------------------------------------------------------------+
"""

import requests
import json
import time

OLLAMA_MODEL = "llama4"
OLLAMA_URL = "http://localhost:11434"

def ensure_model(model):
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json().get("models", [])
        if any(m.get("name", "").split(":")[0] == model for m in tags):
            return True
    except Exception as e:
        print(f"Model check failed: {e}")
        return False
    print(f"Pulling model '{model}' from Ollama Hub...")
    resp = requests.post(f"{OLLAMA_URL}/api/pull", json={"name": model}, stream=True)
    for line in resp.iter_lines():
        if line:
            try:
                msg = json.loads(line.decode())
                print(f"\r{msg.get('status') or msg.get('digest') or msg}", end="", flush=True)
            except Exception:
                continue
    print("\nModel pull complete.")
    return True

def generate_response(prompt):
    url = f"{OLLAMA_URL}/api/generate"
    data = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}
    print("Generating response...\n")
    output, token_count = "", 0
    start = time.time()
    with requests.post(url, json=data, stream=True) as resp:
        if resp.status_code != 200:
            print(f"Error: Ollama API {resp.status_code}\n{resp.text}")
            return
        for line in resp.iter_lines():
            if line:
                try:
                    token = json.loads(line.decode()).get("response", "")
                except Exception:
                    continue
                output += token
                token_count += 1
                elapsed = time.time() - start
                print(f"\rTokens: {token_count} | Elapsed: {elapsed:.1f}s", end="")
    print("\n\n" + output.strip() + "\n")

if __name__ == "__main__":
    print("Ollama Llama4 CLI. Type your prompt and press Enter. Type 'exit' to quit.\n")
    if ensure_model(OLLAMA_MODEL):
        while True:
            prompt = input("Prompt: ")
            if prompt.strip().lower() == "exit":
                break
            generate_response(prompt)