"""
Predictive CLI for Ollama Llama4 Model
This script provides a command-line interface to interact with the Ollama Llama4 model.
It allows users to input prompts and receive generated responses in real-time.

Before running this script, ensure that:
1. Ollama is installed and running on your machine.
After Mac and Windows, available now nn Windows - you can install it via: https://ollama.com/download
2. Check that the Llama4 model is available in your Ollama setup. The below script will attempt to pull the model if it is not found.
3. The Ollama server is accessible at http://localhost:11434 (default port) and checks model name availability.
4. Prompts provided to the model
"""

import requests
import time
import threading
import sys

OLLAMA_MODEL = "llama4"  # The model name as recognized by Ollama
OLLAMA_URL = "http://localhost:11434"

def validate_ollama_and_model():
    """Check if Ollama server is running and the model is available."""
    try:
        # Check if Ollama server is up
        tags_resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        tags_resp.raise_for_status()
        tags = tags_resp.json()
        available_models = [m["name"] for m in tags.get("models", [])]
        if OLLAMA_MODEL not in available_models:
            print(f"Model '{OLLAMA_MODEL}' not found in Ollama. Pulling model...")
            pull_resp = requests.post(f"{OLLAMA_URL}/api/pull", json={"name": OLLAMA_MODEL}, stream=True)
            for line in pull_resp.iter_lines():
                if line:
                    print(line.decode("utf-8"))
            print(f"Model '{OLLAMA_MODEL}' pulled successfully.")
        else:
            print(f"Model '{OLLAMA_MODEL}' is available in Ollama.")
        return True
    except Exception as e:
        print(f"Could not connect to Ollama server or fetch models: {e}")
        print("Make sure Ollama is running (`ollama serve`) and accessible at http://localhost:11434")
        return False

def generate_response(prompt):
    """Generate a response for the given prompt using Ollama."""
    url = f"{OLLAMA_URL}/api/generate"
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True
    }

    print("Generating response...\n")
    start_time = time.time()
    output = ""
    token_count = 0

    with requests.post(url, json=data, stream=True) as response:
        for line in response.iter_lines():
            if line:
                try:
                    chunk = line.decode("utf-8")
                    # Ollama streams JSON objects per line
                    import json
                    obj = json.loads(chunk)
                    token = obj.get("response", "")
                    output += token
                    token_count += 1
                    elapsed = time.time() - start_time
                    print(
                        f"\rTokens received: {token_count} | Elapsed: {elapsed:.1f}s",
                        end=""
                    )
                    sys.stdout.flush()
                except Exception as e:
                    print(f"\n[Error parsing chunk: {e}]")
    print("\n\nGenerated Output:\n" + output.strip() + "\n")

def cli_loop():
    """Continuously wait for user input and generate responses."""
    print("Welcome to the GenAI CLI (Ollama Llama4)!")
    print("Type your prompt below and press Enter. Type 'exit' to quit.\n")

    if not validate_ollama_and_model():
        return

    while True:
        prompt = input("Enter your prompt: ")
        if prompt.lower() == "exit":
            print("Exiting GenAI CLI. Goodbye!")
            break

        generate_thread = threading.Thread(target=generate_response, args=(prompt,))
        generate_thread.start()
        generate_thread.join()

if __name__ == "__main__":
    cli_loop()