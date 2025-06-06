"""
https://huggingface.co/mradermacher/Llama-4-Scout-17B-6E-Instruct-i1-GGUF/resolve/main/Llama-4-Scout-17B-6E-Instruct.i1-Q4_K_S.gguf
Place to your project folder the gguf file.

For llama4 you have to upgrade/patch and re-compile the llama_cpp to handle the new GGUF format.

Pre-req for Windows11 Environment - Python 3.12:
🔧  Ensure Visual Studio Build Tools are Properly Installed
Make sure you’ve installed:

✅ MSVC v143 - VS 2022 C++ x64/x86 build tools
✅ C++ CMake tools for Windows
✅ Windows 10 SDK

After you can run the following command to install the required libraries:

pip install --no-cache-dir --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121/
or
pip install llama-cpp-python --force-reinstall --no-cache-dir --prefer-binary --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2
"""

from llama_cpp import Llama
import time

llm = Llama(
    model_path="./Llama-4-Scout-17B-6E-Instruct.i1-Q4_K_S.gguf",  # Update path!
    n_ctx=4096,
    n_gpu_layers=-1,  # Enable full GPU offload 
    verbose=False
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