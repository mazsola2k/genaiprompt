import time
import threading
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pre-trained LLaMA-2-70B Chat model
model_name = "meta-llama/Llama-2-70b-chat-hf"  # Use the 70B chat-tuned model

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define max memory for GPU and CPU
max_memory = {
    0: "24GiB",  # GPU 0 memory limit
    "cpu": "48GiB"  # CPU memory limit
}

# Load the model with CPU offloading enabled and keep modules in 32-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically map model to GPU and CPU
    max_memory=max_memory,  # Specify memory limits for GPU and CPU
    torch_dtype=torch.float32  # Keep modules in 32-bit precision
)

# Custom stopping criteria to simulate token-by-token generation
class ProgressStoppingCriteria:
    def __init__(self, max_new_tokens, start_time):
        self.max_new_tokens = max_new_tokens
        self.generated_tokens = 0
        self.start_time = start_time

    def __call__(self, input_ids, scores, **kwargs):
        self.generated_tokens += 1
        time_elapsed = time.time() - self.start_time
        estimated_total_time = (time_elapsed / self.generated_tokens) * self.max_new_tokens
        remaining_time = estimated_total_time - time_elapsed

        # Convert remaining time to MM:SS format
        minutes, seconds = divmod(int(remaining_time), 60)
        print(
            f"\rProgress: {self.generated_tokens}/{self.max_new_tokens} tokens generated. "
            f"Estimated remaining time: {minutes:02d}:{seconds:02d}.",
            end=""
        )
        sys.stdout.flush()
        return self.generated_tokens >= self.max_new_tokens

def generate_response(prompt):
    """Generate a response for the given prompt."""
    refined_prompt = f"Question: {prompt}\nAnswer:"
    inputs = tokenizer(refined_prompt, return_tensors="pt").to("cuda:0")
    max_new_tokens = 150  # Reduce the number of tokens to avoid irrelevant content
    start_time = time.time()

    # Use the custom stopping criteria
    stopping_criteria = ProgressStoppingCriteria(max_new_tokens=max_new_tokens, start_time=start_time)

    # Generate tokens with adjusted parameters
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        top_k=10,  # Reduce diversity
        top_p=0.8,  # Reduce nucleus sampling range
        early_stopping=True,
        stopping_criteria=[stopping_criteria]
    )

    # Decode the generated tokens and extract the answer
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[1].strip()
    else:
        answer = generated_text.strip()
    print(f"\n\nGenerated Output:\n{answer}\n")

def cli_loop():
    """Continuously wait for user input and generate responses."""
    print("Welcome to the GenAI CLI!")
    print("Type your prompt below and press Enter. Type 'exit' to quit.\n")

    while True:
        prompt = input("Enter your prompt: ")
        if prompt.lower() == "exit":
            print("Exiting GenAI CLI. Goodbye!")
            break

        # Generate the response in a separate thread for responsiveness
        generate_thread = threading.Thread(target=generate_response, args=(prompt,))
        generate_thread.start()
        generate_thread.join()

if __name__ == "__main__":
    cli_loop()