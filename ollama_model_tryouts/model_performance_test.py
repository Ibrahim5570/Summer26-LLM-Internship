import ollama
import time

model_to_test = "tinyllama:1.1b"  # Change this to the model you want to test
prompt = "Explain the significance of remote sensing in urban planning in 3 sentences."

print(f"Testing performance for: {model_to_test}...")

start_time = time.time()
response = ollama.generate(model=model_to_test, prompt=prompt)
end_time = time.time()

total_duration = end_time - start_time
eval_count = response['eval_count'] # Total tokens generated
tokens_per_second = eval_count / total_duration

print(f"\n--- {model_to_test} Results ---")
print(f"Response: {response['response']}\n")
print(f"Total Time Taken: {total_duration:.2f} seconds")
print(f"Tokens Generated: {eval_count}")
print(f"Generation Speed: {tokens_per_second:.2f} tokens/sec")