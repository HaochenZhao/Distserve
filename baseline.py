import socket
import torch
import time
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

def edge(prompt):
    # Load the smaller model and tokenizer
    model_id = 'facebook/opt-125m'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], num_return_sequences=1)
            
    response_tokens = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    # data center network latency
    # torch.distributed.barrier()
    time.sleep(random.uniform(0.01, 0.05))
            
    # for debugging
    print(f"Received: {prompt}")
    print(f"Response: {response_tokens}")
    print("--------------------")
    
    return response_tokens

def center():
    # Load the larger model and tokenizer
    model_id = 'facebook/opt-1.3b'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()


    def speculative_decoding(prompt):
        
        predictions = edge(prompt)

        # Validate predictions with the larger model
        inputs = tokenizer(prompt, return_tensors="pt")
        best_response = None
        best_score = float('-inf')
        
        for prediction in predictions:
            extended_input = tokenizer.encode(prompt + prediction, return_tensors="pt")
            with torch.no_grad():
                outputs = model(extended_input)
                score = outputs.logits[:, -1, :].max()
                if score > best_score:
                    best_score = score
                    best_response = prediction

        return best_response

    while True:
        prompt = input("Enter your prompt: ")
        if prompt.lower() == 'exit':
            break
        # user-center latency
        time.sleep(random.uniform(0.1, 0.15))
        response = speculative_decoding(prompt)
        time.sleep(random.uniform(0.1, 0.015))
        print(f"Response: {response}")
        
if __name__ == "__main__":
    # measure time spent
    start = time.time()
    center()
    end = time.time()
    print(f"Time spent: {end - start}")