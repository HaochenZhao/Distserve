import socket
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_client():
    # Load the larger model and tokenizer
    model_id = 'facebook/opt-1.3b'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 4444))

    def speculative_decoding(prompt):
        client_socket.sendall(prompt.encode('utf-8'))
        data = client_socket.recv(4096)
        predictions = eval(data.decode('utf-8'))

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
        response = speculative_decoding(prompt)
        print(f"Response: {response}")

if __name__ == "__main__":
    run_client()
