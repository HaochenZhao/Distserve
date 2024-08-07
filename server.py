import socket
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_server():
    # Load the smaller model and tokenizer
    model_id = 'facebook/opt-125m'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    # Setup socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 4444))
    server_socket.listen(1)
    print("Server is listening on 127.0.0.1:4444")
    
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            
            prompt = data.decode('utf-8')
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(inputs["input_ids"], num_return_sequences=1)
            
            response_tokens = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            conn.sendall(str(response_tokens).encode('utf-8'))

if __name__ == "__main__":
    run_server()
