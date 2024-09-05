import sys
import time
import random
import argparse
import torch
import socket
from tqdm import tqdm


class Client:
    def __init__(self):
        self.client2edge_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server2client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server2client_socket.bind(('127.0.0.1', 4003))
        self.server2client_socket.listen(1)
    def accept_server2client(self):
        self.connection,addr = self.server2client_socket.accept()

def run_client():

    client = Client()
    client.client2edge_socket.connect(('127.0.0.1', 4009))
    print("client2edge connected, listening to server2client")
    client.conn,_ = client.server2client_socket.accept()
    
    while True:
        prompt = input("Enter your prompt: ")
        if prompt.lower() == 'exit':
            break

        client.client2edge_socket.sendall(prompt.encode('utf-8'))
            
        answer = client.conn.recv(102400)
        if not answer: break
        answer = answer.decode('utf-8')
        print(answer)

if __name__ == "__main__":
    run_client()


