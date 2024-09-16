import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'

import sys
import time
import random
import argparse
import torch
import socket
from tqdm import tqdm
import pickle
import struct
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from util import *

model_name_or_path = "../models/opt-1.3B"

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--max_new_tokens', type=int, default=128, help='max new tokens number')
parser.add_argument('--target_model', default=model_name_or_path, help='Target model')
parser.add_argument('--temperature', default=0.5, type=float, help='Temperature')
args = parser.parse_args()


class Target_model:
    def __init__(self, model_name_or_path, temperature):
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", offload_folder="../offload")
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.logits_processor = prepare_logits_processor(temperature=temperature)

    def sampling(self, input_ids, draft_tokens, token_indices, retrieve_indices, tree_position_ids):

        # draft_tokens.shape = (1,33)
        # retrieve_indices.shape = (15,7)
        # token_indices.shape = (33,7)
        # tree_position_ids.shape = (33)

        input_ids = input_ids[:, :-1]
        input_ids = input_ids.to(self.device)
        draft_tokens = draft_tokens.to(input_ids.device)

        col_indices = torch.arange(token_indices.shape[1]).unsqueeze(0)
        tree_mask = col_indices <= tree_position_ids.unsqueeze(-1)

        padding = torch.tensor([[self.config.pad_token_id]]).to(input_ids.device)
        new_tokens = torch.cat((draft_tokens[0],padding[0]))[token_indices]
        tmp_input_ids = input_ids.repeat(draft_tokens.shape[1], 1)
        tmp_input_ids = torch.cat((tmp_input_ids, new_tokens), dim=1)
        tmp_mask = torch.ones((draft_tokens.shape[1],input_ids.shape[1]), dtype=torch.bool).to(input_ids.device)
        tmp_mask = torch.cat((tmp_mask, tree_mask), dim=1)
        tmp_mask = tmp_mask.to(torch.int)

        output = self.model(input_ids=tmp_input_ids, attention_mask=tmp_mask)
        # 下三角因果mask ‘与’ tree_mask

        logits = output.logits
        last_idx = (input_ids.shape[-1] + tree_position_ids).view(-1, 1, 1)
        logits = torch.take_along_dim(logits, last_idx, 1)
        logits = logits.reshape(draft_tokens.shape[1], -1)
        logits = logits[retrieve_indices]
        # 修改为最后一个有效的token

        draft_tokens = torch.cat((draft_tokens, padding), dim=1)
        candidates = draft_tokens[0, retrieve_indices]

        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0

        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = self.logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == self.config.pad_token_id:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True

        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)

        input_ids = torch.cat(
            [input_ids, candidates[None, best_candidate, :accept_length].to(input_ids.device)], dim=-1
        )
        token = torch.argmax(sample_p)
        token = token[None, None]
        input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

        # print(f"accept_length {accept_length}")
        # print(candidates[None, best_candidate,:accept_length])
        # tokenizer = AutoTokenizer.from_pretrained("../models/opt-1.3B")
        # print(tokenizer.decode(candidates[None, best_candidate,:accept_length][0]))
        return input_ids


class Server:
    def __init__(self):
        self.edge2server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server2client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.edge2server_socket.bind(('127.0.0.1', 4002))
        self.edge2server_socket.listen(1)
        print("listening to edge2server")

        self.model = Target_model(args.target_model, args.temperature)
        self.tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    def recv_tensor(self, sock):
        length_bytes = sock.recv(4)
        tensor_length = struct.unpack('>I', length_bytes)[0]

        data = b''
        while len(data) < tensor_length:
            packet = sock.recv(min(tensor_length - len(data), 1024))
            if not packet:
                raise RuntimeError("socket connection broken")
            data += packet

        tensor = pickle.loads(data)
        return tensor

    def run_server(self):

        conn, addr = self.edge2server_socket.accept()
        print("edge2server accepted,tring to connect server2client")
        self.server2client_socket.connect(('127.0.0.1', 4003))
        print("server2client connected")

        with conn:
            while True:
                input_ids = self.recv_tensor(conn)
                draft_tokens = self.recv_tensor(conn)
                retrieve_indices = self.recv_tensor(conn)
                token_indices = self.recv_tensor(conn)
                tree_position_ids = self.recv_tensor(conn)
                output_ids = self.model.sampling(input_ids, draft_tokens, retrieve_indices, retrieve_indices,
                                                 tree_position_ids)
                new_ids = output_ids[len(input_ids):]
                response_tokens = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                self.server2client_socket.sendall(str(response_tokens).encode('utf-8'))


if __name__ == "__main__":
    server = Server()
    server.run_server()