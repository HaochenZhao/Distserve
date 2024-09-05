import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'

import sys
import time
import random
import argparse
import torch
import torch.nn as nn
import socket
import pickle
from tqdm import tqdm
import pickle
import struct
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from util import *

model_name_or_path = "../models/opt-125M"

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--draft_model', default=model_name_or_path, help='Draft model')
parser.add_argument('--depth', default=5, type=int, help='Token tree depth')
parser.add_argument('--topk', default=8, type=int, help='TopK')
parser.add_argument('--total_tokens', default=32, type=int, help='TopK')
args = parser.parse_args()

class Draft_model:

    def __init__(self, model_name_or_path, topk, depth, total_tokens):
        pretrained_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
        pretrained_model.eval()
        self.topk = topk
        self.depth = depth
        self.total_tokens = total_tokens
        self.decoder = pretrained_model.model.decoder
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.stable_kv = None
        self.lm_head = pretrained_model.lm_head

    def generate_tree(self, input_ids):

        input_ids = input_ids.clone()
        input_ids = input_ids.to(self.device)

        sample_token = input_ids[:, -1]

        scores_list = []
        parents_list = []
        ss_token = []

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(self.decoder.device)
        seq_len = input_ids.shape[1]
        self.tree_mask = None
        if self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            
            output = self.decoder(
                input_ids, past_key_values=self.stable_kv, use_cache=True
            )
        else:
            output = self.decoder(
                input_ids, use_cache=True
            )

        last_p = self.logsoftmax(self.lm_head(output.last_hidden_state[:, -1]))
        top = torch.topk(last_p, self.topk, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])

        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        # input_ids = topk_index
        # Batch copy input_ids
        input_ids = input_ids.repeat(self.topk, 1)
        self.stable_kv = []
        for layer_kv in output.past_key_values:
            tmp_tup = []
            for korv in layer_kv:
                tmp_tup.append(korv.repeat(self.topk, 1, 1, 1))
            self.stable_kv.append(tuple(tmp_tup))
        self.stable_kv = tuple(self.stable_kv)
        input_ids = torch.cat((input_ids, topk_index[0][:, None]), dim=1)

        topk_cs_index = torch.arange(self.topk, device=self.device)

        for i in range(self.depth):
            output = self.decoder(
                input_ids=input_ids,
                past_key_values=self.stable_kv,
                use_cache=True
            )
            seq_len += 1

            bias1 = self.topk if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + self.topk ** 2 * bias2 + bias1

            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_p = self.logsoftmax(self.lm_head(output.last_hidden_state[:,-1]))
            top = torch.topk(last_p, self.topk, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]
            topk_cs = torch.topk(cu_scores.view(-1), self.topk, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p
            out_ids = topk_cs_index // self.topk
            input_ids = input_ids[out_ids, :]

            self.stable_kv = []
            for layer_kv in output.past_key_values:
                tmp_tup = []
                for korv in layer_kv:
                    out_ids = out_ids.to(korv.device)
                    tmp_tup.append(korv[out_ids, :, :, :])
                self.stable_kv.append(tuple(tmp_tup))
            self.stable_kv = tuple(self.stable_kv)

            input_ids = torch.cat((input_ids, topk_index.view(-1)[topk_cs_index].unsqueeze(1)), dim=1)
            ss_token.append(topk_index)
            scores_list.append(cu_scores)

        # scores_list: shape(51,10)->(510) The scores of each father's children
        # ss_token_list: shape(51,10)->(510) The id of each node
        # draft_parents: shape(total_tokens) The father of each node
        # mask_index_list: list[total_tokens] The mapping of each node's father in top
        # tree_mask The i-th row is the mask of all parents of the i-th path

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)

        top_scores = torch.topk(scores_list, self.total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = draft_tokens.to(sample_token.device)
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)
        draft_parents = parents_list[0].to(self.device)
        for i in range(1,len(parents_list)):
            parents_list[i] = parents_list[i].to(self.device)
            draft_parents = torch.cat((draft_parents, parents_list[i]),dim=0)
        top_scores_index = top_scores_index.to(draft_parents.device)
        draft_parents = draft_parents[top_scores_index // self.topk].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        tree_mask = torch.eye(self.total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(self.total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = self.total_tokens - noleaf_num

        # The node storing the jth depth of the i-th path is retrieve_indices[i][j]
        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        # Match candidate sequence matrix
        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(self.total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        maxitem = self.total_tokens + 5
        def custom_sort(lst):
            sort_keys = []
            for i in range(len(lst)):
                sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
            return sort_keys
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(self.device)

        #draft_tokens.shape = (1,topk)
        #retrieve_indices.shape = e.g. (15,7) 候选序列数*最长长度
        #tree_mask.shape = (1,1,topk,topk)
        #tree_position_ids.shape = (topk)

        self.stable_kv = None
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids
    
class Edge_server:
    def __init__(self):
        
        self.edge2server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client2edge_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client2edge_socket.bind(('127.0.0.1', 4009))
        self.client2edge_socket.listen(1)
        print("listening to client2edge")
        
        self.model = Draft_model(args.draft_model, args.topk, args.depth, args.total_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
        
    def send_tensor_over_socket(self, tensor, sock):
        tensor_bytes = pickle.dumps(tensor)
        tensor_length = len(tensor_bytes)
        length_bytes = struct.pack('>I', tensor_length)
        sock.sendall(length_bytes + tensor_bytes)

    def run_edge(self):
    
        conn, addr = self.client2edge_socket.accept()
        print("client2edge accepted, tring to connect edge2server")
        self.edge2server_socket.connect(('127.0.0.1', 4002))
        print("edge2server connected")

        with conn:
            while True:
                data = conn.recv(102400)
                if not data: break
                prompt = data.decode('utf-8')
                input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids']
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.model.generate_tree(input_ids)
                
                with self.edge2server_socket as server_conn:
                    self.send_tensor_over_socket(input_ids, server_conn)
                    self.send_tensor_over_socket(draft_tokens, server_conn)
                    self.send_tensor_over_socket(retrieve_indices, server_conn)
                    self.send_tensor_over_socket(tree_mask, server_conn)
                    self.send_tensor_over_socket(tree_position_ids, server_conn)

if __name__ == "__main__":
    edge = Edge_server()
    edge.run_edge()


    


    # def run_edge(self):
    #     prompt = "Emily found a mysterious letter on her doorstep one morning."
    #     input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids']
    #     draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.model.generate_tree(input_ids)
    #     s = Server()
    #     input_ids = s.sampling(input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids)
    #     print(self.tokenizer.decode(input_ids[0]))
