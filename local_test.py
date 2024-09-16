import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'

import sys
import time
import random
import argparse
import torch
from edge_server import Draft_model
from server import Target_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

draft_model = Draft_model(model_name_or_path="../models/opt-125M", topk=2, depth=2, total_tokens=5)
target_model = Target_model(model_name_or_path="../models/opt-1.3B", temperature=0.5)
tokenizer = AutoTokenizer.from_pretrained("../models/opt-1.3B")

if __name__ == "__main__":

    # prompt = input("Enter your prompt: ")
    # 目前我认为问题出在采样的代码，因为理论上表现是有target model保底的，但是目前结果没有逻辑性
    prompt = "Emily found a mysterious letter on her doorstep one sunny morning."
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
    while (input_ids.shape[1] < 64):
        draft_tokens, token_indices, retrieve_indices, tree_position_ids = draft_model.generate_tree(input_ids)
        input_ids = target_model.sampling(input_ids, draft_tokens, retrieve_indices, token_indices, tree_position_ids)

    response_tokens = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(response_tokens)
