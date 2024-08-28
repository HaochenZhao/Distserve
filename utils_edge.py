import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
from flask import Flask, request, jsonify
import json
import pickle
import base64

import torch

from transformers import AutoModelForCausalLM

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.configuration_utils import GenerationConfig
    from transformers.generation.logits_process import LogitsProcessorList
    
from transformers.generation.candidate_generator import AssistedCandidateGenerator

if __name__ == "__main__":
    app = Flask(__name__)
    global candidate_generator
    global assistant_model
    
    assistant_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16).cuda()


    @app.route('/init', methods=['POST'])
    def process_init():
        data = pickle.loads(request.data)
        global candidate_generator
        
        try:
            candidate_generator = AssistedCandidateGenerator(
                input_ids=data['input_ids'],
                assistant_model=assistant_model,
                generation_config=data['generation_config'],
                logits_processor=data['logits_processor'],
                model_kwargs=data['model_kwargs'],
                inputs_tensor=data['inputs_tensor'],
            )
            response = {
                "header": "Ok",
                "status": 200
            }
        except Exception as e:
            response = {
                "header": "Error",
                "status": 500,
                "error": str(e)
            }
        
        return jsonify(response)
    
    @app.route('/generate', methods=['POST'])
    def generate_text():
        data = pickle.loads(request.data)
        candidate_input_ids, candidate_logits = candidate_generator.get_candidates(data)
        output = {
            "candidate_input_ids": candidate_input_ids,
            "candidate_logits": candidate_logits
        }
        response = {
            "output": base64.b64encode(pickle.dumps(output)).decode('utf-8'),
        }
        return jsonify(response)
        

    if __name__ == '__main__':
        app.run(port=4444)