from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
)
import torch

from transformers.generation.utils import GenerationMixin
from utils_center import MyGenerationMixin

GenerationMixin.assisted_decoding = MyGenerationMixin.assisted_decoding
GenerationMixin._assisted_decoding = MyGenerationMixin._assisted_decoding
GenerationMixin.generate = MyGenerationMixin.generate

model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16).cuda()
assistant_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16).cuda()

# the fast tokenizer currently does not work correctly
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)

prompt = "Who are you?"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

logits_processor = LogitsProcessorList(
    [
        MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
    ]
)
stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

generated_ids = model.generate(
    input_ids,
    assistant_model=assistant_model,
    use_remote=True
)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
