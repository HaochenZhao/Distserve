# Project distserve
This project aims to optimize performance of LLM inference by utilizing computational source of edge cloud and speculative decoding.


## Normal vllm usage
1. Setup vllm server in openai-style: (change necessary arguments in practice)  
```bash
conda create -n normal python=3.10
conda activate normal
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server --model  meta-llama/Meta-Llama-3-8B-Instruct --max-model-len=2048
```
2. Then start the client:  
```bash
python vllm_client.py
```
3. Benchmark offline inference throughput (vllm or hf), not done yet (may inspect directly from server log actually):
```bash
python benchmark_throughput.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --backend vllm \
        --input-len 64 \
        --output-len 128 \
        --num-prompts 25 \
        --seed 2024 \
    --dtype float16 \
    --max-model-len 512
```


## Speculative decoding in vllm
1. Setup vllm server  in openai-style: (change necessary arguments in practice)  
```bash
conda create -n speculative python=3.10
conda activate speculative
pip install -U vllm
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=2,3 vllm serve facebook/opt-6.7b --speculative-model facebook/opt-125m --num-speculative-tokens 5 --use-v2-block-manager
```
2. Then start the client and timing:  
```bash
python vllm_client_spec.py
```