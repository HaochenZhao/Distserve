from openai import OpenAI
import time

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123", # doesn't matter
)
# do it 5 times and average the time
for i in range(5):
    start_time = time.time()
    completion = client.chat.completions.create(
      model="facebook/opt-6.7b",
      messages=[
        {"role": "user", "content": "please write a very long story about a robot"},
      ],
    )
    end_time = time.time()
    print("Time taken:", end_time - start_time)

    # print(completion.choices[0].message)
    

print(completion.choices[0].message)
# Prompt logprob is not supported by multi step workers. (e.g., speculative decode uses multi step workers).
# Results:
# CUDA_VISIBLE_DEVICES=2,3 vllm serve facebook/opt-6.7b --speculative-model facebook/opt-125m --num-speculative-tokens 5 --use-v2-block-manager
# Time taken: 0.8973073959350586
# Time taken: 1.131464958190918
# Time taken: 1.6189348697662354
# Time taken: 1.2965037822723389
# Time taken: 0.8904790878295898
# Time taken: 1.0218045711517334
# Time taken: 2.050689697265625
# Time taken: 1.8678123950958252
# Time taken: 2.5469071865081787
# Time taken: 0.7774200439453125
# CUDA_VISIBLE_DEVICES=2,3 vllm serve facebook/opt-6.7b --use-v2-block-manager
# Time taken: 0.7754802703857422
# Time taken: 0.7177777290344238
# Time taken: 0.5624654293060303
# Time taken: 2.6629276275634766
# Time taken: 6.27223801612854
# Time taken: 0.48522281646728516
# Time taken: 1.4568719863891602
# Time taken: 0.40433359146118164
# Time taken: 0.6843302249908447
# Time taken: 4.473360061645508


