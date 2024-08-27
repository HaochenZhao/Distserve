from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123", # doesn't matter
)

completion = client.chat.completions.create(
  model="meta-llama/Meta-Llama-3-8B-Instruct",
  messages=[
    {"role": "user", "content": "please write a very long story about a robot"},
  ],
  # extra args
#   extra_body={
#     "stop_token_ids": [151329, 151336, 151338]
#   }
)

print(completion.choices[0].message)