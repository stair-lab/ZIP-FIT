import os
from vllm import LLM

# Log In
from huggingface_hub import create_repo, upload_file, login, whoami
key_file_path = "~/keys/master_hf_token.txt"
key_file_path = os.path.abspath(os.path.expanduser(key_file_path))
with open(key_file_path, "r", encoding="utf-8") as f:
    token = f.read().strip()
login(token=token)
# os.environ['HUGGINGFACE_TOKEN'] = token
# os.environ["HUGGING_FACE_HUB_TOKEN"] = token

user_info = whoami()
print(f"Currently logged in as: {user_info['name']}\n")

# llm = LLM(model="UDACA/gemma-2-2b", trust_remote_code=True)
llm = LLM(model="google/gemma-2-2b", trust_remote_code=True)
output = llm.generate("Hello, my name is")
print("Model output:", output)
