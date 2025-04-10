# train/models.py 

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def quick_tokenization_sanity_check(tok, config = {}, test_text="This is a test input for the tokenizer."):
    print(f'{"--"*15}\nStart Quick Tokenization Sanity Check Hardcoded Example')
    # Print tokenizer name (via type)

    # Tokenize the input text
    encoded = tok(test_text, padding="max_length", truncation=True, max_length=13, return_tensors='pt')

    # Nice Tokenizer Settings to see
    print(f'Nice Tokenizer Settings to see: ')
    print(f'Tokenizer name: {type(tok)=}')
    print(f"Pad Token ID: {tok.pad_token_id}") if tok.pad_token_id is not None else print("Pad Token ID is None")
    print(f"EOS Token ID: {tok.eos_token_id}")
    print(f"Max Length in Tokenizer: {tok.model_max_length=}. Max Seq Length in Config: {config.get('max_seq_length', 'Max Seq Length Not Set in Config.')}")
    print()
    # Display the tokenizer outputs
    print(f'Display the tokenizer outputs: ')
    print(f"Input Text: {test_text}")
    print(f"Tokenized IDs: {encoded['input_ids']}")
    print(f"Attention Mask: {encoded['attention_mask']}")
    # print(f"Labels: {encoded['labels']}")
    print(f"Decoded Tokens: {tok.decode(encoded['input_ids'][0])}")
    print(f'{"--"*15}\nEnd Quick Tokenization Sanity Check Hardcoded Example\n')

def load_model_and_tok(pretrained_model_name_or_path, config: dict = {}) -> tuple:
    print(f'----> {pretrained_model_name_or_path=}')
    # I think we always need to check special tokens are added correctly during training anyway. 
    tok = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="right", trust_remote_code=True)
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32 
    print(f'{torch_dtype=}')
    if 'gemma-' in str(pretrained_model_name_or_path): 
        # bos 1, eos 2, pad 2
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True, device_map=config.get('device_map', 'auto'))
    elif 'internlm2' in pretrained_model_name_or_path:
        # bos 1, eos 2, pad 2
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True, device_map=config.get('device_map', 'auto'))
    elif tok.pad_token is None:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True)
        tok.pad_token = tok.eos_token
        print(f"-----> Warning eos and pad are the same: {tok.eos_token=} == {tok.pad_token}.")
    else:
        raise ValueError(f'Error: model not supported {pretrained_model_name_or_path=}.')
    print(f'tok max length: {tok.model_max_length=} (dont overwrite it, its the true max length of the model, most likely)')
    print(f'model config max length: {model.config.max_position_embeddings=} (dont overwrite it, its the true max length of the model, most likely)')
    quick_tokenization_sanity_check(tok, config)
    return model, tok

def _test():
    model, tok = load_model_and_tok('google/gemma-2-2b-it')
    quick_tokenization_sanity_check(tok)

if __name__ == '__main__':
    _test()