from typing import Any, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def quick_tokenization_sanity_check(tokenizer: AutoTokenizer, config: Dict[str, Any] = {}, test_text: str="This is a test input for the tokenizer."):
    """Run a quick sanity check on the tokenizer to verify it's working correctly."""
    print(f'{"--"*15}\nStart Quick Tokenization Sanity Check')
    # Tokenize the input text
    encoded = tokenizer(
        test_text, 
        padding="max_length", 
        truncation=True, 
        max_length=13, 
        return_tensors='pt'
    )

    # Display tokenizer settings
    print(f'Tokenizer Settings:')
    print(f'Type: {type(tokenizer)}')
    if tokenizer.pad_token_id is not None:
        print(f"Pad Token ID: {tokenizer.pad_token_id}")
    else:
        print("Pad Token ID is None")
    print(f"EOS Token ID: {tokenizer.eos_token_id}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print(f"Config max sequence length: {config.get('max_seq_length', 'Not Set')}")
    
    # Display tokenizer outputs
    print(f'\nTokenizer Outputs:')
    print(f"Input Text: {test_text}")
    print(f"Tokenized IDs: {encoded['input_ids']}")
    print(f"Attention Mask: {encoded['attention_mask']}")
    print(f"Decoded Tokens: {tokenizer.decode(encoded['input_ids'][0])}")
    print(f'{"--"*15}\nEnd Tokenization Sanity Check\n')

def load_model_and_tok(config: Dict[str, Any] = {}) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer based on configuration.

    Warning: if you aren't using filling all the way to the end, you need to make 
    sure your data loaders are aware of the models and tokenizers you are using.
    """
    # Get model name from config or use default
    model_name = config.get('model_name', 'meta-llama/Meta-Llama-3-8B-Instruct')
    print(f'Loading model: {model_name}')
    
    # Determine torch data type for model loading
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    print(f'Using torch dtype: {torch_dtype}')
    
    # Create tokenizer with common parameters
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        padding_side="right", 
        trust_remote_code=True, 
        add_eos_token=True
    )
    
    # --- Model-specific loading logic ---
    # Gemma-2 models (special attention implementation)
    if 'gemma-2' in str(model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype,
            attn_implementation='eager',  # Crucial for Gemma-2
            trust_remote_code=True
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    # Other Gemma models
    elif 'gemma-' in str(model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype,
            trust_remote_code=True, 
            device_map=config.get('device_map', 'auto')
        )
    # Meta-Llama-3 models (check if tokenizer's pad token is None)
    elif 'meta-llama' in str(model_name).lower() or tokenizer.pad_token is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Handle missing pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Warning: EOS and PAD tokens are the same: {tokenizer.eos_token}")
    # All Qwen models
    elif 'Qwen' in str(model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype, 
            trust_remote_code=True, 
            device_map=config.get('device_map', 'auto')
        )
        
    # Unsupported models
    else:
        raise ValueError(f'Error: Model not supported: {model_name}')
    
    # Print model information
    print(f'Tokenizer max length: {tokenizer.model_max_length} (model\'s true max length)')
    print(f'Model config max position embeddings: {model.config.max_position_embeddings}')
    
    # Run sanity check
    quick_tokenization_sanity_check(tokenizer, config)
    print(f'{device=} {torch_dtype=}')
    print(f'{next(model.parameters()).dtype=}')
    return model, tokenizer

def _test():
    """Test function to verify model and tokenizer loading."""
    model, tokenizer = load_model_and_tok({'model_name': 'meta-llama/Meta-Llama-3-8B-Instruct'})
    
if __name__ == '__main__':
    _test()