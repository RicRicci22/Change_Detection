from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from .blip2 import Blip2
import torch
from Otter.src.otter_ai.models.otter.modeling_otter import OtterForConditionalGeneration


def load_hug_model(model, device):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, padding_side="left")
    gptq_config = GPTQConfig(bits=4, use_exllama=True, exllama_config={"version":2})
    model = AutoModelForCausalLM.from_pretrained(model,
                                                device_map=device,
                                                trust_remote_code=False,
                                                quantization_config=gptq_config,
                                                revision="main")
    
    

    if hasattr(model.config, "max_length"):
        context_len = model.config.max_length
    else:
        context_len = "Unknown"
    
    print("Model context length: {}".format(context_len))
    
    model.eval()

    return tokenizer, model

def load_mistral(model_name, device: str="cpu"):
    '''
    Load mistral by calling the load_hug_model function, which is a general function to load models from huggingface
    '''
    tokenizer, model = load_hug_model(model_name, device)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_vicuna(model_name, device: str="cpu"):
    '''
    Load vicuna calling the load_hug_model function, which is a general function to load models from huggingface
    '''
    tokenizer, model = load_hug_model(model_name, device)

    return model, tokenizer


def load_blip2(model_name, device: str="cpu"):
    '''
    Load blip by custom Blip 2 class, TODO transform using the general function load_hug_model
    '''
    model = Blip2(model_name, device=device, bit4=True)

    return model

def load_otter(model_name, device: str="cpu"):
    '''
    This function loads otter model 
    '''
    precision = {"torch_dtype": torch.float16}
    model = OtterForConditionalGeneration.from_pretrained(model_name, device_map=device, **precision)  # OTTER-Image-MPT7B
    model.eval()

    return model

def load_openassistant(model_name, device: str="cpu"):
    '''
    Load openassistant calling the load_hug_model function, which is a general function to load models from huggingface
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Put half precision on weights and load on GPU
    model = model.half().to(device)
    

    return model, tokenizer