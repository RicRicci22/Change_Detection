from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hug_model(model, device):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, padding_side="left")
    #tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model,
                                                device_map=device,
                                                trust_remote_code=False,
                                                revision="main")

    if hasattr(model.config, "max_length"):
        context_len = model.config.max_length
    else:
        context_len = "Unknown"
    
    print("Model context length: {}".format(context_len))

    return tokenizer, model

