from pathlib import Path
import sys
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import transformers
from typing import Iterable

sys.path.insert(
    0, str(Path("repositories/GPTQ-for-LLaMa"))
)  # This line is needed to import the following modules
from modelutils import find_layers
from quant import make_quant


def load_quant(
    model,
    checkpoint,
    wbits,
    groupsize=-1,
    faster_kernel=False,
    exclude_layers=["lm_head"],
    kernel_switch_threshold=128,
):
    config = AutoConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]
    make_quant(
        model,
        layers,
        wbits,
        groupsize,
        faster=faster_kernel,
        kernel_switch_threshold=kernel_switch_threshold,
    )

    del layers

    print("Loading model ...")
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load

        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)
    model.seqlen = 2048
    print("Done.")

    return model


def load_quantized(model_path, wbits=4, groupsize=128, threshold=128):
    found_pts = list(Path(model_path).glob("*.pt"))
    found_safetensors = list(Path(model_path).glob("*.safetensors"))
    pt_path = None

    if len(found_pts) == 1:
        pt_path = found_pts[0]
    # elif len(found_safetensors) == 1:
    #     pt_path = found_safetensors[0]

    if not pt_path:
        print(
            "Could not find the quantized model in .pt or .safetensors format, exiting..."
        )
        exit()

    model = load_quant(
        str(model_path),
        str(pt_path),
        wbits,
        groupsize,
        kernel_switch_threshold=threshold,
    )

    return model


def load_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading GPTQ quantized model...")
    model = load_quantized(model_path)

    model.to(device)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len


# GENERATE THE STREAM
@torch.inference_mode()
def generate_stream(params, model, tokenizer, context_len, device):
    eos_token_id = tokenizer.encode("</s>", add_special_tokens=False)

    if len(eos_token_id) != 1:
        raise ValueError("The stop token must be a single token.")

    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = min(
        int(params.get("max_new_tokens", 256)), 1024
    )  # between 1 and 1024
    stop_str = params.get("stop", None)  # stop symbol --> ###

    input_ids = tokenizer(prompt).input_ids
    output_ids = []

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(torch.as_tensor([input_ids]).to(device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device
            )
            out = model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]
        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in eos_token_id:
            stopped = True
        else:
            stopped = False

        if i == max_new_tokens - 1 or stopped:
            rfind_start = 0

            output = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
            return output
