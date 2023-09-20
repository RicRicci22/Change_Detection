# -------------------------------------------------------------------------
# Author:   Alberto Frizzera
# Date:     22/04/2023
# Version:  1
# -------------------------------------------------------------------------

import requests
import json

# For local streaming, the websockets are hosted without ssl - http://
HOST = "localhost:5000"
URI_STREAM = f"http://{HOST}/api/v1/generate"
URI_CHAT = f"http://{HOST}/api/v1/chat"

# For reverse-proxied streaming, the remote will likely host with ssl - https://
# URI = 'https://your-uri-here.trycloudflare.com/api/v1/generate'


def prompt_response_stream(prompt):
    request = {
        "prompt": prompt,
        "max_new_tokens": 500,
        "do_sample": True,
        "temperature": 1.3,
        "top_p": 0.1,
        "typical_p": 1,
        "epsilon_cutoff": 0,  # In units of 1e-4
        "eta_cutoff": 0,  # In units of 1e-4
        "repetition_penalty": 1.18,
        "top_k": 40,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "penalty_alpha": 0,
        "length_penalty": 1,
        "early_stopping": False,
        "mirostat_mode": 0,
        "mirostat_tau": 5,
        "mirostat_eta": 0.1,
        "seed": -1,
        "add_bos_token": True,
        "truncation_length": 2048,
        "ban_eos_token": False,
        "skip_special_tokens": True,
        "stopping_strings": [],
    }

    response = requests.post(URI_STREAM, json=request)

    if response.status_code == 200:
        result = response.json()["results"][0]["text"]
    return result


def prompt_response_chat(user_input, history, _continue=False):
    request = {
        "user_input": user_input,
        "max_new_tokens": 500,
        "history": history,
        "mode": "instruct",  # Valid options: 'chat', 'chat-instruct', 'instruct'
        "character": "Example",
        "instruction_template": "Vicuna-v1.1",
        "your_name": "Human",
        "regenerate": False,
        "_continue": _continue,
        "stop_at_newline": False,
        "chat_generation_attempts": 1,
        "chat-instruct_command": 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',
        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        "preset": "None",
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.1,
        "typical_p": 1,
        "epsilon_cutoff": 0,  # In units of 1e-4
        "eta_cutoff": 0,  # In units of 1e-4
        "tfs": 1,
        "top_a": 0,
        "repetition_penalty": 1.18,
        "top_k": 40,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "penalty_alpha": 0,
        "length_penalty": 1,
        "early_stopping": False,
        "mirostat_mode": 0,
        "mirostat_tau": 5,
        "mirostat_eta": 0.1,
        "seed": -1,
        "add_bos_token": True,
        "truncation_length": 2048,
        "ban_eos_token": False,
        "skip_special_tokens": True,
        "stopping_strings": [],
    }

    response = requests.post(URI_CHAT, json=request)

    if response.status_code == 200:
        result = response.json()["results"][0]["history"]

    return result


if __name__ == "__main__":
    userinput = "Hi, my name is Riccardo"
    history = {"internal": [], "visible": []}

    request = {
        "user_input": userinput,
        "max_new_tokens": 100,
        "history": history,
        "mode": "instruct",  # Valid options: 'chat', 'chat-instruct', 'instruct'
        "character": "Example",
        "instruction_template": "Vicuna-v1.1",
        "your_name": "Human",
        "regenerate": False,
        "_continue": False,
        "stop_at_newline": False,
        "chat_generation_attempts": 1,
        "chat-instruct_command": 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',
        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        "preset": "None",
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.1,
        "typical_p": 1,
        "epsilon_cutoff": 0,  # In units of 1e-4
        "eta_cutoff": 0,  # In units of 1e-4
        "tfs": 1,
        "top_a": 0,
        "repetition_penalty": 1.18,
        "top_k": 40,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "penalty_alpha": 0,
        "length_penalty": 1,
        "early_stopping": False,
        "mirostat_mode": 0,
        "mirostat_tau": 5,
        "mirostat_eta": 0.1,
        "seed": -1,
        "add_bos_token": True,
        "truncation_length": 2048,
        "ban_eos_token": False,
        "skip_special_tokens": True,
        "stopping_strings": [],
    }

    response = requests.post(URI_CHAT, json=request)

    if response.status_code == 200:
        history = response.json()["results"][0]["history"]

    print(history)

    userinput = "What is my name?"

    request = {
        "user_input": userinput,
        "history": history,
        "max_new_tokens": 100,
        "mode": "instruct",  # Valid options: 'chat', 'chat-instruct', 'instruct'
        "character": "Example",
        "instruction_template": "Vicuna-v1.1",
        "your_name": "Human",
        "regenerate": False,
        "_continue": False,
        "stop_at_newline": False,
        "chat_generation_attempts": 1,
        "chat-instruct_command": 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',
        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        "preset": "None",
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.1,
        "typical_p": 1,
        "epsilon_cutoff": 0,  # In units of 1e-4
        "eta_cutoff": 0,  # In units of 1e-4
        "tfs": 1,
        "top_a": 0,
        "repetition_penalty": 1.18,
        "top_k": 40,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "penalty_alpha": 0,
        "length_penalty": 1,
        "early_stopping": False,
        "mirostat_mode": 0,
        "mirostat_tau": 5,
        "mirostat_eta": 0.1,
        "seed": -1,
        "add_bos_token": True,
        "truncation_length": 2048,
        "ban_eos_token": False,
        "skip_special_tokens": True,
        "stopping_strings": [],
    }

    response = requests.post(URI_CHAT, json=request)

    if response.status_code == 200:
        history = response.json()["results"][0]["history"]

    print(history)
