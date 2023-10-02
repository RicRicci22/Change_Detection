import requests

HOST = "localhost:5000"
URI = f"http://{HOST}/api/v1/chat"

# Important thing for the AIs way to process and/or remember the conversation!
# The Language Model will always work on the second string, in the last arrayobject of the history.internal or *.visible object.
# (I am still not sure whats the difference between "internal" and "visible" here)
# Therefore, when a new "human" prompt is entered, before making the API request, you need to append the user_input
# and an empty string (or the string containing the start of the AI's reply) which the AI will then uses for the generation of its prompt.
# You could also do all that in an external ".json" file, but don't forget to implement a function to clear the memory at a certain point.

# This example does not include a way to check the tokensize for the historyprompt.

# EXAMPLE HISTORY WITH A NEW USER INPUT BEFORE SENDING WITH API

"""
history = {
    'internal': [
        [
            'Hi there',  <=== (first conversation with AI response)
            'Hello, I am an AI. How can I help you?'
        ],
        [
            'Please explain how language models work',  <=== (new user_input)
            ''  <=== (String for the AI response)
        ]
    ],
    'visible': [
        [
            'Hi there',  <=== (first conversation with AI response)
            'Hello, I am an AI. How can I help you?'
        ],
        [
            'Please explain how language models work',  <=== (new user_input)
            ''  <=== (String for the AI response)
        ]
    ]
}
"""

# Optionally, you can use 'Language models are' here to provide an empty string ('') for the AI to work on.
# The history array, has to be outside of the actual running proccess like a FOR or WHILE loop, to keep its memory :)

history = {"internal": [], "visible": []}


# API Request function
def run(user_input, history):
    request = {
        "user_input": user_input,
        "history": history,
        "mode": "chat",  # Options: 'chat', 'chat-instruct', 'instruct'
        "character": "Example",
        "instruction_template": "Vicuna-v1.1",
        "your_name": "USER",
        "regenerate": False,
        "_continue": False,  # Option for sending History
        "stop_at_newline": False,
        "chat_prompt_size": 2048,
        "chat_generation_attempts": 1,
        "chat_instruct_command": 'Continue the chat dialogue below. Write a single reply for the character "Assistant"\n\n',
        "max_new_tokens": 250,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "typical_p": 1,
        "epsilon_cutoff": 0,
        "eta_cutoff": 0,
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
        "stopping_strings": [
            "\n### Assistant:",
            "\n### Human:"
        ],  # I added these because it seems like all AI love Assistant and Human
    }

    try:
        # print(">> SENDING API REQUEST")
        response = requests.post(URI, json=request)

        if response.status_code == 200:
            # Uncomment below to show the full API response
            # print("RESPONSE:", response.json())

            # The "empty" string that now contains the AI's response
            result = response.json()["results"][-1]["history"]
            # print("Received AI answer:", result["visible"][-1][1])

            # Set the empty string, which we first pushed for the AI to use for generating its prompt,
            # to the now received answer, in ["internal"] and ["visible"]
            # history["internal"][-1][1] = result["visible"][-1][1]
            # history["visible"][-1][1] = result["visible"][-1][1]
            # Now the array containing the user_input, now has the AI's response saved.

            # Uncomment below to show the full history
            # print("> NEW HISTORY:\n", history)

            return result

        else:
            print("ERROR IN RESPONSE:", response.text)

    except requests.exceptions.RequestException as error:
        print("ERROR WHEN SENDING API REQUEST:", error)


# Loop 10 times for a small conversation with the AI
for i in range(10):
    # Read user input
    input_text = input(f"[{i}] Human: ")

    # Append a new conversation to the history, but with an empty string (or containing the start of AI reply)
    # for the AI to generate on.

    # Send API request with history
    history = run(input_text, history)
    print(history)

    print(history["visible"][-1][1])
