from utils.chat import *
import json
from tqdm import tqdm
from text_generator_api import prompt_response_chat

if __name__ == "__main__":
    with open("results/img_dialogues_crop.json", "r") as file:
        conversations = json.load(file)

    summaries = dict()
    device_summarizer = "cuda:1"

    # chat = Chatter(summarizer="vicuna", s_device=device_summarizer, a_device="cuda:1")
    chat = Chatter()

    for key, value in tqdm(conversations.items()):
        questions1 = value[0]["questions"]
        answers1 = value[0]["answers"]
        for i in range(len(questions1)):
            chat.conversation.append_question(questions1[i])
            chat.conversation.append_answer(answers1[i])
        prompt = chat.summarize()
        history = {"internal": [], "visible": []}
        summary1 = prompt_response_chat(prompt, history)["visible"][0][1]

        chat.reset_history()

        questions2 = value[1]["questions"]
        answers2 = value[1]["answers"]
        for i in range(len(questions2)):
            chat.conversation.append_question(questions2[i])
            chat.conversation.append_answer(answers2[i])
        prompt = chat.summarize()
        history = {"internal": [], "visible": []}
        summary2 = prompt_response_chat(prompt, history)["visible"][0][1]

        summaries[key] = [{"summary1": summary1}, {"summary2": summary2}]

    # Save the summaries
    with open("summaries_crop.json", "w") as file:
        json.dump(summaries, file, indent=4)
