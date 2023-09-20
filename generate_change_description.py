from utils.chat import *
import json
from tqdm import tqdm


def generate_change_description_dialogues(dialogue_path):
    with open(dialogue_path, "r") as file:
        dialogues = json.load(file)

    chat = Chatter()
    change_descriptions = dict()

    for key, value in tqdm(dialogues.items()):
        caption1 = value[0]["answers"][0]
        caption2 = value[1]["answers"][0]

        change_desc = chat.change_description(caption1, caption2)
        change_descriptions[key] = change_desc

    return change_descriptions


def generate_change_description_summaries(summaries_path):
    with open(summaries_path, "r") as file:
        dialogues = json.load(file)

    chat = Chatter()
    change_descriptions = dict()

    for key, value in tqdm(dialogues.items()):
        summary1 = value[0]["summary1"]
        summary2 = value[1]["summary2"]

        change_desc = chat.change_description(summary1, summary2)
        change_descriptions[key] = change_desc

    return change_descriptions


if __name__ == "__main__":
    change_desc = generate_change_description_summaries(
        "results/summaries_crop.json"
    )

    with open("results/change_descriptions_crop.json", "w") as file:
        json.dump(change_desc, file, indent=4)
