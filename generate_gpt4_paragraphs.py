'''
This code uses GPT-4 vision API to generate change descriptions for the images in the dataset.
'''
import json
import os
import base64
import requests
from tqdm import tqdm 
import time


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

with open("GPT-4_data.json", "r") as f:
    data = json.load(f)
        
path_images = "/media/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test"

images = os.listdir(path_images+"/im1")

images = [image.split(".")[0] for image in images]
already_done = [name.split(".")[0] for name in list(data.keys())]

todo = list(set(images) - set(already_done))

# OpenAI API Key
api_key = "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"

data_new = {}

for image in tqdm(todo):
    im1_path = os.path.join(path_images, "im1", image+".png")
    im2_path = os.path.join(path_images, "im2", image+".png")

    # Getting the base64 string
    im1_base64 = encode_image(im1_path)
    im2_base64 = encode_image(im2_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {"role":"system", "content": "I'm a user interested in textual descriptions of the changes occurred between two satellite images and you are an assistant that can help me in this task. I will upload both images together in the prompt. You must directly describe the changes between them without relying on additional tools. I'm interested in structural changes (for example the construction of a new building, changes in land use, and others that are pertinent to remote sensing). You must avoid to describe changes that are related to the image acquisition or seasonality."},
        {"role": "system", "content":'''You must respond with an accurate, clear and easy to follow description of the changes. The description must be split in two parts. 
                                        The first part is a bullet list of the changes, every item contain the description of one change. The template is as follows. 
                                        List of changes:
                                        - change 1
                                        - change 2 
                                        and so on, listing all the changes between the two images. 

                                        Summary:
                                        Summary of the bullet list, in the form of a descriptive paragraph of the changes that have been observed. Describe the changes like you are in the present, looking at image 2 and you describe what in image 2 is different from image 1.'''},
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Find the differences between the two images. If there are no noticeable changes, say so. Explain as the area morphed from the first to the second image."
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{im1_base64}",
            }
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{im2_base64}",
            }
            }
        ]
        }
    ],
    "max_tokens": 1000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    data_new[image+".txt"] = response.json()["choices"][0]["message"]["content"]
    
    # Save it in the json file
    with open("GPT-4_data_new.json", "w") as f:
        json.dump(data_new, f, indent=4)
    
    time.sleep(1)