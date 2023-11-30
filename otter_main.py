'''
This file contains all the functions and code to run otter based approaches for change captioning.

Approach 1 
Otter is used to directly extract the change paragraph from the two images.
Approach 2
Otter is used to extract single descriptions of the images, and then vicuna is used to analyze the descriptions and extract the changes.
Approach 3
Otter is used to answer questions about the two images in a dialogue setting. Vicuna is used to generate questions. After the dialogue, vicuna summarizes the changes in a descriptive paragraph.
'''

from utils.chat import Chatter
from utils.dataset import MultitemporalImageSet
import os
from tqdm import tqdm
import torch
from utils.dataset import custom_collate
import pickle
from transformers import GenerationConfig
import json
from utils.post_process_chats import chats_postprocessing

def create_description(images_path,approach):
    if approach==1:
        # Steps 
        # 1. Load the dataset of images -> the dataset should return batches of im1,im2.
        # 2. Load the model -> otter 
        # 3. Generate the descriptions 
        
        # 1. Load the dataset and create the dataloader
        assert os.path.isdir(images_path), "Double check the directory path"
        dataset = MultitemporalImageSet(images_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
        
        # 2. Load the model (otter)
        
        
        
        pass
    elif approach==2:
        pass
    elif approach==3:
        pass
    else:
        raise Exception("Find a suitable approach to test!")

if __name__ == "__main__":
    approach = 1
    image_path = "/media/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test"

    create_description(images_path=image_path,approach=approach)