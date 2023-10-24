from torch.utils.data import Dataset
import os 
from typing import Any
from PIL import Image
from utils.conversation import Conversation
import pickle
import json

def custom_collate(original_batch):
    '''
    Custom collate function for batch loading
    '''
    img_names = [item[0] for item in original_batch]
    imgs_pre = [item[1] for item in original_batch]
    prompt_pre = [item[2] for item in original_batch]
    chat_pre = {item[0]:item[3] for item in original_batch}
    imgs_post = [item[4] for item in original_batch]
    prompt_post = [item[5] for item in original_batch]
    chat_post = {item[0]:item[6] for item in original_batch}

    return img_names, imgs_pre, prompt_pre, chat_pre, imgs_post, prompt_post, chat_post

class ChatSet(Dataset):
    '''
    This dataset class serves to load the images and the respective chats in batches, so that it can be processed in batch and faster.
    '''
    def __init__(self, images_path, chats_path, context = 100, mode = "questioning"):
        super(ChatSet, self).__init__()
        # images_path : str -> path of the folder containing all the images
        # chats_path : str -> path of the folder containing all the chats
        self.images_path = images_path
        # Get the path of the images pre change
        self.im_pre_path = os.path.join(images_path, "im1")
        # Get the path of the images post change
        self.im_post_path = os.path.join(images_path, "im2")
        # Get the path of the masks
        # TODO
        # Get the path of the chats
        self.chats_path = chats_path
        self.images_names = os.listdir(self.im_pre_path)
        self.conversation = Conversation()
        self.mode = mode
        self.context = context
    
    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, index) -> Any:
        im_pre_path = os.path.join(self.im_pre_path, self.images_names[index])
        im_post_path = os.path.join(self.im_post_path, self.images_names[index])
        chat_pre_path = os.path.join(self.chats_path, self.images_names[index].split(".")[0]+"_pre.pkl")
        chat_post_path = os.path.join(self.chats_path, self.images_names[index].split(".")[0]+"_post.pkl")
        # Open the image 
        im_pre = Image.open(im_pre_path)
        im_post = Image.open(im_post_path)
        # Open the chat
        if(os.path.exists(chat_pre_path)):
            with open(chat_pre_path, "rb") as file:
                chat_pre = pickle.load(file)
            self.conversation.reset_messages()
            self.conversation.load_messages(chat_pre)
            if(self.mode=="answering"):
                # Get answer prompt 
                prompt_pre = self.conversation.get_answer_prompt(model="blip2", context=1)
            elif(self.mode=="questioning"):
                prompt_pre = self.conversation.get_question_prompt(model="vicuna", context=self.context)
            elif self.mode == "summarization":
                prompt_pre = self.conversation.get_summary_prompt(model="vicuna", context=self.context)
        else:
            assert self.mode=="answering"
            chat_pre = [["ASSISTANT", "give a detailed description of this satellite image."]]
            self.conversation.load_messages(chat_pre)
            prompt_pre = self.conversation.get_answer_prompt(model="blip2", context=self.context)
        
        if(os.path.exists(chat_post_path)):
            with open(chat_post_path, "rb") as file:
                chat_post = pickle.load(file)
            
            self.conversation.reset_messages()
            self.conversation.load_messages(chat_post)
            if(self.mode=="answering"):
                # Get answer prompt 
                prompt_post = self.conversation.get_answer_prompt(model="blip2", context=1)
            elif(self.mode=="questioning"):
                prompt_post = self.conversation.get_question_prompt(model="vicuna", context=self.context)
            elif self.mode == "summarization":
                prompt_post = self.conversation.get_summary_prompt(model="vicuna", context=self.context)
        else:
            assert self.mode=="answering"
            chat_post = [["ASSISTANT", "give a detailed description of this satellite image."]]
            self.conversation.load_messages(chat_post)
            prompt_post = self.conversation.get_answer_prompt(model="blip2", context=self.context)
            
        return self.images_names[index], im_pre, prompt_pre, chat_pre,  im_post, prompt_post, chat_post

class SummarySet(Dataset):
    def __init__(self, chats_path):
        # Open the dict of chats
        with open(chats_path, "rb") as file:
            self.chats = json.load(file)
            self.image_names = list(self.chats.keys())
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        return image_name, self.chats[image_name]
    