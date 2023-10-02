from torch.utils.data import Dataset
import os 
from typing import Any
from PIL import Image

class ChatSet(Dataset):
    '''
    This dataset class serves to load the images and the respective chats in batches, so that it can be processed in batch and faster.
    '''
    def __init__(self, images_path, chats_path):
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
        self.images_names = os.listdir(images_path)
    
    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, index) -> Any:
        im_pre_path = os.path.join(self.im_pre_path, self.images_names[index])
        im_post_path = os.path.join(self.im_post_path, self.images_names[index])
        chat_path = os.path.join(self.chats_path, self.images_names[index].split(".")[0]+".txt")
        # Open the image 
        im_pre = Image.open(im_pre_path)
        im_post = Image.open(im_post_path)
        # Open the chat
        with open(chat_path, "r") as file:
            chat = file.read()
        return self.images_names[index], im_pre, im_post, chat
    