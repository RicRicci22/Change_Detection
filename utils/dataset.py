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
        self.chats_path = chats_path
        self.images_names = os.listdir(images_path)
    
    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, index) -> Any:
        image_path = os.path.join(self.images_path, self.images_names[index])
        chat_path = os.path.join(self.chats_path, self.images_names[index].split(".")[0]+".txt")
        # Open the image 
        image = Image.open(image_path)
        # Open the chat
        with open(chat_path, "r") as file:
            chat = file.read()
        return image, chat
    