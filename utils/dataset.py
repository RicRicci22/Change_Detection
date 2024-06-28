import pickle
import json
import os 
from typing import Any

from PIL import Image

from torch.utils.data import Dataset
from utils.conversation import Conversation

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

class MultitemporalImageSet(Dataset):
    '''
    This dataset return couples of multitemporal images given an index --- for otter method 1 
    '''
    def __init__(self, images_path, image_processor, method=1):
        super(MultitemporalImageSet, self).__init__()
        # images_path : str -> path of the folder containing all the images
        # image_processor : CLIPImageProcessor -> image processor to use to process the images
        self.images_path = images_path
        # Get the path of the images pre change
        self.im_pre_path = os.path.join(images_path, "im1")
        # Get the path of the images post change
        self.im_post_path = os.path.join(images_path, "im2")
        self.images_names = os.listdir(self.im_pre_path)
        self.image_processor = image_processor
        self.method = method
        
    def __len__(self):
        return len(self.images_names)
    
    def __getitem__(self, index: int) -> Any:
        # Read the two images and return PIL objects 
        image_name = self.images_names[index]
        im_pre_path = os.path.join(self.im_pre_path, image_name)
        im_post_path = os.path.join(self.im_post_path, image_name)
        image_1 = Image.open(im_pre_path)
        image_2 = Image.open(im_post_path)
        
        if self.method == 1:    
            vision_x = self.image_processor.preprocess([image_1,image_2], return_tensors="pt")["pixel_values"].unsqueeze(0)
            return image_name, vision_x
        if self.method == 2:
            vision_x = self.image_processor.preprocess([image_1], return_tensors="pt")["pixel_values"].unsqueeze(1)
            vision_y = self.image_processor.preprocess([image_2], return_tensors="pt")["pixel_values"].unsqueeze(1)
            return image_name, vision_x, vision_y
        else:
            raise NotImplementedError("Method not implemented")
class ChatSet(Dataset):
    '''
    This dataset class serves when using the chat approach. It handle image loading and chat loading. 
    '''
    def __init__(self, images_path, chats_cache, context = 100, mode="questioning", image_processor=None):
        super(ChatSet, self).__init__()
        # images_path : str -> path of the folder containing all the images
        # chats_path : str -> path of the folder containing all the chats
        self.images_path = images_path
        # Get the path of the images pre change
        self.im_pre_path = os.path.join(images_path, "im1")
        # Get the path of the images post change
        self.im_post_path = os.path.join(images_path, "im2")
        # Get the path of the chats
        self.chats_cache = chats_cache
        self.image_processor = image_processor
        self.images_names = os.listdir(self.im_pre_path)
        self.context = context
        self.mode = mode
    
    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, index) -> Any:
        conversation=Conversation()
        if self.mode == "questioning":
            # I need just the chats
            chat_path = os.path.join(self.chats_cache, self.images_names[index].split(".")[0]+".pkl")
            
            if(os.path.exists(chat_path)):
                with open(chat_path, "rb") as file:
                    chat = pickle.load(file)
                conversation.load_messages(chat)
            
            # Get question prompt 
            prompt = conversation.generate_prompt_vicuna()
            
            return self.images_names[index], prompt
        
        elif self.mode == "answering":
            # I need the images and the last question
            im_pre_path = os.path.join(self.im_pre_path, self.images_names[index])
            im_post_path = os.path.join(self.im_post_path, self.images_names[index])
            chat_path = os.path.join(self.chats_cache, self.images_names[index].split(".")[0]+".pkl")
            # Open and process the images
            image_1 = Image.open(im_pre_path)
            image_2 = Image.open(im_post_path)
            vision_x = self.image_processor.preprocess([image_1,image_2], return_tensors="pt")["pixel_values"].unsqueeze(0)
            # Open the chat
            if(os.path.exists(chat_path)):
                with open(chat_path, "rb") as file:
                    chat = pickle.load(file)
                conversation.reset_messages()
                conversation.load_messages(chat)
                # Get answer prompt 
                prompt = conversation.get_answer_prompt(model="otter", context=1)
            else:
                raise RuntimeError("The chat file does not exist, impossible situation, check!")
            
            return self.images_names[index], vision_x, prompt

class SummarySet(Dataset):
    def __init__(self, chats_path):
        # Open the dict of chats
        with open(chats_path, "rb") as file:
            self.chats = json.load(file)
            self.image_names = list(self.chats.keys())
            
        self.conversation = Conversation()
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        self.conversation.load_messages(self.chats[image_name])
        prompt = self.conversation.get_summary_prompt(model="vicuna", context=100)
        return image_name, prompt
    

class CDSet(Dataset):
    def __init__(self, path_dict_descriptions):
        with open(path_dict_descriptions, "rb") as file:
            self.descriptions = json.load(file)
        
        self.image_names = list(self.descriptions.keys())
        self.conversation = Conversation()
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        description1, description2 = self.descriptions[image_name]
        prompt = self.conversation.get_cd_prompt(description1, description2, model="vicuna")
        return image_name, prompt
    
class EvaluationDataset(Dataset):
    '''
    Dataset to handle evaluation of the model using reasoning capabilities of a LLM.
    Inputs:
        - path_gen_change_desc: str -> path to the json file cotaining the change descriptions.
        
    '''
    def __init__(self, path_gen_change_desc:str, gt_descriptions:str=None):
        
        self.prompt_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Here is a paragraph describing some changes: \"<paragraph>\". In the paragraph, are there references to the fact that <fact>? ASSISTANT: Short answer:"
        
        with open(path_gen_change_desc, "r") as file:
            self.change_desc = json.load(file)
        
        # Find all the possible changes between two classes
        if gt_descriptions is not None:
            self.changes = dict()
            # Read the change captions of levir
            with open(gt_descriptions, "r") as file:
                gt = json.load(file)
            
            for dict_image in gt["images"]:
                if dict_image["filepath"]=="test":
                    image = dict_image["filename"]
                    for sentence_dict in dict_image["sentences"]:
                        sentence = sentence_dict["raw"][:-1].strip()+"."
                        try:
                            self.changes[image].append(sentence)
                        except KeyError:
                            self.changes[image] = [sentence]
        else:
            self.changes = list()
            classes = ['water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
            for i in range(len(classes)):
                for j in range(len(classes)):
                    if i != j:
                        self.changes.append("a " + classes[i]+" area has transformed into a "+classes[j]+" area.")
        
        
        samples = list()
        
        for image, change_desc in self.change_desc.items():
            if type(self.changes) == dict:
                for change in self.changes[image+".png"]:
                    samples.append((image, change_desc, change))
            elif type(self.changes) == list:
                for change in self.changes:
                    samples.append((image, change_desc, change))
        
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, change_desc, change = self.samples[index]
        
        prompt = self.prompt_template.replace("<fact>", change[:-1])
        prompt = prompt.replace("<paragraph>", change_desc[:-1])

        return image, prompt, change
    
    
if __name__=="__main__":
    dataset = EvaluationDataset("results_otter/results_levir_otter_chat_open_guided_buildings_sports_fields.json", "levir_cc/LevirCCcaptions.json")
    
    
    print(dataset[0])
