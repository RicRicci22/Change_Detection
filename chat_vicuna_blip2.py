'''
This piece of code is inteded to be used to generate a dialogue between the questioner AI and the answerer AI.
It can use full image or cropped image to generate the dialogue.
'''

from utils.chat import *
from utils.dataset import ChatSet
import os
from tqdm import tqdm
import torch
from utils.dataset import custom_collate
import pickle
from transformers import GenerationConfig

def main(llms_params:dict, dataset_params:dict):
    '''
    This function run the dialogue and store the result.
    
    Method 
    Answerer in batch on all images
    Questioner in batch on all images 
    Answerer in batch on all images
    Questioner in batch on all images
    .. 
    .. 
    
    Parameters:
    llm_params: dict
        The parameters for the LLM model.
    dataset_params: dict
        The parameters for the dataset, and in general the handling of the images.
    '''
    # Sanity checks on llm params
    assert llms_params["answerer_type"] in ["blip2"], "Answerer not supported"
    assert llms_params["questioner_type"] in ["vicuna"], "Questioner not supported"
    
    # Sanity checks on dataset params
    assert os.path.exists(dataset_params["dataset_path"]), "Path of dataset not found"
    assert os.path.exists(os.path.join(dataset_params["dataset_path"],"im1")), "Folder of pre-change images not found"
    assert os.path.exists(os.path.join(dataset_params["dataset_path"],"im2")), "Folder of post-change images not found"
    assert dataset_params["crop"] in [True, False], "Define crop as True or False"
    
    if dataset_params["crop"] and dataset_params["use_labels"]:
        assert os.path.exists(dataset_params["dataset_path_labels_1"]), "Folder of pre-change labels not found"
        assert os.path.exists(dataset_params["dataset_path_labels_2"]), "Folder of post-change labels not found"
    elif dataset_params["crop"]:
        # in this case it should perform prediction and then crop
        pass 
    
    ########## Initialize the chat ##########
    dialogue_steps = llms_params["dialogue_steps"]
    #########################################
    
    for i in range(dialogue_steps):
        chat = Chatter()
        print("Step {}".format(i))
        # ANSWERING 
        print("Creating the dataset in answering mode")
        dataset = ChatSet(dataset_params["dataset_path"], dataset_params["chats_path"], mode="answering")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn = custom_collate)
        # Loading the answerer model 
        chat.load_answerer(llms_params["answerer_type"], llms_params["answerer_model"], llms_params["answerer_device"])
        print("Answering questions in batch")
        for batch in tqdm(dataloader):
            img_names, imgs_pre, prompt_pre, chat_pre, imgs_post, prompt_post, chat_post = batch
            # print("answering")
            # print(prompt_pre)
            out_pre = chat.call_blip2(imgs_pre, prompt_pre)
            out_post = chat.call_blip2(imgs_post, prompt_post)
            # Save the results in the chats_cache
            for i in range(len(out_pre)):
                chat_pre[img_names[i]].append(["USER", out_pre[i]])
                chat_post[img_names[i]].append(["USER", out_post[i]])
                # Save the lists in the cache
                with open(os.path.join(dataset_params["chats_path"], img_names[i].split(".")[0]+"_pre.pkl"), "wb") as file:
                    pickle.dump(chat_pre[img_names[i]], file)
                with open(os.path.join(dataset_params["chats_path"], img_names[i].split(".")[0]+"_post.pkl"), "wb") as file:
                    pickle.dump(chat_post[img_names[i]], file)
                    
        
        del chat.answerer
        torch.cuda.empty_cache()
        # QUESTIONING 
        print("Creating the dataset in questioning mode")
        dataset = ChatSet(dataset_params["dataset_path"], dataset_params["chats_path"], mode="questioning")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn = custom_collate)
        # Loading the answerer model 
        chat.load_questioner(llms_params["questioner_type"], llms_params["questioner_model"], llms_params["questioner_device"])
        print("Asking questions in batch")
        ############# SET GENERATION CONFIG #####################################
        gen_cfg = GenerationConfig.from_pretrained(llms_params["questioner_model"])
        gen_cfg.max_new_tokens=200
        gen_cfg.do_sample=True
        gen_cfg.temperature=0.6
        gen_cfg.top_p=0.95
        gen_cfg.top_k=40
        gen_cfg.repetition_penalty=1.1
        #########################################################################
        for batch in tqdm(dataloader):
            img_names, imgs_pre, prompt_pre, chat_pre, imgs_post, prompt_post, chat_post = batch
            # print("questioning")
            # print(prompt_pre)
            out_pre = chat.call_vicuna(prompt_pre, gen_cfg)
            out_post = chat.call_vicuna(prompt_post, gen_cfg)
            # Save the results in the chats_cache
            for i in range(len(out_pre)):
                chat_pre[img_names[i]].append(["ASSISTANT", out_pre[i]])
                chat_post[img_names[i]].append(["ASSISTANT", out_post[i]])
                # Save the lists in the cache
                with open(os.path.join(dataset_params["chats_path"], img_names[i].split(".")[0]+"_pre.pkl"), "wb") as file:
                    pickle.dump(chat_pre[img_names[i]], file)
                with open(os.path.join(dataset_params["chats_path"], img_names[i].split(".")[0]+"_post.pkl"), "wb") as file:
                    pickle.dump(chat_post[img_names[i]], file)
            
            
        del chat.questioner
        del chat
        torch.cuda.empty_cache()
    
if __name__ == "__main__":
    llms_params = {
        "answerer_type": "blip2",
        "answerer_model": "FlanT5 XXL", # To revise
        "answerer_device": "cuda:0",
        "dialogue_steps": 10, # How many rounds of dialogue to generate
        "questioner_type": "vicuna",
        "questioner_model": "TheBloke/vicuna-13B-v1.5-GPTQ", # lmsys/vicuna-7b-v1.5, lmsys/vicuna-13b-v1.5
        "questioner_device": "cuda:0",
    }
    
    dataset_params = {
        "dataset_path": "/media/riccardoricci/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test",
        "chats_path": "chats_cache/",
        "crop": False,
        "area_threshold" : 5,
        "use_labels": True, # If true, it can use the labels to crop the image in the area of the biggest change
    }
    
    
    # with open("chats_cache/00017_post.pkl", "rb") as file:
    #     conv = pickle.load(file)
        
    # print(conv)
    main(llms_params, dataset_params)