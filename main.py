'''
This piece of code is inteded to be used to generate a dialogue between the questioner AI and the answerer AI.
It can use full image or cropped image to generate the dialogue.
'''

from utils.chat import *
from utils.dataset import ChatSet, SummarySet, CDSet
import os
from tqdm import tqdm
import torch
from utils.dataset import custom_collate
import pickle
from transformers import GenerationConfig
import json
from utils.post_process_chats import chats_postprocessing

def chat_on_images(llms_params:dict, dataset_params:dict):
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
    dialogue_steps = 2
    #########################################
    with torch.no_grad():
        for i in range(dialogue_steps):
            chat = Chatter()
            print("Step {}".format(i))
            # ANSWERING 
            print("Creating the dataset in answering mode")
            dataset = ChatSet(dataset_params["dataset_path"], dataset_params["chats_path"], mode="answering")
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn = custom_collate)
            # Loading the answerer model 
            chat.load_lmm(llms_params["answerer_type"], llms_params["answerer_model"], llms_params["answerer_device"])
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
                        
                break
            del chat.multimodal_model
            torch.cuda.empty_cache()
            # QUESTIONING 
            print("Creating the dataset in questioning mode")
            dataset = ChatSet(dataset_params["dataset_path"], dataset_params["chats_path"], mode="questioning")
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn = custom_collate)
            # Loading the answerer model 
            chat.load_llm(llms_params["questioner_type"], llms_params["questioner_model"], llms_params["questioner_device"])
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
                
                break
            
            del chat.language_model # Maybe the line below is sufficient, to verify
            del chat
            torch.cuda.empty_cache()

def summarize_chats(llms_params:dict, path_dict_chats:str):
    '''
    Given some llms_params for generation and a dict of chats in the form {img_name: chat_list}, it generates a summary of the chats.
    Input:
    llms_params: dict
        The parameters for the LLM model.
    path_dict_chats: stri
        The path of the dictionary of chats (dictionary should be in the form {img_name: chat_list})
    Returns 
    summaries: dict
        The dictionary of summaries in the form {img_name: summary}
    '''
    assert llms_params["summarizer_type"] in ["vicuna"], "Summarizer not supported"
    # Sanity checks on dataset params
    assert os.path.exists(path_dict_chats), "dict_chats not found"
    chat = Chatter()
    dataset = SummarySet(path_dict_chats)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    chat.load_llm(llms_params["summarizer_type"], llms_params["summarizer_model"], llms_params["summarizer_device"])
    gen_cfg = GenerationConfig.from_pretrained(llms_params["summarizer_model"])
    gen_cfg.max_new_tokens=200
    gen_cfg.do_sample=True
    gen_cfg.temperature=0.6
    gen_cfg.top_p=0.95
    gen_cfg.top_k=40
    gen_cfg.repetition_penalty=1.1
    
    results = {}
    for batch in tqdm(dataloader):
        img_names, prompts = batch
        out = chat.call_vicuna(prompts, gen_cfg, task="summarization")
        for i in range(len(img_names)):
            results[img_names[i]] = out[i]
    
    # Save the dict of summaries
    with open("summaries.json", "w") as file:
        json.dump(results, file, indent=4)
    
    return

def generate_cd(llms_params:dict, path_dict_summaries:str):
    '''
    Given some llms_params for generation and a dict of summaries in the form {img_name: summary}, it generates a dict of change captions.
    Input:
    llms_params: dict
        The parameters for the LLM model.
    path_dict_summaries: str
        The path of the dictionary of summaries (dictionary should be in the form {img_name: summary})
    Returns 
    cds: dict
        The dictionary of change captions in the form {img_name: change_caption}
    '''
    assert llms_params["changecaptioner_type"] in ["vicuna"], "Change captioner not supported"
    # Sanity checks on dataset params
    assert os.path.exists(path_dict_summaries), "dict of summaries not found"
    chat = Chatter()
    dataset = CDSet(path_dict_summaries)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    chat.load_llm(llms_params["changecaptioner_type"], llms_params["changecaptioner_model"], llms_params["changecaptioner_device"])
    gen_cfg = GenerationConfig.from_pretrained(llms_params["changecaptioner_model"])
    gen_cfg.max_new_tokens=200
    gen_cfg.do_sample=True
    gen_cfg.temperature=0.6
    gen_cfg.top_p=0.95
    gen_cfg.top_k=40
    gen_cfg.repetition_penalty=1.1
    
    results = {}
    for batch in tqdm(dataloader):
        img_names, prompts = batch
        out = chat.call_vicuna(prompts, gen_cfg, task="change_captioning")
        for i in range(len(img_names)):
            results[img_names[i]] = out[i]

    # Save the dict of summaries
    with open("cds.json", "w") as file:
        json.dump(results, file, indent=4)
    
    return
        
if __name__ == "__main__":
    llms_params = {
        "answerer_type": "blip2",
        "answerer_model": "FlanT5 XXL", # To revise
        "answerer_device": "cuda:1",
        "dialogue_steps": 10, # How many rounds of dialogue to generate
        "questioner_type": "vicuna",
        "questioner_model": "TheBloke/vicuna-13B-v1.5-GPTQ", # lmsys/vicuna-7b-v1.5, lmsys/vicuna-13b-v1.5
        "questioner_device": "cuda:1",
        "summarizer_type": "vicuna",
        "summarizer_model": "TheBloke/vicuna-13B-v1.5-GPTQ",
        "summarizer_device": "cuda:1",
        "changecaptioner_type": "vicuna",
        "changecaptioner_model": "TheBloke/vicuna-13B-v1.5-GPTQ",
        "changecaptioner_device": "cuda:1",
    }
    
    dataset_params = {
        "dataset_path": "/media/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test",
        "chats_path": "chats_cache/",
        "crop": False,
        "area_threshold" : 5,
        "use_labels": True, # If true, it can use the labels to crop the image in the area of the biggest change
    }
    print("################### Starting chatting ###################")
    chat_on_images(llms_params, dataset_params)
    print("################### Finished chatting ###################")
    chats_postprocessing("chats_cache", "chats_postprocessed.json")
    chats_path = "chats_postprocessed.json"
    print("################### Starting summarizing ###################")
    summarize_chats(llms_params, chats_path)
    print("################### Finished summarizing ###################")
    summaries_path = "summaries.json"
    print("################### Starting generating change description ###################")
    generate_cd(llms_params, summaries_path)
    print("################### Finished generating change description ###################")
    print("Entire process finished!")