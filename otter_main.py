'''
This file contains all the functions and code to run otter based approaches for change captioning.

Approach 1 
Otter is used to directly extract the change paragraph from the two images.
Approach 2
Otter is used to extract single descriptions of the images, and then vicuna is used to analyze the descriptions and extract the changes.
Approach 3
Otter is used to answer questions about the two images in a dialogue setting. Vicuna is used to generate questions. After the dialogue, vicuna summarizes the changes in a descriptive paragraph.
'''

import os
from tqdm import tqdm
import json
import pickle

import torch

from utils.dataset import MultitemporalImageSet, CDSet, ChatSet, SummarySet
from utils.chat import Chatter
import transformers
from transformers import GenerationConfig
from utils.conversation import Conversation

template_questions = ["Have there been any changes in the appearence of buildings between the two images?",
"Are there any signs of damage to the buidings between the two images?",
"Have new buildings been constructed in the area?",
"Have buildings been removed between the two images?",
"Have parts been added to the existing buidings between the two images?",
"Are there signs of new construction sites in the area?",
"Have new playgrounds appeared in the area?",
"Have there been any changes in the appearence of playgrounds between the two images?",
"Have playgrounds been removed between the two images?",
"Are there any signs of damage to the playgrounds between the two images?"]

template_only_buidlings = ["Have there been any changes in the appearence of buildings between the two images?",
"Are there any signs of damage to the buidings between the two images?",
"Have new buildings been constructed in the area?",
"Have buildings been removed between the two images?",
"Have parts been added to the existing buidings between the two images?"]

template_questions_added_ = ["Have there been any changes in the appearence of buildings between the two images?",
"Are there any signs of damage to the buidings between the two images?",
"Have new buildings been constructed in the area?",
"Have buildings been removed between the two images?",
"Have parts been added to the existing buidings between the two images?",
"Are there signs of new construction sites in the area?",
"Have new playgrounds appeared in the area?",
"Have there been any changes in the appearence of playgrounds between the two images?",
"Have playgrounds been removed between the two images?",
"Are there any signs of damage to the playgrounds between the two images?"]



def create_description(images_path, approach:str, device="cuda:0"):
    if approach=="otter_direct":
        # Steps
        # 1. Load the model -> otter
        # 2. Load the dataset of images -> the dataset should return batches of im1,im2.
        # 3. Generate the descriptions
        
        chat = Chatter()
        multimodal_model = "luodian/OTTER-Image-MPT7B"
        # 1. Load the model (otter)
        chat.load_lmm("otter", multimodal_model, device)
        image_processor = transformers.CLIPImageProcessor()
        
        # 2. Load the dataset and create the dataloader
        assert os.path.isdir(images_path), "Double check the directory path"
        dataset = MultitemporalImageSet(images_path=images_path, image_processor=image_processor, method=1)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        
        # model.config.text_config.eos_token_id = 50277
        # 3. Generate the descriptions
        results_otter_direct = {}
        
        # Getting generation config
        gen_cfg_aswerer = GenerationConfig.from_pretrained(multimodal_model, config_file_name="config.json")
        gen_cfg_aswerer.text_config["do_sample"] = True
        gen_cfg_aswerer.text_config["top_p"] = 0.95
        gen_cfg_aswerer.text_config["temperature"] = 0.2
        
        # Encode the prompt
        prompt = ["<image>User: Describe the changes between these two views of the same area at different times. Describe only the changes, if there are no noticeable changes, say so. GPT:<answer>"]
        
        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader)):
                image_names, vision_x = batch
                assert len(image_names) == 1, "For now, the batch size must be 1"
                
                vision_x = vision_x.to(dtype=chat.multimodal_model.dtype)
                out = chat.call_otter(vision_x=vision_x, prompts=prompt, generation_config=gen_cfg_aswerer)
                
                results_otter_direct[image_names[0]] = out  # to change when implementing batch inference
                
        # Return the results   
        return results_otter_direct
    
    elif approach=="otter_indirect":
        # Steps
        # 1. Load the model -> otter
        # 2. Load the dataset of images -> the dataset should return batches of im1,im2.
        # 3. Generate the descriptions of the single images
        # 4. Load vicuna
        # 5. Load the dataset of descriptions
        # 6. Use vicuna to extract the changes from the descriptions
        chat = Chatter()
        multimodal_model = "luodian/OTTER-Image-MPT7B"
        # 1. Load the model (otter)
        chat.load_lmm("otter", multimodal_model, device)
        image_processor = transformers.CLIPImageProcessor()
        # 2. Load the dataset and create the dataloader
        assert os.path.isdir(images_path), "Double check the directory path"
        dataset = MultitemporalImageSet(images_path=images_path, image_processor=image_processor, method=2)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        
        # 3. Generate the descriptions
        results_otter_indirect = {}
        
        # Getting generation config
        gen_cfg_aswerer = GenerationConfig.from_pretrained(multimodal_model, config_file_name="config.json")
        gen_cfg_aswerer.text_config["do_sample"] = True
        gen_cfg_aswerer.text_config["top_p"] = 0.95
        gen_cfg_aswerer.text_config["temperature"] = 0.2
        
        # Encode the prompt
        prompt = ["<image>User: Describe in detail this satellite image. GPT:<answer>"]
        
        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader)):
                image_names, vision_1, vision_2 = batch
                # Convert tensors to the model's data type
                vision_1 = vision_1.to(dtype=chat.multimodal_model.dtype)
                vision_2 = vision_2.to(dtype=chat.multimodal_model.dtype)
                # Generate
                out_1 = chat.call_otter(vision_x=vision_1, prompts=prompt, generation_config=gen_cfg_aswerer)
                out_2 = chat.call_otter(vision_x=vision_2, prompts=prompt, generation_config=gen_cfg_aswerer)
                
                results_otter_indirect[image_names[0]] = (out_1,out_2)  # to change when implementing batch inference
                # Save the results dictionary as a json file
                with open("results_otter/intermediate_results_otter_indirect.json", "w") as f:
                    json.dump(results_otter_indirect, f, indent=4)
        
        chat.del_lmm()
        
        # 4.
        model = "TheBloke/vicuna-13B-v1.5-GPTQ"
        chat.load_llm("vicuna", model, device)
        
        # 5. 
        path_descriptions = "results_otter/intermediate_results_otter_indirect.json"
        dataset = CDSet(path_dict_descriptions=path_descriptions)
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
        
        gen_cfg = GenerationConfig.from_pretrained(model)
        gen_cfg.max_new_tokens=200
        gen_cfg.do_sample=True
        gen_cfg.temperature=0.6
        gen_cfg.top_p=0.95
        gen_cfg.top_k=40
        gen_cfg.repetition_penalty=1.1
        
        results_otter_indirect = {}
        
        for batch in tqdm(dataloader):
            image_names, prompts = batch
            out_pre = chat.call_vicuna(prompts=prompts, generation_config=gen_cfg, task="change_captioning")
            for i in range(len(image_names)):
                name = image_names[i]
                results_otter_indirect[name] = out_pre[i].strip().replace("\n","")
            
        # Return the results
        return results_otter_indirect
        
    elif approach=="otter_chat":
        template=True
        chat_cache = "chats_cache_template/"
        questioner_name = "TheBloke/vicuna-13B-v1.5-GPTQ"
        answerer_name = "luodian/OTTER-Image-MPT7B"
        
        gen_cfg_questioner = GenerationConfig.from_pretrained(questioner_name)
        gen_cfg_questioner.max_new_tokens=200
        gen_cfg_questioner.do_sample=True
        gen_cfg_questioner.temperature=0.6
        gen_cfg_questioner.top_p=0.95
        gen_cfg_questioner.repetition_penalty=1.1
        
        gen_cfg_aswerer = GenerationConfig.from_pretrained(answerer_name, config_file_name="config.json")
        gen_cfg_aswerer.text_config["do_sample"] = True
        gen_cfg_aswerer.text_config["top_p"] = 0.95
        gen_cfg_aswerer.text_config["temperature"] = 0.2
        
        # Steps:
        # 1. Load the large language model -> vicuna
        # 2. Load the dataset of chats. Each chat is a list of questions and answers.
        # 3. Generate in batch the next question (the first will be based on the user intentions)
        # 4. Load the dataset of images + question. 
        # 5. Answer all the questions using otter
        # 6. Repeat n times. 
        n_rounds = 10
        # 1. Load the large language model -> vicuna
        chat = Chatter()
        if not template:
            # 2. Load the dataset for questioning and create the dataloader
            q_dataset = ChatSet(images_path=images_path, chats_cache=chat_cache, mode="questioning")
            q_dataloader = torch.utils.data.DataLoader(q_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        
        # 2. Load the dataset for answering and create the dataloader
        image_processor = transformers.CLIPImageProcessor()
        a_dataset = ChatSet(images_path=images_path, chats_cache=chat_cache, mode="answering",image_processor=image_processor)
        a_dataloader = torch.utils.data.DataLoader(a_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        
        conversation = Conversation()
        user_intention = ""
        conversation.generate_first_conversations(image_path=image_path, user_intention=user_intention, chat_cache=chat_cache)
        
        for i in range(n_rounds):
            if not template:
                # Generate in batch the questions using Vicuna
                chat.load_llm("vicuna", questioner_name, device)
                for batch in tqdm(q_dataloader):
                    image_names, prompts = batch
                    # Generate 
                    question = chat.call_vicuna(prompts=prompts, generation_config=gen_cfg_questioner)
                    question = chat.trim_output(question, task="questioning")
                    # Save the chats   
                    for i in range(len(image_names)):
                        name = image_names[i]
                        # Open the messages
                        with open(chat_cache+name.split(".")[0]+".pkl", "rb") as file:
                            messages = pickle.load(file)
                        # append the new messages
                        messages.append(["ASSISTANT",question[i]])
                        # Dump the messages
                        with open(chat_cache+name.split(".")[0]+".pkl", "wb") as file:
                            pickle.dump(messages, file)
                chat.del_llm()
            else:
                # Get templates 
                template_question = template_questions[i]
                for img_name in os.listdir(images_path+"/im1"):
                    with open(chat_cache+img_name.split(".")[0]+".pkl", "rb") as file:
                        messages = pickle.load(file)
                        messages.append(["ASSISTANT",template_question])

                    with open(chat_cache+img_name.split(".")[0]+".pkl", "wb") as file:
                        pickle.dump(messages, file)
                        
            # Load otter model
            if not template or i==0:
                chat.load_lmm("otter", answerer_name, device)
            # Answer the questions using Otter
            for batch in tqdm(a_dataloader):
                image_names, vision_x, prompts = batch 
                # Convert tensors to the model's data type
                vision_x = vision_x.to(dtype=chat.multimodal_model.dtype)
                # Generate
                answer = chat.call_otter(vision_x=vision_x, prompts=prompts, generation_config=gen_cfg_aswerer)
                # Save the chats
                for i in range(len(image_names)):
                    name = image_names[i]
                    # Open the messages
                    with open(chat_cache+name.split(".")[0]+".pkl", "rb") as file:
                        messages = pickle.load(file)
                    # append the new messages
                    messages.append(["USER",answer])
                    # Dump the messages
                    with open(chat_cache+name.split(".")[0]+".pkl", "wb") as file:
                        pickle.dump(messages, file)
            
            if not template:
                chat.del_lmm()

        # Load the llm
        chat.load_llm("vicuna", questioner_name, device)
        
        results = dict()
        for image in tqdm(os.listdir(chat_cache)):
            conversation = Conversation()
            with open(chat_cache+image, "rb") as file:
                messages = pickle.load(file)
            conversation.load_messages(messages)
            prompt = conversation.generate_summary_prompt_vicuna()
            out = chat.call_vicuna(prompts=prompt, generation_config=gen_cfg_questioner)
            
            # Trim the output 
            out = out[0].split("ASSISTANT:")[-1].strip()
            results[image.split(".")[0]] = out
        
        return results
    else:
        raise Exception("Find a suitable approach to test!")

if __name__ == "__main__":
    approach = "otter_direct"
    #image_path = "/media/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test"
    image_path = "images_test"

    results = create_description(images_path=image_path,approach=approach,device="cuda:0")
    
    # Save the results dictionary as a json file 
    with open("results_otter/results_"+approach+"_template_new.json", "w") as f:
        json.dump(results, f, indent=4)