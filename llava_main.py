'''
This module uses the llava model. 
It can be used in two modes: direct and indirect.
Direct mode takes two images and returns the change caption.  -> to develop
Undirect mode takes two images, return a caption from each image and then uses vicuna to extract the changes.
'''
import torch
import json
from tqdm import tqdm

# from LLaVA.llava.model.builder import load_pretrained_model
# from LLaVA.llava.utils import disable_torch_init
# from LLaVA.llava.mm_utils import get_model_name_from_path, tokenizer_image_token, tokenizer_image_token, KeywordsStoppingCriteria
# from LLaVA.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
# from LLaVA.llava.conversation import conv_templates
from utils.dataset import LlavaDataset
from utils.chat import Chatter
from utils.dataset import CDSet
from transformers import GenerationConfig

def main_llava(llms_params:dict, dataset_params:dict, mode="direct"):
    '''
    Main function, launch the change caption generation using llava model
    Input: 
        llms_params: dict, parameters for the llava model
        dataset_params: dict, parameters for the dataset
    Output:
        None
    '''
    results = {}
    # PSEUDO CODE
    # 1. Load the llava model
    # 2. Load the dataset (choose between direct prompt o undirect prompt)
    # 3. Call generate
    # 4. If direct prompt -> stop and save the results
    # 5. If undirect prompt -> call vicuna, process and save the results
    # 1 #####################
    # disable_torch_init()
    # model_name = get_model_name_from_path(llms_params["captioner_model"])
    # tokenizer, model, image_processor, context_len = load_pretrained_model(llms_params["captioner_model"], None, model_name, load_8bit=False, load_4bit=True, device=llms_params["captioner_device"])
    # print("Model context length: ", context_len)
    # # CHOOSE THE RIGHT CONVERSATION FORMAT
    # if 'llama-2' in model_name.lower():
    #     conv_mode = "llava_llama_2"
    # elif "v1" in model_name.lower():
    #     conv_mode = "llava_v1"
    # elif "mpt" in model_name.lower():
    #     conv_mode = "mpt"
    # else:
    #     conv_mode = "llava_v0"

    # 2 #####################
    if mode == "create_summary":
        dataset = LlavaDataset(dataset_params["dataset_path"], image_processor=image_processor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
        # Get the stopping criteria
        # PROMPT
        conv = conv_templates[conv_mode].copy()
        prompt = "Give a detailed description of this satellite image."
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(llms_params["captioner_device"])
        keywords = ["</s>"]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        temperature = 0.7
        max_new_tokens = 512
        for batch in tqdm(dataloader):
            image_names, image_tensor = batch
            input_ids = torch.stack([input_ids[0]]*len(image_names))
            image_tensor = image_tensor.to(llms_params["captioner_device"],dtype=torch.float16)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    streamer=None,
                    stopping_criteria=[stopping_criteria])
            
            outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip().replace("\n", "")
            results[image_names[0]] = outputs
            
        # Save the dict
        with open("results_llava/results_indirect.json", "w") as file:
            json.dump(results, file, indent=4)
        
    elif mode=="create_cd_captions":
        path_dict_summaries = "results_llava/results_indirect.json"
        chat = Chatter()
        dataset = CDSet(path_dict_summaries)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        chat.load_llm(llms_params["changecaptioner_type"], llms_params["changecaptioner_model"], llms_params["changecaptioner_device"])
        gen_cfg = GenerationConfig.from_pretrained(llms_params["changecaptioner_model"])
        gen_cfg.max_new_tokens=200
        gen_cfg.do_sample=True
        gen_cfg.temperature=0.6
        gen_cfg.top_p=0.95
        gen_cfg.top_k=40
        gen_cfg.repetition_penalty=1.1
        for batch in tqdm(dataloader):
            img_names, prompts = batch
            out = chat.call_vicuna(prompts, gen_cfg, task="change_captioning")
            for i in range(len(img_names)):
                results[img_names[i]] = out[i]

        # Save the dict of summaries
        with open("results_llava/cds.json", "w") as file:
            json.dump(results, file, indent=4)
        

if __name__ == "__main__": 
    llms_params = {
        "captioner_type": "llava",
        "captioner_model": "liuhaotian/llava-v1.5-7b",
        "captioner_device": "cuda:0",
        "changecaptioner_type": "vicuna",
        "changecaptioner_model": "TheBloke/vicuna-13B-v1.5-GPTQ",
        "changecaptioner_device": "cuda:0",
    }
    
    dataset_params = {
        "dataset_path": "/media/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test",
    }
    
    # mode = "create_summary"
    # main_llava(llms_params, dataset_params, mode)
    mode = "create_cd_captions"
    main_llava(llms_params, dataset_params, mode)
    
    