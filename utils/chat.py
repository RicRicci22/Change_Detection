from .blip2 import Blip2
from .vicuna import load_hug_model
import torch

FIRST_QUESTION = "Describe this remote sensed image in detail."
LLM_MODELS = ["chatgpt", "vicuna"]
LMM_MODELS = ["blip2"]

def load_vicuna(model_name, device: str="cpu"):
    '''
    Load vicuna calling the load_hug_model function, which is a general function to load models from huggingface
    '''
    tokenizer, model = load_hug_model(model_name, device)

    return model, tokenizer


def load_blip2(model_name, device: str="cpu"):
    '''
    Load blip by custom Blip 2 class, TODO transform using the general function load_hug_model
    '''
    model = Blip2(model_name, device=device, bit4=True)

    return model

def load_language_model(model_type: str, model_name:str, device: str="cpu"):
    '''
    Load the questioner model, by first checking if it is a valid model
    '''
    if model_type in LLM_MODELS:
        if model_type == "chatgpt":
            raise NotImplementedError("chatgpt is not implemented yet")
        elif model_type == "vicuna":
            model, tokenizer = load_vicuna(model_name, device)
    else:
        raise ValueError("{} is not a valid question model".format(model_name))

    return model, tokenizer


def load_multimodal_model(model_type: str, model_name:str, device: str="cpu"):
    '''
    Load the answerer model, by first checking if it is a valid model
    '''
    if model_type in LMM_MODELS:
        if model_type == "blip2":
            model = load_blip2(model_name, device)
    else:
        raise ValueError("{} is not a valid question model".format(model_name))

    return model

class Chatter:
    """
    This class serves to include functions to call the various models
    """
    
    def __init__(
        self
    ):
        pass
        # if params is None:
        #     raise ValueError("No parameters given")
        # else:
        #     # Extract the info from the params
        #     questioner_type = params["questioner_type"] # Options: vicuna
        #     questioner_model = params["questioner_model"] # Options: TheBloke/vicuna-13B-v1.5-GPTQ, lmsys/vicuna-7b-v1.5, lmsys/vicuna-13b-v1.5
        #     questioner_device = params["questioner_device"] # Options: cuda:0, cuda:1, cpu
        #     self.questioner_context = params["questioner_context"] 
        #     # ANSWERER
        #     answerer_type = params["answerer_type"] # Options: blip2
        #     answerer_model = params["answerer_model"] # Options: flant5xxl
        #     answerer_device = params["answerer_device"] # Options: cuda:0, cuda:1, cpu 
        #     self.answerer_context = params["answerer_context"]

    def load_llm(self, llm_type, llm_model, device):
        '''
        Load large language model (can handle only text)
        This is used for the question generation part, the summarization and the extraction of the changes
        '''
        print("Loading large language model {}, version {}".format(llm_type, llm_model))
        self.language_model, self.llm_tokenizer = load_language_model(llm_type, llm_model, device)
        self.device_llm = device
    
    def load_lmm(self, lmm_type, lmm_model, device):
        '''
        Load large multimodal model (can handle both text and images)
        This is used either for answer to the questions looking at the image, or for direct extraction (eg. llava model)
        '''
        print("Loading large multimodal model {}, version {}".format(lmm_type, lmm_model))
        self.multimodal_model = load_multimodal_model(lmm_type, lmm_model, device)
        self.device_lmm = device
        
    def call_vicuna(self, prompts:list, generation_config:dict=None, task:str="questioning"):
        '''
        This function call the questioner model passing the parameters and the prompt. 
        The questioner model should implement the generate function (huggingface syntax).
        It can work in batch or not, depending on the length of prompts
        
        Inputs:
            - prompts: list of prompts to be passed to the questioner model
            - generation_config: generation configuration for model.generate() function
            - task: 
        '''
        # Set the generation config 
        if generation_config is not None:
            self.language_model.generation_config = generation_config
        ############################# GENERATE #############################
        input_ids = self.llm_tokenizer(prompts, padding=True, return_tensors='pt').input_ids.to(self.device_llm)
        outputs = self.language_model.generate(inputs=input_ids)
        outputs = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ############################# POST PROCESS #############################
        if task == "questioning":
            outputs = [output.split("ASSISTANT:")[-1].split("?")[0].strip()+"?" for output in outputs]
        elif task == "summarization" or task == "change_captioning":
            outputs = [output.split("ASSISTANT:")[-1].strip() for output in outputs]
        
        return outputs

    def call_blip2(self, images, prompts):
        '''
        Function to call the blip model with images and corresponding prompts. Can work in batch.
        Inputs: 
            - images: list of PIL images to be passed to the model
            - prompts: list of prompts to be passed to the model
        '''
        inputs = self.multimodal_model.blip2_processor(images, prompts, return_tensors="pt", padding=True).to(
            self.device_lmm, torch.float16
        )
        outputs = self.multimodal_model.blip2.generate(**inputs, max_new_tokens=100)
        answer = self.multimodal_model.blip2_processor.batch_decode(outputs, skip_special_tokens=True)
        return answer
