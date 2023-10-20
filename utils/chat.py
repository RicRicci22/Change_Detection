from utils.blip2 import Blip2
from utils.vicuna import load_hug_model
import torch

FIRST_QUESTION = "Describe this remote sensed image in detail."

QUESTIONER_MODELS = ["chatgpt", "vicuna"]
ANSWERER_MODELS = ["blip2"]
SUMMARIZERS = ["chatgpt", "vicuna"]


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

def load_questioner(model_type: str, model_name:str, device: str="cpu"):
    '''
    Load the questioner model, by first checking if it is a valid model
    '''
    if model_type in QUESTIONER_MODELS:
        if model_type == "chatgpt":
            raise NotImplementedError("chatgpt is not implemented yet")
        elif model_type == "vicuna":
            model, tokenizer = load_vicuna(model_name, device)
    else:
        raise ValueError("{} is not a valid question model".format(model_name))

    return model, tokenizer


def load_answerer(model_type: str, model_name:str, device: str="cpu"):
    '''
    Load the answerer model, by first checking if it is a valid model
    '''
    if model_type in ANSWERER_MODELS:
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

    def load_questioner(self, questioner_type, questioner_model, questioner_device):
        # TODO MODIFY IT
        # Load the questioner
        print("Loading questioner {}, model {}".format(questioner_type, questioner_model))
        self.questioner, self.q_tokenizer = load_questioner(questioner_type, questioner_model, questioner_device)
    
    def load_answerer(self, answerer_type, answerer_model, answerer_device):
        # TODO MODIFY IT
        # Load the questioner
        print("Loading answerer {}, model {}".format(answerer_type, answerer_model))
        self.answerer = load_answerer(answerer_type, answerer_model, answerer_device)
    
    # TODO load summarizer function
        
    def call_vicuna(self, prompts, generation_config):
        '''
        This function call the questioner model passing the parameters and the prompt. 
        The questioner model should implement the generate function (huggingface syntax).
        It can work in batch or not, depending on the length of the prompt list
        
        Inputs:
            - prompts: list of prompts to be passed to the questioner model
            - generation_config: generation configuration for model.generate() function
        '''
        # Set the generation config 
        self.questioner.generation_config = generation_config
        # Tokenize the prompts
        input_ids = self.q_tokenizer(prompts, padding=True, return_tensors='pt').input_ids.to(self.questioner.device)
        # Generate the outputs
        outputs = self.questioner.generate(inputs=input_ids)
        # Decode
        outputs = self.q_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Trim on the last question
        outputs = [output.split("ASSISTANT:")[-1].split("?")[0].strip()+"?" for output in outputs]
        
        return outputs

    def call_blip2(self, images, prompts):
        '''
        Function to call the blip model with images and corresponding prompts. Can work in batch.
        Inputs: 
            - images: list of PIL images to be passed to the model
            - prompts: list of prompts to be passed to the model
        '''
        inputs = self.answerer.blip2_processor(images, prompts, return_tensors="pt", padding=True).to(
            self.answerer.device, torch.float16
        )
        outputs = self.answerer.blip2.generate(**inputs, max_new_tokens=100)
        answer = self.answerer.blip2_processor.batch_decode(outputs, skip_special_tokens=True)
        return answer
