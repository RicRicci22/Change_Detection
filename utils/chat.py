import torch
import gc
from utils.utils import load_mistral, load_openassistant, load_hug_model, load_vicuna, load_blip2, load_otter
from numba import cuda

LLM_MODELS = ["chatgpt", "vicuna", "mistral", "openassistant"]
LMM_MODELS = ["blip2", "otter"]

def load_language_model(model_type: str, model_name:str, device: str="cpu"):
    '''
    Load the questioner model, by first checking if it is a valid model
    '''
    if model_type in LLM_MODELS:
        if model_type == "chatgpt":
            raise NotImplementedError("chatgpt is not implemented yet")
        elif model_type == "vicuna":
            model, tokenizer = load_vicuna(model_name, device)
        elif model_type == "mistral":
            model, tokenizer = load_mistral(model_name, device)
        elif model_type == "openassistant":
            model, tokenizer = load_openassistant(model_name, device)
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
        if model_type == "otter":
            model = load_otter(model_name, device)
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

    def load_llm(self, llm_type, llm_model, device):
        '''
        Load large language model (can handle only text)
        This is used for the question generation part, the summarization and the extraction of the changes
        '''
        print("Loading large language model {}, version {}".format(llm_type, llm_model))
        self.language_model, self.llm_tokenizer = load_language_model(llm_type, llm_model, device)
        self.device_llm = device
    
    def move_llm(self, device):
        '''
        Delete the language model to save memory
        '''
        self.language_model.to(device)
        # del self.language_model
        # del self.llm_tokenizer
        # gc.collect()
        torch.cuda.empty_cache()
        
    def load_lmm(self, lmm_type, lmm_model, device):
        '''
        Load large multimodal model (can handle both text and images)
        This is used either for answer to the questions looking at the image, or for direct extraction (eg. otter model)
        '''
        print("Loading large multimodal model {}, version {}".format(lmm_type, lmm_model))
        self.multimodal_model = load_multimodal_model(lmm_type, lmm_model, device)
        self.device_lmm = device
        self.dtype = next(self.multimodal_model.parameters()).dtype
        
    def move_lmm(self, device):
        '''
        Delete the multimodal model to save memory
        '''
        self.multimodal_model.to(device)
        # del self.language_model
        # del self.llm_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
    def call_vicuna(self, prompts:list, generation_config:dict=None, return_probs=False):
        '''
        This function call the questioner model passing the parameters and the prompt. 
        The questioner model should implement the generate function (huggingface syntax).
        It can work in batch or not, depending on the length of prompts
        
        Inputs:
            - prompts: list of prompts to be passed to the questioner model
            - generation_config: generation configuration for model.generate() function
            - task: 
        '''
        ############################# GENERATE #############################
        with torch.no_grad():
            input_ids = self.llm_tokenizer(prompts, padding=True, return_tensors='pt').input_ids.to(self.device_llm)
            assert generation_config is not None, "Generation config must be specified"
            if return_probs:
                outputs = self.language_model.generate(inputs=input_ids, generation_config=generation_config, return_dict_in_generate=True, output_scores=True)
                first_token_normalized = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
                probs = []
                for batch_item in first_token_normalized:
                    probs.append((batch_item[3869].item(),batch_item[1939].item())) # Probabilities of yes and no in the first token
                
                # Decode the outputs 
                outputs = self.llm_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)  
                return outputs, probs
            else:
                outputs = self.language_model.generate(inputs=input_ids, generation_config=generation_config)
                outputs = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                return outputs
    
    def call_mistral(self, prompts:list):
        with torch.no_grad():
            input_ids = self.llm_tokenizer(prompts, padding=True, return_tensors='pt').input_ids.to(self.device_llm)
            outputs = self.language_model.generate(inputs=input_ids, do_sample=True, max_new_tokens=200)
            outputs = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return outputs
    
    def call_otter(self, vision_x:torch.Tensor, prompts:list, generation_config:dict=None, task:str="answering"):
        assert task=="answering", "Otter can only answer questions"
        lang_x = self.multimodal_model.text_tokenizer(prompts, return_tensors="pt")
        with torch.no_grad():
            answer = self.multimodal_model.generate(vision_x=vision_x.to(self.multimodal_model.device, dtype=torch.float16),

            lang_x=lang_x["input_ids"].to(self.multimodal_model.device, dtype=torch.int64),

            attention_mask=lang_x["attention_mask"].to(self.multimodal_model.device),

            max_new_tokens=512,

            no_repeat_ngram_size=3,
            
            num_beams=3,
            
            pad_token_id=self.multimodal_model.text_tokenizer.pad_token_id,
            
            generation_config = generation_config
            )
            
            # Decode the answer
            answer = self.multimodal_model.text_tokenizer.decode(answer[0], skip_special_tokens=True).split("GPT:")[-1].lstrip().rstrip().split("<|endofchunk|>")[0]
            
        return answer
        
    def trim_output(self, outputs, task="questioning"):
        '''
        function to trim the output of the llm depending on the task
        '''
        ############################# POST PROCESS #############################
        if task == "questioning":
            outputs = [output.split(":")[-1].split("?")[0].strip()+"?" for output in outputs]
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
