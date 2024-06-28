from dataclasses import dataclass, field
from typing import List
import os
import pickle

SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. " 
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
)

FIRST_USER_MESSAGE = (
    "I am interested in knowing which changes happened between two satellite images of the same area acquired at different times. "
    "I can look at the two images and answer your questions one by one. "
    "You must produce questions focused on structural changes in the area layout. It must be possible to answer the questions just by looking at the rgb images. "
)

NEXT_USER_MESSAGE = (
    ""
)


ANSWER_INSTRUCTION = (
    "Answer the question. If you are unsure on the answer, say you don't know."
)

SUMMARY_INSTRUCTION = (
    "This dialogue is about possible changes between two satellite images. Summarize it creating a descriptive paragraph of the changes happened. The paragraph should not mention that the information is taken from the dialogue, it must be just a description of the changes as it would be by looking at the images."
)

CHANGE_INSTRUCTION = (
    "A chat between a curious user and an artificial intelligence assistant. " 
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "USER: I have two descriptions of two satellite images. Based on the two descriptions, create a paragraph summarizing the main changes that you deduce. Do not describe what is in each image but just changes. If there are no significant changes, just say that."
)

@dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str = SYSTEM
    a_system: str = ANSWER_INSTRUCTION
    summary_instruction: str = SUMMARY_INSTRUCTION
    cd_system: str = CHANGE_INSTRUCTION
    roles: tuple[str] = ("USER", "ASSISTANT")
    messages: List[List[str]] = field(default_factory=list)
    sep: str = " " # This is the separator in the same interaction to separate the roles from the messages
    sep2: str = "</s>" # This is the separator between two interactions (USER-ASSISTANT)

    def reset_messages(self):
        '''
        Function that resets the messages, so the conversation can be reused
        '''
        self.messages = []
    
    def load_messages(self, messages:list):
        '''
        Function that loads a list of messages into the conversation
        '''
        # Check if the messages are in the correct format 
        role = "ASSISTANT"
        for message in messages:
            if len(message) != 2:
                raise ValueError("The messages must be a list of tuples, where each tuple has two elements: role and message")
            if message[0] not in self.roles:
                raise ValueError(f"The role {message[0]} is not recognized. Please use one of the following roles: {self.roles}")
            if role == message[0]:
                # Wrong 
                raise ValueError("The messages should alternate between the two roles, and the first must come from the USER!")
            
            role = message[0]
        # Checks passed, set the messages
        self.messages = messages
        
    def get_answer_prompt(self, model:str="blip2", context:int=100):
        '''
        Function that return the prompt for generating a question for different models
        Each model should have its own generate prompt function
        '''
        if model == "blip2":
            return self.generate_prompt_blip2(system = self.a_system, context = context)
        elif model=="otter":
            return self.generate_prompt_otter(context = context)
        else:
            raise ValueError(f"The model {model} is not recognized. Please use one of the following models: blip2")
        
    def get_summary_prompt(self, model:str="vicuna", context:int=100):
        '''
        Function that return the prompt for generating a question for different models
        Each model should have its own generate prompt function
        '''
        if model == "vicuna":
            return self.generate_summaryprompt_vicuna(system = self.s_system)
        else:
            raise ValueError(f"The model {model} is not recognized. Please use one of the following models: vicuna")
    
    def get_cd_prompt(self, description1:str="", description2:str="", model:str="vicuna"):
        '''
        Function that return the prompt for generating a question for different models
        Each model should have its own generate prompt function
        '''
        #assert description1 != "" and description2 != "", "The descriptions cannot be empty!"
        
        if model == "vicuna":
            return self.generate_cd_prompt_vicuna(system = self.cd_system, description1 = description1, description2 = description2)
        else:
            raise ValueError(f"The model {model} is not recognized. Please use one of the following models: vicuna")
        
    def generate_first_conversations(self, image_path:str, user_intention:str="", chat_cache = "chats_cache"):
        '''
        Function that generates a first conversation for each image
        '''
        for image_name in os.listdir(image_path+"/im1"):
            if user_intention!="":
                user_message = FIRST_USER_MESSAGE + " " + user_intention
            else:
                user_message = FIRST_USER_MESSAGE
                
            messages=[["USER", user_message]]
                
            # Save it in a pkl file 
            with open(os.path.join(chat_cache, image_name.split(".")[0]+".pkl"), "wb") as file:
                pickle.dump(messages, file)
    
    def generate_prompt_vicuna(self, last_message:str="", init_assistant:str="", context:int=100):
        '''
        General prompt format 
        "<system>. USER: First message. ASSISTANT: First response.</s>USER: Second message. <optional last_message>. ASSISTANT: <optional init_assistant> second response "
        Questioning prompt format 
        "<system>. USER: First message. ASSISTANT: First question: generated first question.</s>USER: First answer. Next question ASSISTANT: Next question: generated next question"
        Input: 
        last_message: A message to append to the end of the conversation, right before "ASSISTANT:"
        init_assistant: the beginning of the answer for the assistant, to better direct in the right answer. 
        '''
        prompt = self.system
        messages = self.messages
        
        if len(messages) == 1:
            init_assistant="First question:"
        else:
            # Limit the messages by context
            if len(messages) > context:
                messages = messages[-context:]
                
            last_message=NEXT_USER_MESSAGE
            init_assistant="Next question:"
            
        for message in messages:
            role, message = message
            message = message.strip()
            # Check the message
            if message != "":
                if role == "USER":
                    if message[-1]!=".":
                        message += "."
                    prompt += role + ": " + message + self.sep
                else:
                    prompt += role + ": " + message + self.sep2
        
        # Append the last role and optional last message
        # Check last message 
        if last_message != "":
            last_message = last_message.strip()
            if last_message[-1] != ".":
                last_message = last_message[:-1]+"."
            prompt += last_message + self.sep
        
        prompt += self.roles[1] + ": " + init_assistant
        return prompt
    
    def generate_summary_prompt_vicuna(self):
        '''
        Prompt format 
        "USER: Description. ASSISTANT: first question.</s>USER: first answer. ASSISTANT:second question.</s>USER: second answer. ... ASSISTANT:last question.</s>USER: last answer. <last_message> ASSISTANT:"
        '''
        # Remove the first message (instructions from the user)
        prompt=""
        prompt += self.system
        if self.messages[0][0]=="USER":
            messages = self.messages[1:]
        else:
            messages = self.messages
        assert messages[0][0]=="ASSISTANT", "The first message must be from the assistant! (question)"
        if len(messages) % 2 != 0:
            # Discard the last message
            messages = messages[:-1]
            
        interactions = list()
        for i in range(0,len(messages),2):
            assistant = messages[i]
            user = messages[i+1]
            if assistant[1]=="" or user[1]=="":
                continue
            interactions.append((assistant,user))
        
        # Change the order of user and assistant to help the model
        for interaction in interactions:
            prompt += "USER: " + interaction[0][1] + self.sep + "ASSISTANT: " + interaction[1][1] + self.sep2
        
        prompt += self.roles[0] + ": " + self.summary_instruction + self.sep + "ASSISTANT:"
            
        return prompt
    
    def generate_cd_prompt_vicuna(self, system:str, description1:str="", description2:str=""):
        '''
        Prompt format
        "<system> Description 1: <description1> Description 2: <description2> Changes:"
        ''' 
        prompt = system + self.sep
        prompt += "Description 1: " + description1 + self.sep
        prompt += "Description 2: " + description2 + self.sep
        prompt += self.roles[1] + ": Changes:"
        
        return prompt
    
    def generate_prompt_blip2(self, system:str, context:int=100):
        '''
        Prompt format 
        "Question: What is a dinosaur holding? <system> Answer:"
        '''
        # Limit the messages by context 
        if len(self.messages) > context:
            messages = self.messages[-context:]
        else:
            messages = self.messages
        
        if system[-1] != ".":
            system += "."
        
        prompt = self.a_system + self.sep

        # Check that the last message is from the USER (a question)
        if messages[-1][0] != self.roles[1]:
            raise ValueError("The last message must be from the ASSISTANT!")
        
        for message in messages:
            role, message = message
            if message != "":
                if role == "ASSISTANT":
                    prompt += "Question: " + message + self.sep
                else:
                    # Manipulate the message
                    if message[-1] != ".":
                        message += "."
                    prompt += "Answer: " + message + self.sep
            else:
                raise ValueError("History cannot contain empty messages!")
        
        # Append the last role and optional last message
        prompt += "Answer:"
        return prompt
    
    def generate_prompt_otter(self, context:int=100):
        '''
        Prompt format 
        <image><image> User: <question> GPT:<answer>
        '''
        assert context == 1, "The context must be 1 for otter!"
        assert self.messages[-1][0]=="ASSISTANT", "The last message must be from the assistant!"
        
        prompt = "<image>User: "+self.messages[-1][1]+" GPT:<answer>"
        return prompt        
        
     
    def return_messages(self):
        """
        Returns two lists, one for the questions and the other for the answers
        """
        questions = []
        answers = []
        for message in self.messages:
            if message[0] == self.roles[1]:
                questions.append(message[1])
            elif message[0] == self.roles[0]:
                answers.append(message[1])
            else:
                raise ValueError("Role not recognized!")

        return questions, answers


if __name__ == "__main__":
    conv_v1 = Conversation()
    #chat = Chatter()
    # Simulate a conversation
    question1 = "Can you give a detailed description of this satellite image?"
    answer1 = "An image of a blue sky"
    question2 = "What is the color of the sky?"
    answer2 = "Blue"
    question3 = "What is the color of the sea?"
    answer3 = "Black"
    messages = [["USER", FIRST_USER_MESSAGE], ["ASSISTANT", question1], ["USER",answer1], ["ASSISTANT", question2], ["USER", answer2], ["ASSISTANT", question3], ["USER", answer3]]
    conv_v1.load_messages(messages)
    # Get the prompt for the next question
    prompt = conv_v1.generate_summary_prompt_vicuna()
    print(prompt)
    # chat.load_llm("vicuna", "TheBloke/vicuna-13B-v1.5-GPTQ", "cuda:1")
    # gen_cfg = GenerationConfig.from_pretrained("TheBloke/vicuna-13B-v1.5-GPTQ")
    # gen_cfg.max_new_tokens=200
    # gen_cfg.do_sample=False
    # gen_cfg.temperature=0.7
    # gen_cfg.top_p=0.95
    # gen_cfg.top_k=40
    # gen_cfg.repetition_penalty=1.1
    # out = chat.call_vicuna(prompt, gen_cfg, task="summarization")
    # print(prompt+"\n\n")
    # print(out)