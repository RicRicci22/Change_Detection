from dataclasses import dataclass, field
from typing import List

QUESTION_INSTRUCTION = (
    "A chat between a curious user and an artificial intelligence assistant. " 
    "The assistant asks meaningful, detailed, accurate questions to the user based on the context. "
    "The questions serves as a way to explore further the satellite image contents. "
    "The context consists of a brief description provided by the user and all the next questions and answers. "
    "The questions must be posed such as to be answered just by looking at the satellite image on a screen. "
    "Provide one question at a time."
)

ANSWER_INSTRUCTION = (
    "Answer the question. If you are not sure about the answer, say you don't know honestly."
)

SUMMARY_INSTRUCTION = (
    "Summarize the information in the chat between the assistant and the user, creating a descriptive paragraph of the contents of the image. "
    "Don't add information. Don't miss information."
)

CHANGE_INSTRUCTION = (
    "I will provide descriptions of two remote sensing images taken at the same location at different times. "
    "You are a useful ASSISTANT, and you generate a textual paragraph summarizing the changes occurred, if any, based solely on the two descriptions. "
    "Don't add information. Don't miss information."
)

@dataclass
class Conversation:
    """A class that keeps all conversation history."""
    q_system: str = QUESTION_INSTRUCTION
    a_system: str = ANSWER_INSTRUCTION
    s_system: str = SUMMARY_INSTRUCTION
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
        role = "USER" # The first question is always from the USER!
        for message in messages:
            if len(message) != 2:
                raise ValueError("The messages must be a list of lists, where each list has two elements: role and message")
            if message[0] not in self.roles:
                raise ValueError(f"The role {message[0]} is not recognized. Please use one of the following roles: {self.roles}")
            if role == message[0]:
                # Wrong 
                raise ValueError("The messages should alternate between the two roles, and the first must come from the ASSISTANT!") # The first message is the user describing the image
            
            role = message[0]
        # Set the messages
        self.messages = messages
        
    def get_question_prompt(self, model:str="vicuna", context:int=100):
        '''
        Function that return the prompt for generating a question for different models
        Each model should have its own generate prompt function
        '''
        if model == "vicuna":
            return self.generate_prompt_vicuna(system = self.q_system, context = context)
        else:
            raise ValueError(f"The model {model} is not recognized. Please use one of the following models: vicuna")
    
    def get_answer_prompt(self, model:str="blip2", context:int=100):
        '''
        Function that return the prompt for generating a question for different models
        Each model should have its own generate prompt function
        '''
        if model == "blip2":
            return self.generate_prompt_blip2(system = self.a_system, context = context)
        else:
            raise ValueError(f"The model {model} is not recognized. Please use one of the following models: blip2")
        
    def get_summary_prompt(self, model:str="vicuna", context:int=100):
        '''
        Function that return the prompt for generating a question for different models
        Each model should have its own generate prompt function
        '''
        if model == "vicuna":
            return self.generate_prompt_vicuna(system = self.q_system, context = context, last_message=self.s_system)
        else:
            raise ValueError(f"The model {model} is not recognized. Please use one of the following models: vicuna")
    
    def get_change_description_prompt(self, model:str="vicuna", description1:str="", description2:str=""):
        '''
        Function that return the prompt for generating a question for different models
        Each model should have its own generate prompt function
        '''
        if model == "vicuna":
            return self.generate_cd_prompt_vicuna(system = self.cd_system, description1 = description1, description2 = description2)
        else:
            raise ValueError(f"The model {model} is not recognized. Please use one of the following models: vicuna")
    
    def generate_prompt_vicuna(self, system:str, last_message:str="", context:int=100):
        '''
        Prompt format 
        "<system>. USER: Description. ASSISTANT: First question.</s>USER: First answer. ASSISTANT:"
        '''
        # Limit the messages by context
        # Remove the first message (describe this image in detail)
        self.messages = self.messages[1:]
        # Go on
        if len(self.messages) > context:
            messages = self.messages[-context:]
        else:
            messages = self.messages
        
        if system[-1] != ".":
            system += "."
        
        prompt = system + self.sep
        
        for message in messages:
            role, message = message
            # Check the message
            if message != "":
                if role == "USER":
                    if message[-1] != ".":
                        message += "."
                else:
                    if message[-1] != "?":
                        message += "?"
        
            prompt += role + ": " + message + self.sep2
                
        
        # Append the last role and optional last message
        # Check last message 
        if last_message != "":
            if last_message[-1] != ".":
                last_message += "."
            prompt = prompt[:-1] + "." + self.sep + last_message + self.sep + self.roles[1] + ":"
        else:
            prompt += self.roles[1] + ":"
        return prompt
    
    def generate_cd_prompt_vicuna(self, system:str, description1:str="", description2:str=""):
        '''
        Prompt format
        "<system> Description 1: <description1> Description 2: <description2> Changes:"
        '''
        # Check the descriptions
        if description1[-1] != ".":
            description1 += "."
        if description2[-1] != ".":
            description2 += "."
            
        prompt = system + self.sep
        prompt += "Description 1: " + description1 + self.sep
        prompt += "Description 2: " + description2 + self.sep
        prompt += self.roles[1] + ":"
        
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
    # Simulate a conversation
    question1 = "give a detailed description of this satellite image"
    answer1 = "An image of a blue sky"
    question2 = "What is the color of the sky?"
    answer2 = "Blue"
    question3 = "What is the color of the sea?"
    answer3 = "Blue"
    messages = [["ASSISTANT", question1], ["USER", answer1], ["ASSISTANT", question2], ["USER", answer2], ["ASSISTANT", question3]]
    conv_v1.load_messages(messages)
    # Get the prompt for the next question
    prompt = conv_v1.get_answer_prompt(context=100)
    print(prompt)
    