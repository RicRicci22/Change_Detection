'''
This code takes a dicrionary of chats and summarize them, returning a dictionary of summaries. The chats are already post-processed and ready 
to be summarized.
Inputs: 
- chats: dictionary of chats
Outpus: 
- summaries: dictionary of summaries
'''
from utils.chat import Chatter

def summarize_chats(chats:dict, model_type="vicuna", model_version="vicuna1.5", device="cuda:0"):
    chat = Chatter()
    chat.load_summarizer(model_type, model_version, device)
    