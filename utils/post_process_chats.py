'''
This file contains functions to post-process the chats generated by chat_vicuna_blip2.py
The main post-processing involves removing questions that have a "no" answer and removing questions that ends with a . instead of a ?, which likely suggests problems in the generation of the question.
'''
import os 
import pickle
import json

def read_chats(path:str):
    '''
    This function takes a path to a folder containing a different pkl file for each chat.
    Each file contains a list of questions and answers.
    Each conversation is put in a dictionary with the key being the name of the file.
    '''
    files = os.listdir(path)
    chats = {}
    for file in files:
        with open(os.path.join(path, file), "rb") as f:
            chats[file] = pickle.load(f)[:-1]
    return chats

def remove_answers(dict_in:dict, answers_blacklist:list):
    '''
    This function removes the rounds that have an answer that is in the answers_blacklist list.
    Input:
        dict_in: dict -> dictionary containing the chats
        answers_blacklist: list -> list of answers to remove
    Output: 
        dict_out: dict -> cleaned dictionary
    '''
    dict_out = {}
    discarded = 0 
    all = 0 
    for key in dict_in.keys():
        for i in range(int(len(dict_in[key])/2)):
            question = dict_in[key][i*2]
            answer = dict_in[key][i*2+1]
            all+=1
            if(answer[1] in answers_blacklist):
                discarded+=1
            else:
                if key not in dict_out.keys():
                    dict_out[key] = []
                dict_out[key].append(question)
                dict_out[key].append(answer)
    
    print("All questions: ", all)
    print("Discarded questions: ", discarded)
    
    return dict_out

def remove_affirmative_questions(dict_in:dict):
    '''
    This function removes the questions ending with a point.
    '''
    dict_out = {}
    discarded = 0 
    all = 0 
    for key in dict_in.keys():
        for i in range(int(len(dict_in[key])/2)):
            question = dict_in[key][i*2]
            answer = dict_in[key][i*2+1]
            all+=1
            if(question[1][-2] != "."):
                if key not in dict_out.keys():
                    dict_out[key] = []
                dict_out[key].append(question)
                dict_out[key].append(answer)
            else:
                discarded+=1
    
    print("All questions: ", all)
    print("Discarded questions: ", discarded)
    
    return dict_out

def save_dict(dict_in:dict, path:str):
    '''
    This function saves the dictionary in a path using json formatting.
    '''
    with open(path, "w") as f:
        json.dump(dict_in, f, indent=4)
        
def chats_postprocessing(path_in:str, path_out:str)-> None:
    '''
    This function handles the complete post-processing of the chats
    '''
    print("################### Post-processing chats ###################")
    chats = read_chats(path_in)
    # Remove round with answer "no"
    chats = remove_answers(chats, answers_blacklist=["no", "don't know", "not sure"])
    # Remove questions ending with a point
    chats = remove_affirmative_questions(chats)
    # Save the dict
    save_dict(chats, path_out)
    print("################### Finished chats post-processing ###################")
    