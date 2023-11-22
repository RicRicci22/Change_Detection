import json

with open("results_chat/chats_postprocessed.json", "r") as file:
    chats = json.load(file)

#print(chats["04451_post.pkl"])
for piece in chats["02527_pre.pkl"]:
    print(piece[1])
    
with open("results_chat/summaries.json", "r") as file:
    summaries = json.load(file)
    
print(summaries["02527_post.pkl"])


with open("results_chat/cds.json", "r") as file:
    cds = json.load(file)
    
print(cds["02527"])