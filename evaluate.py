# # RUN THIS CODE TO CREATE THE DICTIONARY OF CHANGES BASED ON THE GROUND TRUTHS
# import os

from skimage import io
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
from tqdm import tqdm
from utils.dataset import EvaluationDataset
from torch.utils.data import DataLoader
from torch import no_grad

# ST_COLORMAP = [[0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]]

# ST_CLASSES = ['water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']

# changes = dict()

# for img_name in tqdm(os.listdir("/media/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/label1")):
#     label_1 = io.imread("/media/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/label1/" + img_name, pilmode="RGB")
#     label_2 = io.imread("/media/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/label2/" + img_name, pilmode="RGB")
#     label_1 = label_1.reshape((512,512,3))
#     label_2 = label_2.reshape((512,512,3))
    
#     converted_1 = np.zeros((label_1.shape[0], label_1.shape[1]))
#     converted_2 = np.zeros((label_2.shape[0], label_2.shape[1]))
    
#     # Convert in indices 
#     for color in ST_COLORMAP:
#         color_array = np.array(color).reshape((1,1,3))
#         test = np.subtract(label_1,color_array)
#         locations_1 = np.sum(np.abs(np.subtract(label_1,color_array)),axis=2)
#         #print(locations_1[460,460])
#         locations_2 = np.sum(np.abs(np.subtract(label_2,color_array)),axis=2)
#         converted_1[locations_1==0] = ST_COLORMAP.index(color)+1
#         converted_2[locations_2==0] = ST_COLORMAP.index(color)+1
        
#     # Compute the overall changes 
#     changes_string = []
#     for index in range(0, len(ST_CLASSES)):
#         indices_1 = np.where(converted_1==index+1)
#         #print(indices_1)
#         values_2 = converted_2[indices_1]
#         single_results = np.unique(values_2).astype(np.int8)
#         for value in list(single_results):
#             if value-1 != index:
#                 changes_string.append("a " + ST_CLASSES[index]+" area has transformed into a "+ST_CLASSES[value-1]+" area.")
                
#     changes[img_name] = changes_string
    
#     # Save the dictionary 
#     with open('changes.json', 'w') as fp:
#         json.dump(changes, fp, indent=4)

# RUN THIS CODE TO EVALUATE THE RESULTS

# Steps:
# 1. Create the dictionary of changes based on the ground truths with the code above
# 2. Load a large language model
# 3. Load the dictionary of changes
# 4. Load the predictions (generated change descriptions)
# 5. Evaluate the predictions using the prompt

def main(path_results:str=None, path_cds:str=None):
    if path_results is None:
        # 2.
        model = "TheBloke/vicuna-13B-v1.5-GPTQ"
        device = "cuda:1"
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, padding_side="left")
        gptq_config = GPTQConfig(bits=4, exllama_config={"version":2})
        model = AutoModelForCausalLM.from_pretrained(model,
                                                    device_map=device,
                                                    quantization_config=gptq_config)
        # 5.
        yes = 0 
        no = 0 
        unknown = 0

        # 3. 4.
        dataloader = DataLoader(EvaluationDataset(path_cds), batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

        results = dict()

        model.eval()
        model.generation_config.max_new_tokens = 512
        model.generation_config.temperature = 0.2
        model.do_sample = True
        model.top_p = 0.9

        with no_grad():
            for batch in tqdm(dataloader):
                images, prompts, changes = batch

                # Tokenize the prompt
                input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(device)
                output = model.generate(inputs=input_ids)
                out = tokenizer.batch_decode(output, skip_special_tokens=True)
                
                for i in range(len(out)):
                    response = out[i]
                    response = response.split("ASSISTANT:")[1].strip()
                    image = images[i]
                    change = changes[i]
                    try:
                        results[images[i]].append((change, response))
                    except:
                        results[images[i]] = [(change, response)]
            # Save the results
            with open('results_llava/evaluation_results.json', 'w') as fp:
                json.dump(results, fp, indent=4)
    else:
        with open(path_results, "r") as f:
    
            results = json.load(f)
        
        # Summary of the results
        no = 0
        yes = 0 
        unknown = 0
        for image in results:
            for change, response in results[image]:
                if "yes," in response.lower():
                    yes += 1
                elif "no," in response.lower():
                    no += 1
                else:
                    unknown += 1
        print("Yes: ", yes)
        print("No: ", no)
        print("Unknown: ", unknown)

def evaluation_summary(path_results:str, path_changes:str):
    '''
    This function evaluates change descriptions based on the LLM response to the presence of every fact.
    Input:
    - path_results: path to the results file, a dictionary where for each image (change description) there is a list of tuples (change, response).
    '''
    with open(path_results, "r") as f:
        results = json.load(f)
        
    for changes_evaluated in results.values():
        assert len(changes_evaluated) == 30, "The number of changes evaluated is not 30."
    
    # Create the dictionary containing the results for each change
    classes = ['water', 'ground', 'low_vegetation', 'tree', 'building', 'sports_field']
    changes = dict()
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j:
                changes[(classes[i],classes[j])] = {"tp":0, "fp":0, "fn":0, "tn":0}
    with open(path_changes, "r") as f:
        changes_gt = json.load(f)
    
    for image_name, changes_evaluated in results.items():
        gt_changes = changes_gt[image_name+'.png']
        for change, response in changes_evaluated:
            class_1 = change.split(" ")[1]
            if class_1 == "low":
                class_1 = "low_vegetation"
                class_2 = change.split(" ")[8]
            elif class_1 == "sports":
                class_1 = "sports_field"
                class_2 = change.split(" ")[8]
            else:
                class_2 = change.split(" ")[7]
                
            if class_2 == "low":
                class_2 = "low_vegetation"
                
            if class_2 == "sports":
                class_2 = "sports_field"
                
            if change in gt_changes and "yes," in response.lower():
                # A true positive
                changes[(class_1,class_2)]["tp"] += 1
            if change in gt_changes and "no," in response.lower():
                # A false negative
                changes[(class_1,class_2)]["fn"] += 1
            if change not in gt_changes and "yes," in response.lower():
                # A false positive
                changes[(class_1,class_2)]["fp"] += 1
            if change not in gt_changes and "no," in response.lower():
                # A true negativem
                changes[(class_1,class_2)]["tn"] += 1
                
    for change in changes:
        tp = changes[change]["tp"]
        fp = changes[change]["fp"]
        fn = changes[change]["fn"]
        tn = changes[change]["tn"]
        
        if tp == 0:
            precision = 0
            recall = 0
        else:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
        print("Change: ", change)
        print("Precision: ", precision)
        print("Recall: ", recall)
    
if __name__=="__main__":
    path_results = None
    path_cds = "results_llava/cds.json"
    main(path_results, path_cds)
    #evaluation_summary("results_chat/evaluation_results.json", "GT_changes.json")