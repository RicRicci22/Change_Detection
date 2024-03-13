# # RUN THIS CODE TO CREATE THE DICTIONARY OF CHANGES BASED ON THE GROUND TRUTHS
# import os
import os
import pickle
import json
from skimage import io
import numpy as np
from tqdm import tqdm
from utils.dataset import EvaluationDataset
from torch.utils.data import DataLoader
from torch import no_grad
import requests
from transformers import GenerationConfig
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import keys

from utils.chat import Chatter

ST_COLORMAP = [[0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]]

ST_CLASSES = ['water', 'ground', 'low_vegetation', 'tree', 'building', 'sports_field']

def create_gt_changes()->None:
    '''
    This function examines the semantic change maps and creates a dictionary containing the ground truth changes for each image.
    '''
    changes = dict()
    
    areas = np.zeros((len(ST_CLASSES), len(ST_CLASSES)), dtype=np.int32)
    images_with_change = np.zeros((len(ST_CLASSES), len(ST_CLASSES)), dtype=np.int32)
    
    for img_name in tqdm(os.listdir("/media/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/label1")):
        label_1 = io.imread("/media/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/label1/" + img_name, pilmode="RGB")
        label_2 = io.imread("/media/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/label2/" + img_name, pilmode="RGB")
        label_1 = label_1.reshape((512,512,3))
        label_2 = label_2.reshape((512,512,3))
        
        converted_1 = np.zeros((label_1.shape[0], label_1.shape[1]))
        converted_2 = np.zeros((label_2.shape[0], label_2.shape[1]))
        
        # Convert in indices 
        for color in ST_COLORMAP:
            color_array = np.array(color).reshape((1,1,3))
            locations_1 = np.sum(np.abs(np.subtract(label_1,color_array)),axis=2)
            locations_2 = np.sum(np.abs(np.subtract(label_2,color_array)),axis=2)
            converted_1[locations_1==0] = ST_COLORMAP.index(color)+1
            converted_2[locations_2==0] = ST_COLORMAP.index(color)+1
            
        # Compute the overall changes 
        changes_string = []
        for index in range(0, len(ST_CLASSES)):
            indices_1 = np.where(converted_1==index+1)
            values_2 = converted_2[indices_1]
            single_results = np.unique(values_2).astype(np.int8)
            for value in list(single_results):
                if value-1 != index:
                    changes_string.append("a " + ST_CLASSES[index]+" area has transformed into a "+ST_CLASSES[value-1]+" area.")
                    areas[index, value-1] += np.sum(values_2==value)
                    images_with_change[index, value-1] += 1
                    
        changes[img_name] = changes_string
        
    print(areas)
    print(images_with_change)
    
    # Save the dictionary 
    with open('changes.json', 'w') as fp:
        json.dump(changes, fp, indent=4)

def evaluate_with_llm(path_cds:str=None, device="cuda:0", bunch=False)->None:
    '''
    This function takes some paragraphs, and evaluates if some facts are present or not in each paragraph. 
    Input:
    - path_cds: path to the file containing the change descriptions. It is a dictionary with keys the image names and values the change descriptions.
    - device: device to use for the evaluation.
    - bunch: if True, the path_cds is a directory containing a set of the files to evaluate. Otherwise, it is a single file.
    '''
    # 1. Check if path_cds is a directory
    if os.path.isdir(path_cds):
        files = [os.path.join(path_cds,i) for i in os.listdir(path_cds)]
    else:
        files = [path_cds]
        
    # 2. Load the model 
    chat = Chatter()
    model_name = "TheBloke/vicuna-13B-v1.5-GPTQ"
    chat.load_llm("vicuna", model_name, device)

    # 3. Load model configuration -> sampling is removed for deterministic outputs
    gen_cfg = GenerationConfig.from_pretrained(model_name)
    gen_cfg.max_new_tokens=512
    gen_cfg.do_sample=False
    gen_cfg.repetition_penalty=1.1

    # 4. Run the evaluation
    for file in tqdm(files):
        print("Elaborating " + file)
        results = dict()
        if bunch:
            if file.replace("examples","results").split("/")[1] in os.listdir(path_cds) or "examples" not in file:
                continue
        
        dataloader = DataLoader(EvaluationDataset(file), batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
        with no_grad():
            for j, batch in enumerate(tqdm(dataloader)):
                images, prompts, changes = batch
                outputs, probs = chat.call_vicuna(prompts=prompts, generation_config=gen_cfg, return_probs=True)    
                for i in range(len(outputs)):
                    response = outputs[i]
                    prob = probs[i]
                    prob = (round(prob[0],2), round(prob[1],2))
                    response = response.split("ASSISTANT:")[1].strip()
                    change = changes[i]
                    try:
                        results[images[i]].append((change, response, prob))
                    except:
                        results[images[i]] = [(change, response, prob)]
            
        del dataloader
        
        # 5. Save the results
        if bunch:
            with open(file.replace("examples","results"), "w") as f:
                json.dump(results, f, indent=4)
        else:
            file_chunks = file.split("/")
                
            with open(file_chunks[0]+"/evaluation_"+file_chunks[1], "w") as f:
                json.dump(results, f, indent=4)

def llm_evaluation_summary(path_results:str, path_changes:str):
    '''
    This function evaluates a paragraph coherence based on the LLM response to the presence or not of fact(s).
    In this function, the LLM is regarded as an oracle that can always provide the correct response (it has been validated using templates).
    This function has been crafted specifically for evaluating change descriptions on the Second dataset. 
    The facts are derived from the semantic change maps, there are 30 possible changes, and all must be evaluated. 
    Input:
    - path_results: path to the results file, a dictionary with key=image name, value: list of tuples (change, response).
    '''
    with open(path_results, "r") as f:
        results = json.load(f)
    
    # Assert if all the 30 possible changes have been evaluated
    for changes_evaluated in results.values():
        assert len(changes_evaluated) == 30, "The number of changes evaluated is not 30."

    with open(path_changes, "r") as f:
        changes_gt = json.load(f)
    
    ground_truth = -np.ones((len(results), len(ST_CLASSES), len(ST_CLASSES)))
    predictions = np.zeros((len(results), len(ST_CLASSES), len(ST_CLASSES)))
    
    image_names = []
    skipped = 0
    for i, (image_name, changes_evaluated) in enumerate(results.items()):
        image_names.append(image_name)
        if "png" in image_name:
            gt_changes = changes_gt[image_name]
        else:
            gt_changes = changes_gt[image_name+'.png']
            
        for change, response, _ in changes_evaluated:
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
            if change in gt_changes:
                ground_truth[i, ST_CLASSES.index(class_1), ST_CLASSES.index(class_2)] = 1
        
        
            if "yes," in response.lower() or "yes." in response.lower() or "yes</s>" in response.lower():
                #percentage = percentages[0]
                predictions[i, ST_CLASSES.index(class_1), ST_CLASSES.index(class_2)] = 1 #* percentage
            elif "no," in response.lower() or "no." in response.lower() or "no</s>" in response.lower():
                #percentage = percentages[1]
                predictions[i, ST_CLASSES.index(class_1), ST_CLASSES.index(class_2)] = -1 #* percentage
            else:
                skipped += 1
    
    print("Skipped answers: ", skipped)
    # Calculate general precision and recall (for all the classes)
    recalls = np.zeros((ground_truth.shape[0],))
    precisions = np.zeros((ground_truth.shape[0],))
    for i in range(ground_truth.shape[0]):
        detections = 0
        missed = 0 
        false = 0 
        for j in range(ground_truth.shape[1]):
            for z in range(ground_truth.shape[2]):
                if j != z:
                    if ground_truth[i,j,z] == 1 and predictions[i,j,z] == 1:
                        detections += 1     
                    if ground_truth[i,j,z] == 1 and predictions[i,j,z] != 1:
                        missed += 1
                    if ground_truth[i,j,z] == -1 and predictions[i,j,z] == 1:
                        false += 1
        
        if detections == 0:
            # It is ok if there is at least one change per image.
            recalls[i] = 0
            precisions[i] = 0
        else:
            recalls[i] = detections/(detections+missed)
            precisions[i] = detections/(detections+false)
        
    for c in range(0,6):
        recalls_class = []
        precisions_class = []
        for i in range(ground_truth.shape[0]):
            detections_class = 0
            missed_class = 0
            false_class = 0
            for j in range(ground_truth.shape[1]):
                for z in range(ground_truth.shape[2]):
                    if j != z:
                        if ground_truth[i,j,z] == 1 and predictions[i,j,z] == 1:
                            detections += 1     
                        if ground_truth[i,j,z] == 1 and predictions[i,j,z] != 1:
                            missed += 1
                        if ground_truth[i,j,z] == -1 and predictions[i,j,z] == 1:
                            false += 1
                        if j == c or z == c :
                            if ground_truth[i,j,z] == 1 and predictions[i,j,z] == 1:
                                detections_class += 1
                            if ground_truth[i,j,z] == 1 and predictions[i,j,z] != 1:
                                missed_class += 1
                            if ground_truth[i,j,z] == -1 and predictions[i,j,z] == 1:
                                false_class += 1
            
            if detections_class == 0 and np.sum(ground_truth[i,c,:]+1)+np.sum(ground_truth[i,:,c]+1) != 0:
                recalls_class.append(0)
                precisions_class.append(0)
            elif detections_class != 0 and np.sum(ground_truth[i,c,:]+1)+np.sum(ground_truth[i,:,c]+1) != 0:
                recalls_class.append(detections_class/(detections_class+missed_class))
                precisions_class.append(detections_class/(detections_class+false_class))
            else:
                continue
        
        recalls_class = np.array(recalls_class)
        precisions_class = np.array(precisions_class)
        f1_scores_class = 2*(precisions_class*recalls_class)/(precisions_class+recalls_class)
        f1_scores_class = np.nan_to_num(f1_scores_class)
        print("Average F1 class " + str(c)+": ", np.mean(f1_scores_class))
            
    f1_scores = 2*(precisions*recalls)/(precisions+recalls)
    # Substitute 0 for nan values in f1_scores
    f1_scores = np.nan_to_num(f1_scores)
    average = np.mean(f1_scores)
    # Order the results from higher to lower 
    # index_new = np.argsort(f1_scores)
    # index_new = index_new[::-1]
    # print(f1_scores[index_new[-150]])
    # print(image_names[index_new[-150]])
    # #print(f1_scores[image_names.index("01624.png")])
    # #print(image_names[index_new[-1]])
    # # Get a confusion matrix like plot for the index_new[0] entry of results
    # import matplotlib.pyplot as plt
    # plt.imshow(predictions[index_new[-150]])
    # print(predictions[index_new[0]])
    # plt.savefig("predictions.png")
    # plt.imshow(ground_truth[index_new[-150]])
    # plt.savefig("true.png")
    
    # import matplotlib.pyplot as plt
    # plt.imshow(predictions[image_names.index("01624.png")],vmin=-1, vmax=1)
    # print(predictions[image_names.index("01624.png")])
    # plt.savefig("predictions.png")
    # plt.imshow(ground_truth[image_names.index("01624.png")])
    # plt.savefig("true.png")
    return average

def bleu_evaluation_summary(path_gt:str, path_cds:str):
    '''
    This function evaluates change descriptions comparing them to ground truth change descriptions using standard metrics. 
    Input:
    - path_gt: path to the ground truth file, a dictionary with key=image name, value: ground truth change description.
    - path_cds: path to the results file, a dictionary with key=image name, value: predicted change description.
    '''
    with open(path_gt, "r") as f:
        gt = json.load(f)
        
    with open(path_cds, "r") as f:
        cds = json.load(f)
        
    # Format in the style of coco
    cocoGT = dict()
    
    list_images = list()
    list_annotations = list()
    for img, description in gt.items():
        list_images.append({"file_name":img, "id":img.split(".")[0]})
        list_annotations.append({'image_id':img.split(".")[0], "id":img.split(".")[0], 'caption': description})
    
    cocoGT["images"] = list_images
    cocoGT["annotations"] = list_annotations
    
    coco = COCO(cocoGT)
    # Creating the predictions
    predictions = list()
    
    for img, description in cds.items():
        predictions.append({"image_id":img.split(".")[0], "caption":description})
    
    coco_result = coco.loadRes(predictions)
    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)
    
    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
def create_validation_examples_GPT35(true_positives:float=0.2, false_positives:float=0.0, template=True):
    '''
    This function uses GPT 3.5, or template, to create validation examples for the LLM evaluation.
    Basically the validation is built on two things to prove:
    When creating the descriptions, we will have different scenarios.
    - no false positives + different percentages of true positives (20, 60, 100): precision must be high, recall must vary accordingly. 
    - keeping true positives at 40% and varying the false positives: precision must vary accordingly, recall must be 40%.
    '''
    # OpenAI API Key
    api_key = keys.OPENAI_API_KEY
    
    # Defining some examples for in context learning generation
    # ex1 00709
    ex_1 = "observing the second image, there has been noticeable redevelopment in the area. a significant change is the introduction of a new road or wide pathway slicing through the center of the area, which has led to the removal of a line of trees and alteration of the surrounding land. the earlier presence of greenery and smaller structures has given way to this new pathway, changing the dynamic of the area. adjacent to this new feature, there appears to be the development of parking or storage areas where buildings with dark roofs used to be, particularly noticeable in the lower left quadrant where a structure has been removed. this suggests a shift from a more green, potentially mixed-use area to one focused on transportation and possibly industrial or commercial use."    
    ex_2 = "looking at the second image compared to the first, a significant amount of development is apparent. the most striking difference is the construction of new large buildings in place of what was primarily empty or undeveloped land. the cleared and leveled ground indicates the area is being prepared for further construction or land development. a new road or access path appears, seamlessly integrating with preexisting roadways, potentially to provide infrastructure to the new developments. vegetation that was once present, particularly in the top central part of the scene, has been mostly removed, perhaps to make way for this ongoing development. the geometric patterns noticeable on the ground suggest additional areas are being marked or foundational work is taking place in preparation for more construction activities."
    ex_3 = "from the first image to the second, the most significant change is the clearing of the dense vegetation in the top-central area. this section now shows signs of soil disturbance, likely from ground preparation or early stages of construction. there are no more small structures or equipment visible in this area that were initially among the vegetation. additionally, there seems to be a change in the appearance of the roof of the large building in the lower-central part of the view; it looks darker in the second image, which could be due to a change in the roofing material or varying lighting conditions casting a shadow."
    
    examples = "Example 1:\n"
    examples += ex_1 + "\n"
    examples += "Example 2:\n"
    examples += ex_2 + "\n"
    examples += "Example 3:\n"
    examples += ex_3 + "\n"
    
    with open("GT_changes.json", "r") as f:
        gt_changes = json.load(f)
    
    # Open the set of random images
    with open("validate_llm_eval/random_images.pkl", "rb") as f:
        random_images = pickle.load(f)
        
    all_changes = list()
    
    examples = dict()
    
    classes = ['water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j:
                all_changes.append("a " + classes[i] + " area has transformed into a " + classes[j] + " area.")

    for image_name in tqdm(random_images):
        true_changes = gt_changes[image_name]
        left_out = list(set(all_changes) - set(true_changes))
        
        if true_positives != 0:
            num_true_positives = round(len(true_changes)*true_positives)
            if num_true_positives == 0:
                num_true_positives = 1
        else:
            num_true_positives = 0
            
        if false_positives != 0:
            num_false_positives = round(len(left_out)*false_positives)
            if num_false_positives == 0:
                num_false_positives = 1
        else:
            num_false_positives = 0
            
        if template:
            # Shuffle the changes
            np.random.shuffle(true_changes)
            np.random.shuffle(left_out)
            description = ""
            for change in true_changes[:num_true_positives]:
                    description += change + " "
            
            for change in left_out[:num_false_positives]:
                description += change + " "

            examples[image_name] = description
        
        else:
            tp = "Changes:\n"
            # Create the validation examples
            for change in gt_changes[random_images[0]][:num_true_positives]:
                tp += "- "+change + "\n"
                
            print(tp)
            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role":"system", "content": "You have some examples of output. The user provides a list of changes that occurred in a certain area. The assistant must produce a description, such that each change is clearly deducible from it. The assistant must not directly mention the changes."},
                {"role":"system", "content": examples},
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": tp +"\nDescription:"
                    }
                ]
                }
            ],
            "max_tokens": 512
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            
            print(response.json()["choices"][0]["message"]["content"])
        
    # Save the examples
    if template:
        with open("validate_llm_eval/validation_examples_template_"+str(true_positives)+"_"+str(false_positives)+".json", "w") as f:
            json.dump(examples, f, indent=4)
            

def calculate_perfect_score(templates_path:str):
    # This function calculates the "perfect score" of the template paragraph. Since we know which changes are in the paragraph, we can construct the "perfect" matrix, which is the matrix that the LLM should return if perfect. 
    # Then we can compare this matrix to the one returned by the LLM.
    # Create the dictionary containing the results for each change
    
    # GT changes 
    with open("GT_changes.json", "r") as f:
        changes_gt = json.load(f)
    
    all_scores = np.zeros((11,11))
    
    for template in tqdm(os.listdir(templates_path)):
        # Build the "ground truth" tensor
        ground_truth = -np.ones((50, len(ST_CLASSES), len(ST_CLASSES)))
        predictions = -np.ones((50, len(ST_CLASSES), len(ST_CLASSES)))
        
        if("examples" in template):
            data = json.load(open(os.path.join(templates_path, template), "r"))
        
            pieces = template.split("_")
            tp = int(float(pieces[-2])*10)
            fp = int(float(pieces[-1].split(".json")[0])*10)     
            
            for i, (img_name, template_description) in enumerate(data.items()):
                chunks = template_description.strip().split(".")[:-1]
                for chunk in chunks:
                    pieces = chunk.strip().split(" ")
                    class_1 = pieces[1]
                    if class_1 == "low":
                        class_1 = "low_vegetation"
                        class_2 = pieces[8]
                    elif class_1 == "sports":
                        class_1 = "sports_field"
                        class_2 = pieces[8]
                    else:
                        class_2 = pieces[7]
                        
                    if class_2 == "low":
                        class_2 = "low_vegetation"
                        
                    if class_2 == "sports":
                        class_2 = "sports_field"

                    predictions[i, ST_CLASSES.index(class_1), ST_CLASSES.index(class_2)] = 1
                
                gt_chang = changes_gt[img_name]
                for change in gt_chang:
                    pieces = change.strip().split(" ")
                    class_1 = pieces[1]
                    if class_1 == "low":
                        class_1 = "low_vegetation"
                        class_2 = pieces[8]
                    elif class_1 == "sports":
                        class_1 = "sports_field"
                        class_2 = pieces[8]
                    else:
                        class_2 = pieces[7]
                    
                    if class_2 == "low":
                        class_2 = "low_vegetation"
                    
                    if class_2 == "sports":
                        class_2 = "sports_field"
                    
                    ground_truth[i, ST_CLASSES.index(class_1), ST_CLASSES.index(class_2)] = 1
            
            recalls = np.zeros((ground_truth.shape[0],))
            precisions = np.zeros((ground_truth.shape[0],))
            for i in range(ground_truth.shape[0]):
                detections = 0 
                missed = 0 
                false = 0 
                for j in range(ground_truth.shape[1]):
                    for z in range(ground_truth.shape[2]):
                        if j != z:
                            if ground_truth[i,j,z] == 1 and predictions[i,j,z] == 1:
                                detections += 1
                            if ground_truth[i,j,z] == 1 and predictions[i,j,z] != -1:
                                missed += 1
                            if ground_truth[i,j,z] == -1 and predictions[i,j,z] == 1:
                                false += 1
                
                if detections == 0:
                    recalls[i] = 0
                    precisions[i] = 0
                else:
                    recalls[i] = detections/(detections+missed)
                    precisions[i] = detections/(detections+false)
            
            f1_scores = 2*(precisions*recalls)/(precisions+recalls)
            # Substitute 0 for nan values in f1_scores
            f1_scores = np.nan_to_num(f1_scores)
            average = np.mean(f1_scores)
            all_scores[tp, fp] = round(average,2)
        
        print(all_scores)
        
    with open("best_scores.pkl", "wb") as f:
        pickle.dump(all_scores, f)
        
def evaluate_with_GPT35(image):
    '''
    This function evaluates all the results of the five methods on a particular image using GPT 3.5
    '''
    files = ["results_GPT4/cds.json", "results_otter/results_otter_chat.json", "results_otter/results_otter_chat_template.json", "results_otter/results_otter_direct.json", "results_otter/results_otter_indirect.json"]
    classes = ['water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
    
    # OpenAI API Key
    api_key = keys.OPENAI_API_KEY
    predictions = np.zeros((len(classes), len(classes)))
    
    for file in files:
        skipped = 0 
        cds = json.load(open(file, "r"))
        try: 
            description = cds[image]
        except:
            description = cds[image+".png"]
        print("Evaluating " + file)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j:
                    change = "a " + classes[i]+" area has transformed into a "+classes[j]+" area."
                    # Build the prompt 
                    prompt = "Here is a paragraph describing some changes: \"<paragraph>\". In the paragraph, are there references to the fact that <fact>?"
                    prompt = prompt.replace("<paragraph>", description)
                    prompt = prompt.replace("<fact>", change)
                    # Interrogate GPT 3.5 with this prompt
                    headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                    }
                    
                    payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {
                        "role": "user",
                        "content": [
                            {
                            "type": "text",
                            "text": prompt
                            }
                        ]
                        }
                    ],
                    "max_tokens": 512,
                    "temperature": 0.0,
                    }
                    
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                    
                    response = response.json()["choices"][0]["message"]["content"]
                    
                    if "yes," in response.lower() or "yes." in response.lower() or "yes</s>" in response.lower():
                        #percentage = percentages[0]
                        predictions[i, j] = 1 #* percentage
                    elif "no," in response.lower() or "no." in response.lower() or "no</s>" in response.lower():
                        #percentage = percentages[1]
                        predictions[i, j] = -1 #* percentage
                    else:
                        skipped += 1
        
        print("Skipped answers: ", skipped)
        import matplotlib.pyplot as plt
        plt.imshow(predictions,vmin=-1, vmax=1)
        plt.savefig(file+".png")
    

if __name__=="__main__":
    create_gt_changes()
    # import os
    # path_cds = "validate_llm_eval"
    
    # results = evaluate_with_llm(path_cds, device="cuda:0", bunch=True)
    
    # with open("validate_llm_eval/validation_results_template_1.0_1.0.json", "w") as f:
    #     json.dump(results, f, indent=4)
    
    # average = llm_evaluation_summary("results_otter/evaluation_results_otter_chat_template.json", "GT_changes.json")
    # print(average)
    #evaluate_with_GPT35("04639")
    # all_results = np.zeros((11,11))
    # for i, tp in enumerate(range(0, 11, 1)):
    #     tp = tp/10
    #     for j, fp in enumerate(range(0, 11, 1)):
    #         if tp==0.0 and fp==0:
    #             continue
    #         fp = fp/10
    #         print(tp, fp)
    #         path = "validate_llm_eval/validation_results_template_"+str(tp)+"_"+str(fp)+".json"
    #         score = llm_evaluation_summary(path,"GT_changes.json")
    #         all_results[i,j] = round(score,2)
    # print(all_results)
    
    # # Save the matrix
    # with open("validation_results_vicuna.pkl", "wb") as f:
    #     pickle.dump(all_results, f)

    # evaluate_with_llm("results_GPT4/cds.json", device="cuda:1")
        
    # print(all_results)
    # # validate_templates("validate_llm_eval")
    # bleu_evaluation_summary("CDVQA_dataset/qualitative_summaries.json", "results_GPT4/cds.json")
    # create_qualitative_summaries("CDVQA_dataset/Test_CD_VQA_summary_Final.json")
    # calculate_perfect_score("validate_llm_eval")