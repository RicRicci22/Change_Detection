'''
This piece of code is inteded to be used to generate a dialogue between the questioner AI and the answerer AI.
It can use full image or cropped image to generate the dialogue.
'''

from utils.chat import *
from utils.dataset import ChatSet
from PIL import Image
import os
import json
from tqdm import tqdm
from utils.images_utils import extract_blobs, get_largest_poly, crop_image
import numpy as np

def main(llms_params:dict, dataset_params:dict):
    '''
    This function run the dialogue and store the result.
    Parameters:
    llm_params: dict
        The parameters for the LLM model.
    dataset_params: dict
        The parameters for the dataset, and in general the handling of the images.
    '''
    # Sanity checks on llm params
    assert llms_params["answerer"] in ["blip2"], "Answerer not supported"
    assert llms_params["questioner"] in ["vicuna"], "Questioner not supported"
    assert os.path.exists(dataset_params["dataset_path_im1"]), "Path of dataset images not found"
    assert os.path.exists(dataset_params["dataset_path_im2"]), "Path of dataset images not found"
    
    # Sanity checks on dataset params
    assert os.path.exists(dataset_params["dataset_path"]), "Path of dataset not found"
    assert os.path.exists(os.path.join(dataset_params["dataset_path"],"im1")), "Folder of pre-change images not found"
    assert os.path.exists(os.path.join(dataset_params["dataset_path"],"im2")), "Folder of post-change images not found"
    assert dataset_params["crop"] in [True, False], "Define crop as True or False"
    
    if dataset_params["crop"] and dataset_params["use_labels"]:
        assert os.path.exists(dataset_params["dataset_path_labels_1"]), "Folder of pre-change labels not found"
        assert os.path.exists(dataset_params["dataset_path_labels_2"]), "Folder of post-change labels not found"
    elif dataset_params["crop"]:
        # in this case it should perform prediction and then crop
        pass 
    
    
    # Create the dataset that spits images names
    dataset = ChatSet(dataset_params["dataset_path"])
    
    images_names = os.listdir(os.path.join(dataset_params["dataset_path"],"im1")) # Names of the images 
    
    results = dict()
    chat = Chatter(
        answerer=llms_params["answerer_type"],
        model=llms_params["answerer_model"],
        a_device=llms_params["answerer_device"],
    )
    dialogue_steps = llms_params["dialogue_steps"]
    
    for img_name in tqdm(images_names):
        # Open the images
        img_1_pil = Image.open(os.path.join(dataset_params["dataset_path"],"im1",img_name))
        img_2_pil =Image.open(os.path.join(dataset_params["dataset_path"],"im2",img_name))
        
        if dataset_params["crop"]:
            if dataset_params["use_labels"]:
                mask_1 = Image.open(Image.open(os.path.join(dataset_params["dataset_path"],"label1",img_name)))
                mask_2 = Image.open(Image.open(os.path.join(dataset_params["dataset_path"],"label2",img_name)))
            else:
                # Implement the creation of masks for both images 
                raise NotImplementedError("This part is not implemented yet")
            
            # Extracts the blobs from the binary masks
            mask_array_1 = np.array(mask_1)
            polygons_1 = extract_blobs(mask_array_1)
            mask_array_2 = np.array(mask_2)
            polygons_2 = extract_blobs(mask_array_2)

            all_poly = polygons_1 + polygons_2
            # Get the largest polygon (largest changed area)
            largest_polygon = get_largest_poly(all_poly)

            if largest_polygon and largest_polygon.area>area_threshold:
                # Crop the image and substitute the original image with the cropped one
                img_1_pil = crop_image(img_1_pil, largest_polygon)
                img_2_pil = crop_image(img_2_pil, largest_polygon)

        for _ in range(dialogue_steps):
            # Pseudocode
            # Provide the first template question for all the images
            # question is a dictioanry with the name of the image and a list of questions and answers
            # Answer the question in batch with the answerer 
            #  
            # 
            question = chat.ask_question_API(batch=False)
            question = chat.question_trim(question)
            chat.conversation.append_question(question)
            answer = chat.answer_question(crop_1)
            chat.conversation.append_answer(answer)

            # Save in the dict
            questions, answers = chat.conversation.return_messages()
            results[img_name] = [{"questions": questions, "answers": answers}]

            chat.reset_history()

            for i in range(dialogue_steps):
                question = chat.ask_question_API()
                question = chat.question_trim(question)
                chat.conversation.append_question(question)
                answer = chat.answer_question(crop_2)
                chat.conversation.append_answer(answer)

            questions, answers = chat.conversation.return_messages()

            results[img_name].append({"questions": questions, "answers": answers})

            chat.reset_history()
        else:
            print("Polygon too small!")
    else:
        print("No polygon found for image: {}, or polygon too small".format(img_name))

# Save the dict
with open("results/img_dialogues_crop.json", "w") as file:
    json.dump(results, file, indent=4)
    

    

if __name__ == "__main__":
    llms_params = {
        "answerer_type": "blip2",
        "answerer_model": "flantxxl", # To revise
        "answerer_device": "cuda:1",
        "dialogue_steps": 10, # How many rounds of dialogue to generate
        "questioner_type": "vicuna",
        "questioner_model": "vicuna1.5", # To revise
        "questioner_device": "cuda:0",
    }
    
    dataset_params = {
        "dataset_path": "/media/melgani/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test",
        "crop": False,
        "area_threshold" : 5,
        "use_labels": True, # If true, it can use the labels to crop the image in the area of the biggest change
    }
    
    # Path of dataset images. here two because they are pre/post change images.
    dataset_path_im1 = "/media/melgani/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/im1"
    dataset_path_im2 = "/media/melgani/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/im2"
    crop = False
    # Path of dataset labels. Not needed if crop is set to false
    dataset_path_labels_1 = "/media/melgani/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/label1"
    dataset_path_labels_2 = "/media/melgani/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/label2"

    with open("CDVQA_dataset/Test_CD_VQA_summary_Final.json", "r") as file:
        test_merged = json.load(file)
    
    area_threshold = 5

    results = dict()
    
    device_answerer = "cuda:1"
    chat = Chatter(
        answerer="blip2",
        a_device=device_answerer,
    )
    dialogue_steps = 10

    for element in tqdm(test_merged["CDVQA"]):
        img_name = element["image"]
        path_image_1 = os.path.join(dataset_path_im1, img_name)
        path_image_2 = os.path.join(dataset_path_im2, img_name)

        path_label_1 = os.path.join(dataset_path_labels_1, img_name)
        path_label_2 = os.path.join(dataset_path_labels_2, img_name)

        mask_1 = Image.open(path_label_1)
        mask_2 = Image.open(path_label_2)

        img_1_pil = Image.open(path_image_1)
        img_2_pil = Image.open(path_image_2)

        mask_array_1 = np.array(mask_1)
        polygons_1 = extract_blobs(mask_array_1)

        mask_array_2 = np.array(mask_2)
        polygons_2 = extract_blobs(mask_array_2)

        all_poly = polygons_1 + polygons_2
        # Get the largest polygon (largest changed area)
        largest_polygon = get_largest_poly(all_poly)

        if largest_polygon and largest_polygon.area>area_threshold:
            crop_1 = crop_image(img_1_pil, largest_polygon)
            crop_2 = crop_image(img_2_pil, largest_polygon)

            if crop_1.size[0]>1 and crop_1.size[0]>1:
                for i in range(dialogue_steps):
                    question = chat.ask_question_API()
                    question = chat.question_trim(question)
                    chat.conversation.append_question(question)
                    answer = chat.answer_question(crop_1)
                    chat.conversation.append_answer(answer)

                # Save in the dict
                questions, answers = chat.conversation.return_messages()
                results[img_name] = [{"questions": questions, "answers": answers}]

                chat.reset_history()

                for i in range(dialogue_steps):
                    question = chat.ask_question_API()
                    question = chat.question_trim(question)
                    chat.conversation.append_question(question)
                    answer = chat.answer_question(crop_2)
                    chat.conversation.append_answer(answer)

                questions, answers = chat.conversation.return_messages()

                results[img_name].append({"questions": questions, "answers": answers})

                chat.reset_history()
            else:
                print("Polygon too small!")
        else:
            print("No polygon found for image: {}, or polygon too small".format(img_name))

    # Save the dict
    with open("results/img_dialogues_crop.json", "w") as file:
        json.dump(results, file, indent=4)
