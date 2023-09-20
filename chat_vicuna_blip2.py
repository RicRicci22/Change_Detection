from utils.chat import *
from PIL import Image
import os
import json
from tqdm import tqdm
from utils.images_utils import extract_blobs, get_largest_poly, crop_image
import numpy as np
import matplotlib.pyplot as plt 
if __name__ == "__main__":
    dataset_path_im1 = "/media/melgani/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/im1"
    dataset_path_im2 = "/media/melgani/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/im2"

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
