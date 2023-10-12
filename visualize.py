import streamlit as st
import json
import random
from PIL import Image
import pickle

# Make a choice between three results using a streamlit radio button

result = st.radio("Choose a result", ("Result 1", "Result 2", "Result 3"))

if result == "Result 1":
    # Load the data
    with open("results/img_dialogues.json", "r") as file:
        dialogues = json.load(file)
    with open("results/summaries.json", "r") as file:
        summaries = json.load(file)
    with open("results/change_descriptions.json", "r") as file:
        change_descriptions = json.load(file)

    all_imgs = list(dialogues.keys())

    if st.button("Get random image!"):
        # Get random img
        img_name = random.choice(all_imgs)
        st.write(img_name)
        # Show the image, and the three results
        col1, col2, col3 = st.columns(3)
        # Load the image
        path1 = "/media/riccardoricci/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/im1"
        path2 = "/media/riccardoricci/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/im2"
        img1 = Image.open(f"{path1}/{img_name}")
        img2 = Image.open(f"{path2}/{img_name}")
        col1.image(img1, caption="Image 1")
        col3.image(img2, caption="Image 2")
        # Plot results
        st.write(summaries[img_name])


elif result == "Result 2":
    # Load the data
    with open("results/data_test_new_final.pkl", "rb") as file:
        change_descriptions = pickle.load(file)

    all_imgs = list(change_descriptions["image_id"])
    # Add the png extension
    all_imgs = [f"{img}.png" for img in all_imgs]

    if st.button("Get random image!"):
        # Get random img
        img_name = random.choice(all_imgs)
        # Show the image, and the three results
        col1, col2, col3 = st.columns(3)
        # Load the image
        path1 = "/media/riccardoricci/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/im1"
        path2 = "/media/riccardoricci/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/im2"
        img1 = Image.open(f"{path1}/{img_name}")
        img2 = Image.open(f"{path2}/{img_name}")
        col1.image(img1, caption="Image 1")
        col3.image(img2, caption="Image 2")
        # Plot results
        st.write(
            change_descriptions[
                change_descriptions["image_id"] == img_name.split(".")[0]
            ]["text"].values[0]
        )
elif result == "Result 3":
    # Load the data
    with open("results/data_test_new_final_template.pkl", "rb") as file:
        change_descriptions = pickle.load(file)

    all_imgs = list(change_descriptions["image_id"])
    # Add the png extension
    all_imgs = [f"{img}.png" for img in all_imgs]

    if st.button("Get random image!"):
        # Get random img
        img_name = random.choice(all_imgs)
        # Show the image, and the three results
        col1, col2, col3 = st.columns(3)
        # Load the image
        path1 = "/media/riccardoricci/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/im1"
        path2 = "/media/riccardoricci/Melgani/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/public/test/im2"
        img1 = Image.open(f"{path1}/{img_name}")
        img2 = Image.open(f"{path2}/{img_name}")
        col1.image(img1, caption="Image 1")
        col3.image(img2, caption="Image 2")
        # Plot results
        st.write(
            change_descriptions[
                change_descriptions["image_id"] == img_name.split(".")[0]
            ]["text"].values[0]
        )
