from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import json
from tqdm import tqdm
from evaluate import load
import pickle


def get_scores(predictions, references):
    scores_dict = dict()
    # Score the predictions
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Rouge(), "ROUGE_L"),
        # (Meteor(), "METEOR"),
        # (Cider(), "CIDEr"),
        # (Spice(), "SPICE"),
    ]

    for scorer, method in scorers:
        print("computing %s score..." % (scorer.method()))
        score, _ = scorer.compute_score(references, predictions)
        if type(score) == list:
            for i in range(len(score)):
                scores_dict[method[i]] = round(score[i], 3)
        else:
            scores_dict[method] = round(score, 3)

    return scores_dict


def adapt_preds(preds, references=None, method="one"):
    new_preds = dict()
    if method == "one":
        for key in preds.keys():
            new_preds[key] = [data[key]]
    elif method == "two":
        all_imgs = list(references.keys())
        for img in all_imgs:
            img_name = img.split(".")[0]
            try:
                new_preds[img] = [
                    preds[preds["image_id"] == img_name]["text"].values[0]
                ]
            except:
                new_preds[img] = ["There are no notable changes in the two images"]

    return new_preds


if __name__ == "__main__":
    # Load the references
    with open("CDVQA_dataset/Test_CD_VQA_summary_Final.json", "r") as file:
        data = json.load(file)

    references = dict()
    for image in data["CDVQA"]:
        references[image["image"]] = [image["summary"]]

    # with open("results/change_descriptions.json", "rt") as file:
    #     data = json.load(file)

    with open("results/data_test_new_final_template.pkl", "rb") as file:
        data = pickle.load(file)

    # Format the data correctly
    # Method "one" for json file
    # Method "two" for pickle file
    predictions = adapt_preds(data, references=references, method="two")

    # Compute standard scores
    scores = get_scores(predictions, references)
    print(scores)

    # Compute bert score
    bertscore = load("bertscore")
    f1 = 0
    # Compute BERT score
    print("Computing bert score")
    for image, caption in tqdm(predictions.items()):
        results = bertscore.compute(
            predictions=caption,
            references=[references[image]],
            lang="en",
            rescale_with_baseline=True,
            model_type="microsoft/deberta-xlarge-mnli",
        )
        f1 += results["f1"][0]

    print(f1 / len(predictions))
