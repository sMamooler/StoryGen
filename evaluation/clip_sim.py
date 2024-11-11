import os
import numpy as np
import clip
import torch
import torch.utils.data
import torch.utils.checkpoint

from tqdm.auto import tqdm
from PIL import Image


def calc_clip_score(golds, preds):
    model, preprocess = clip.load("ViT-L/14")

    scores = []
    for gold_img, pred_img in zip(golds, preds):
        gold_img_input = preprocess(gold_img).unsqueeze(0)
        pred_img_input = preprocess(pred_img).unsqueeze(0)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        gold_img_input = gold_img_input.to(device)
        pred_img_input = pred_img_input.to(device)
        model = model.to(device)

        with torch.no_grad():
            gold_img_features = model.encode_image(gold_img_input)
            pred_img_features = model.encode_image(pred_img_input)

        # Normalize the features
        gold_img_features = gold_img_features / gold_img_features.norm(
            dim=-1, keepdim=True
        )
        pred_img_features = pred_img_features / pred_img_features.norm(
            dim=-1, keepdim=True
        )

        # Calculate the cosine similarity to get the CLIP score
        clip_score = torch.matmul(pred_img_features, gold_img_features.T).item()

        scores.append(clip_score)

    return np.mean(scores)
