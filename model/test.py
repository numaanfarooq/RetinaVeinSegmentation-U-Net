import os
import time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from model import build_unet
from utlis import create_dir, seeding

def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

def ensure_dir_exists(directory):
    """ Ensures that a directory exists and has write permissions """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    elif not os.access(directory, os.W_OK):
        raise PermissionError(f"No write access to the directory: {directory}")
    else:
        print(f"Directory {directory} is ready for saving results.")

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    results_dir = "results"
    ensure_dir_exists(results_dir)  # Ensure the results folder is ready

    """ Load dataset """
    test_x = sorted(glob("./new_data/test/image/*"))
    test_y = sorted(glob("./new_data/test/mask/*"))

    if len(test_x) == 0 or len(test_y) == 0:
        raise ValueError("No test images or masks found in the specified paths.")

    """ Hyperparameters """
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = "files/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        name = os.path.basename(x).split(".")[0]  # Use os.path.basename for cross-platform compatibility

        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)  ## (512, 512, 3)
        if image is None:
            print(f"Warning: Failed to read image {x}")
            continue
        ## image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        """ Reading mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
        if mask is None:
            print(f"Warning: Failed to read mask {y}")
            continue
        ## mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
        y = y / 255.0
        y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))

            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        ori_mask = mask_parse(mask)  # mask is already a numpy array, no need to call .cpu()
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate(
            [image, line, ori_mask, line, pred_y * 255], axis=1
        )

        # Use os.path.join to create system-agnostic file paths
        save_path = os.path.join(results_dir, f"{name}.png")
        success = cv2.imwrite(save_path, cat_images)
        if not success:
            print(f"Failed to save image at {save_path}")
        else:
            print(f"Image saved successfully at {save_path}")

    """ Calculate average metrics """
    jaccard = metrics_score[0] / len(test_x)
    f1 = metrics_score[1] / len(test_x)
    recall = metrics_score[2] / len(test_x)
    precision = metrics_score[3] / len(test_x)
    acc = metrics_score[4] / len(test_x)
    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")

    fps = 1 / np.mean(time_taken)
    print("FPS: ", fps)
