import numpy as np
import os
import csv

"""
This file extracts ground truth for validation image set
Input: 
    ground truth of whole dataset: gt_train.csv
    validation set image names : val_image_names.txt
Output: ground truth of validation dataset: gt_val.csv
"""

PATH_GT_FULL = "./gt_train.csv"
PATH_VAL_IMGS = "./val_image_names.txt"
val_gt =[]
with open(PATH_VAL_IMGS, 'r') as f:
    val_imgs = (f.readlines())

val_imgs = [img.rstrip() for img in val_imgs]


with open(PATH_GT_FULL, "r") as f:
    reader = csv.reader(f)
    for i, line in enumerate(reader):
        x = line[0].zfill(8)
        if x in val_imgs:
            val_gt.append(line)

with open("gt_val.csv", 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(val_gt)
