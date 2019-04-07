import csv
import pandas as pd

"""
Extracts ground truth of validation set for SSD-Subset model in a separate csv file
"""

df = pd.read_csv('gt_val.csv', sep=',',  names = ["image_id", "class", "b1", "b2", "b3", "b4"])

filter_classes = ["bicycle", "motorcycle", "motorized_vehicle", "pedestrian", "work_van",
                                   "non-motorized_vehicle"]
df = df.loc[df['class'].isin(filter_classes)]

df.to_csv("filtered_gt_val.csv", sep=',', encoding='utf-8')
