import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

"""
This file plots the class distribution in our original dataset.
It can be used to see how many vehicles are there for each class in the dataset.
"""

data = pd.read_csv("../data_set_files/gt_train_plot.csv")

counter = Counter(data["obj_class"])
dic = dict(counter)
print(dic.keys())


dic['PT'] = dic.pop('pickup_truck')
dic['C'] = dic.pop('car')
dic['AT'] = dic.pop('articulated_truck')
dic['B'] = dic.pop('bus')
dic['MV'] = dic.pop('motorized_vehicle')
dic['WV'] = dic.pop('work_van')
dic['SUT'] = dic.pop('single_unit_truck')
dic['P'] = dic.pop('pedestrian')
dic['BC'] = dic.pop('bicycle')
dic['NMV'] = dic.pop('non-motorized_vehicle')
dic['MC'] = dic.pop('motorcycle')
print(dic)


plt.ylabel ('Count')
plt.xlabel ('Class')


plt.bar(list(dic.keys()), dic.values(), color='g')
plt.legend()
plt.show()
