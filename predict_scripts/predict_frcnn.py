from matplotlib import pyplot as plt
import numpy as np
import gluoncv as gcv
from gluoncv.utils import download, viz

VAL_IMAGE_NAMES = "../data_set_files/valid.txt"

"""
This file can be used to make predictions on new images using Faster R-CNN model

Input:
    image or a list of images
Output:
    images are plotted with bounding boxes around the vehicles present in each image
"""

def non_zero(lst):
    lst = np.squeeze(lst[0])
    index = [i for i, e in enumerate(lst) if e != -1 and e > 0.5]
    return index


def read_image_names_from_list(path):
    f = open(path, 'r')
    x = f.readlines()
    f.close()
    return x


def remove_nl(lst):
    return [item.rstrip() for item in lst]


image_names = remove_nl(read_image_names_from_list(VAL_IMAGE_NAMES))

# 206, 118
classes = ['car', 'articulated_truck', 'bus', 'bicycle', 'motorcycle', 'motorized_vehicle', 'pedestrian',
           'single_unit_truck', 'work_van', 'pickup_truck', 'non-motorized_vehicle']

net = gcv.model_zoo.faster_rcnn_resnet50_v1b_voc(pretrained_base=False)
net.reset_class(classes)

net.load_parameters('../models_for_ensemble/faster_rcnn_resnet50_v1b_voc_0061_0.7285.params')

x, image = gcv.data.transforms.presets.ssd.load_test('../sample_data/00110281.jpg', 512)
cid, score, bbox = net(x)
ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
plt.show()

x, image = gcv.data.transforms.presets.ssd.load_test('../sample_data/00110288.jpg', 512)
cid, score, bbox = net(x)
ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
plt.show()

x, image = gcv.data.transforms.presets.ssd.load_test('../sample_data/00110306.jpg', 512)
cid, score, bbox = net(x)
ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
plt.show()

x, image = gcv.data.transforms.presets.ssd.load_test('../sample_data/00110374.jpg', 512)
cid, score, bbox = net(x)
ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
plt.show()

x, image = gcv.data.transforms.presets.ssd.load_test('../sample_data/00110390.jpg', 512)
cid, score, bbox = net(x)
ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
plt.show()
