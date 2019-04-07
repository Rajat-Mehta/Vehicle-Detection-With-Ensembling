from matplotlib import pyplot as plt
import gluoncv as gcv
from gluoncv.utils import download, viz

"""
This file can be used to make predictions on new images using SSD model

Input:
    image or a list of images
Output:
    images are plotted with bounding boxes around the vehicles present in each image
"""

classes = ['car', 'articulated_truck', 'bus', 'bicycle', 'motorcycle', 'motorized_vehicle', 'pedestrian',
           'single_unit_truck', 'work_van', 'pickup_truck', 'non-motorized_vehicle']

net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained_base=False)
net.reset_class(classes)
net.load_parameters('../models_for_ensemble/epoch_22_ssd_512_resnet50_v1_voc_mio_tcd.params')


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
