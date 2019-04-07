from matplotlib import pyplot as plt
import gluoncv as gcv
from gluoncv.utils import download, viz

"""
This file can be used to make predictions on new images using SSD-Subset model

Note: SSD-Subset model detects only a subset of vehicles like: bicycle, motorcycle, pedestrian etc.

Input:
    image or a list of images
Output:
    images are plotted with bounding boxes around the vehicles present in each image
"""

classes = ['bicycle', 'motorcycle', 'motorized_vehicle', 'pedestrian',
           'work_van', 'non-motorized_vehicle']

net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained_base=False)
net.reset_class(classes)
net.load_parameters('../models_for_ensemble/epoch_29_ssd_512_resnet50_v1_voc_mio_tcd.params')


x, image = gcv.data.transforms.presets.ssd.load_test('../sample_data/00000049.jpg', 512)
cid, score, bbox = net(x)
ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
plt.show()


x, image = gcv.data.transforms.presets.ssd.load_test('../sample_data/00000118.jpg', 512)
cid, score, bbox = net(x)
ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
plt.show()

x, image = gcv.data.transforms.presets.ssd.load_test('../sample_data/00000124.jpg', 512)
cid, score, bbox = net(x)
ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
plt.show()
