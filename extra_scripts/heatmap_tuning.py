
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.pyplot as plt
import ternary
from matplotlib import cm

TYPE_OF_WEIGHTS = '/simple_weights'

def color_point(x, y, z, scale):
    w = 255
    x_color = x * w / float(scale)
    y_color = y * w / float(scale)
    z_color = z * w / float(scale)
    r = math.fabs(w - y_color) / w
    g = math.fabs(w - x_color) / w
    b = math.fabs(w - z_color) / w
    return (r, g, b, 1.)


def color_point_new(acc):
    hot = cm.get_cmap('jet', 12)
    rgba = hot(acc)
    return (rgba)


def generate_heatmap_data(scale=5):
    from ternary.helpers import simplex_iterator
    d = dict()
    for (i, j, k) in simplex_iterator(scale):
        d[(i, j, k)] = color_point(i, j, k, scale)
    return d

def generate_heatmap_data_new(scale=20):
    d = dict()
    acc = M #read mAP values for all simplex_iterator values generated in tune_weights file
    from ternary.helpers import simplex_iterator
    a=0
    for (i, j, k) in simplex_iterator(scale):
        d[(i, j, k)] = color_point_new(float(acc[a]))
        a+=1
    return d


def read_data_from_file(file_obj):
    for item in file_obj:
        item = item.strip()
        if item is not "":
            M.append(item.split(":")[-1].strip())
            w = (item.split("M")[0].strip()[1:-2]).split(",")
            X.append(float(w[0].strip()))
            Y.append(float(w[1].strip()))
            Z.append(float(w[2].strip()))


def normalize_accuracy(acc):
    acc_n = []
    for item in acc:
        acc_n.append(float(item))

    min = np.amin(np.array(acc_n))
    max = np.amax(np.array(acc_n))

    acc_n = (np.array(acc_n) - min) / (max - min)

    return acc_n


file_obj = open("../ensemble_scripts/tuning_results" + TYPE_OF_WEIGHTS + "/mAP_tuned_weights.txt", "r")

X=[]
Y=[]
Z=[]
M=[]


read_data_from_file(file_obj)

accuracy = normalize_accuracy(M)
title_size = 18
fontsize = 10
scale = 20
offset = 0.13

cb_kwargs = {"shrink": 0.75,
             "orientation": "vertical",
             "label": "Accuracy",
             }

data = generate_heatmap_data_new(scale)
figure, tax = ternary.figure(scale=scale)
tax.heatmap(data, style="triangular", use_rgba=True, colorbar=True, cmap=cm.get_cmap('jet'),
            cb_kwargs=cb_kwargs)
# Remove default Matplotlib Axes
tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')
tax.boundary()
tax.set_title("Accuracy heatmap", fontsize=title_size)

tax.right_corner_label("Faster R-CNN", fontsize=fontsize, offset=0.10)
tax.left_corner_label("RetinaNet", fontsize=fontsize, offset=0.10)
tax.top_corner_label("SSD", fontsize=fontsize, offset=offset)

plt.show()
