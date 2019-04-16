
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.pyplot as plt
import ternary



file_obj = open("../ensemble_scripts/tuning_results/mAP_tuned_weights.txt", "r")

X=[]
Y=[]
Z=[]
M=[]

for item in file_obj:
    item = item.strip()
    if item is not "":
        M.append(item.split(":")[-1].strip())
        w = (item.split("M")[0].strip()[1:-2]).split(",")
        X.append(float(w[0].strip()))
        Y.append(float(w[1].strip()))
        Z.append(float(w[2].strip()))


def color_point(x, y, z, scale):
    w = 255
    x_color = x * w / float(scale)
    y_color = y * w / float(scale)
    z_color = z * w / float(scale)
    r = math.fabs(w - y_color) / w
    g = math.fabs(w - x_color) / w
    b = math.fabs(w - z_color) / w
    return (r, g, b, 1.)


def color_point_new(x, y, z, m):
    w = 255
    print(x,y,z)
    m_color = float(m) * w/255
    r = math.fabs(w - m_color) / w
    g=0
    b=0
    print(m_color)
    return (m_color, g, b, 1.)


def generate_heatmap_data(scale=5):
    from ternary.helpers import simplex_iterator
    d = dict()
    for (i, j, k) in simplex_iterator(scale):
        d[(i, j, k)] = color_point(i, j, k, scale)
        print(color_point(i, j, k, scale))
        exit()
    return d

def generate_heatmap_data_new(scale=5):
    d = dict()
    for i in range(len(X)):
        total = X[i] + Y[i] + Z[i]
        x_norm = X[i] / total
        y_norm = Y[i] / total
        z_norm = Z[i] / total
        d[(X[i], Y[i], Z[i])] = color_point_new(x_norm, y_norm, z_norm, M[i])
    return d

scale = 1
data = generate_heatmap_data_new(scale)
figure, tax = ternary.figure(scale=scale)
tax.heatmap(data, style="hexagonal", use_rgba=True, colorbar=False)
# Remove default Matplotlib Axes
tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')
tax.boundary()
tax.set_title("RGBA Heatmap")
plt.show()
exit()

print(len(X), len(Y), len(Z), len(M))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(X, Y, Z, c=M, cmap=plt.hot())
plt.show()