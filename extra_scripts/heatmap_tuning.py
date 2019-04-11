
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


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


print(len(X), len(Y), len(Z), len(M))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(X, Y, Z, c=M, cmap=plt.hot())
plt.show()