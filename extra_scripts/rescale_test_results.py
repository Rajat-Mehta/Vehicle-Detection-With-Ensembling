import pickle
import csv

THRESH = 0.0
TEST_PATH = "../data_set_files/record_format_files/data_set_test/test.txt"
TEST_RESULT_PATH = "../ensemble_scripts/dets_numpy/test/results/result_ensemble_fsr.csv"


"""
image_names = [line.strip() for line in open(TEST_PATH)]

with open('../data_set_files/record_format_files/data_set_test/test_image_shapes', 'rb') as fp:
    shapes = pickle.load(fp)

shapess = []
scales = {}


for i, item in enumerate(shapes):
    shape = []
    h = item[0]
    w = item[1]
    im_size_min, im_size_max = (h, w) if w > h else (w, h)
    scale = float(512) / float(im_size_min)
    scales[image_names[i].split("/")[-1].split(".")[0]] = scale
    shapess.append([im_size_min, im_size_max])
"""

with open(TEST_RESULT_PATH, "r") as f:
    reader = csv.reader(f, delimiter=",")
    filtered_results = []

    for i, line in enumerate(reader):
        if float(line[2]) > THRESH:
            # scale = scales[line[0].zfill(8)]
            line[0] = line[0].zfill(8)
            line[3] = int(float(line[3]))
            line[4] = int(float(line[4]))
            line[5] = int(float(line[5]))
            line[6] = int(float(line[6]))
            filtered_results.append(line)


print("Writing filtered results to csv")
with open("../ensemble_scripts/dets_numpy/test/results/filtered_result_ensemble_fsr_" + str(THRESH) + "_rescaled.csv", 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(filtered_results)