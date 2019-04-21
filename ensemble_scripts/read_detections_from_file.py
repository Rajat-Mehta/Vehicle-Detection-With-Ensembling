import numpy as np
import os
import csv
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import pickle
from ensemble import *

"""
This file reads the detections of Faster R-CNN, SSD and SSD-Subset models from the saved numpy files and
combines them into a single list which is then fed to the ensemble.py file to get the final
ensemble detection output.

We ran predict_boxes.py(which takes one day to run) only once to generate the detection results of the three models 
and saved them in numpy files. Then we read those numpy files in this script to play around with them for ensemble
approaches.

Input:
    3 numpy files containing detections of each models SSD, SSD-Subset and Faster R-CNN

Output:
    result_ensemble_sfe.csv file which contains the Adaptive weight based ensemble output. this result file can
    then be used by localization_evaluation.py file to generate mAP of our ensemble model
"""

DET_PATH_SSD_EXPERT = "./dets_numpy/ssd_expert/"
DET_PATH_SSD = "./dets_numpy/ssd"
DET_PATH_FRCNN = "./dets_numpy/frcnn"
DET_PATH_RETINA = "./dets_numpy/retina/"
DET_PATH_COMBINED = "./dets_numpy/input_ensembles"
VALID = "../data_set_files/valid.txt"
VAL_IMAGE_NAMES = "../data_set_files/val_image_names.txt"
BATCH_NO = "_0-20000_sfe"


def eval_format(ens_detections):
    """ change format of detections to input_ensemble format """

    permutations = [6, 4, 5, 0, 1, 2, 3]
    new = [[[x[r]for r in permutations] for x in img] for img in ens_detections]
    return new


def eval_format_final(ens_detections):
    """ change format of ensemble detections """

    permutations = [6, 4, 5, 0, 1, 2, 3]
    new = [[r[x] for x in permutations] for r in ens_detections]
    return new


def transform_detections(lst):
    """ transforms lst into new format """

    final_det = []
    for img in lst:
        for det in img:
            final_det.append(det)
    return final_det


def eval_format_single(ens_detections):
    """ rearranges the values in detection vector """

    permutations = [6, 4, 5, 0, 1, 2, 3]
    new = [ens_detections[x] for x in permutations]
    return new


def keep_ensembled_confident(det):
    """ keep only confident detections """

    conf = []
    for item in det:
        if item[2] > 0.5:
            conf.append(item)
    return conf


def extract_frcnn(lst):
    """ reformats frcnn detections """

    frcnn = []
    for item in lst:
        for item1 in item[1]:
            frcnn.append(eval_format_single(np.asarray(item1)))
    with open("frcnn.csv", 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(frcnn)
    return frcnn


def extract_ssd(lst):
    """ reformats ssd detections """

    ssd = []
    for item in lst:
        for item1 in item[0]:
            ssd.append(eval_format_single(np.asarray(item1)))
    with open("ssd.csv", 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(ssd)
    return ssd


def extract_retina(lst):
    """ reformats retina detections """

    retina = []
    for item in lst:
        for item1 in item[2]:
            retina.append(eval_format_single(np.asarray(item1)))
    with open("retina.csv", 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(retina)
    return retina


def read_retina_image_shapes(path):
    """ read validation images from the path provided """

    file_obj = open(path, "r")
    shapes = []
    print("Reading image sizes")
    i = 0
    for item in file_obj:
        item = item.strip()
        if item:
            if (i % 500 == 0) & (i > 0):
                print("Read %d/20,000 images"% i)
            image = read_image_bgr(item)
            shapes.append(image.shape)
        i += 1
    print("Finished reading images")

    return shapes


def rescale_detections(ssd, shapes):
    """ rescale the detections given by ssd model """

    rescaled_boxes = []
    for i, item in enumerate(ssd):

        height = shapes[i][0]
        width = shapes[i][1]

        height_scale = float(height) / 512
        width_scale = float(width) / 512

        new_item = []
        for det in item:
            det_item = []
            det_item.append(det[0] * width_scale)
            det_item.append(det[1] * height_scale)
            det_item.append(det[2] * width_scale)
            det_item.append(det[3] * height_scale)
            det_item.append(det[4])
            det_item.append(det[5])
            new_item.append(det_item)

        rescaled_boxes.append(new_item)

    return rescaled_boxes


def add_image_names_to_detections(rescaled, image_names):
    """ Adding image names to the detection result lists """

    for i in range(0, len(rescaled)):
        name = ((image_names[i].split('/')[-1]).split('.')[0]).lstrip("0")
        if name.strip() == "":
            name = "0"
        [j.insert(len(j), name.strip()) for j in rescaled[i]]

    return rescaled


def read_image_names_from_list(path):
    """ read the image names from given file """

    f = open(path, 'r')
    x = f.readlines()
    f.close()

    return x


def replace_id_with_name(lst, filter):
    """ Replacing integer class ids in detection list with class names """

    retina_labels = {0: 'car', 1: 'articulated_truck', 2: 'bus', 3: 'bicycle', 4: 'motorcycle',
                     5: 'motorized_vehicle', 6: 'pedestrian', 7: 'single_unit_truck', 8: 'work_van',
                     9: 'pickup_truck', 10: 'non-motorized_vehicle'}

    filter_classes = {0: "bicycle", 1: "motorcycle", 2: "motorized_vehicle", 3: "pedestrian", 4: "work_van",
                      5: "non-motorized_vehicle"}
    if filter:
        classes = filter_classes
    else:
        classes = retina_labels

    for item in lst:
        for j in item:
            j[4] = classes[j[4]]

    return lst


def format_list_for_ensemble(formatted_op_ssd, formatted_op_frcnn, ssd_expert, ret_detection):
    """ creating a new list with detections from all models to feed into the model ensemble method """

    input_ensemble = []
    for item in range(0, len(formatted_op_ssd)):
        temp = []
        #temp_ssd = formatted_op_ssd[item]
        temp_frcnn = formatted_op_frcnn[item]
        temp_expert = ssd_expert[item]
        temp_ret = ret_detection[item]
        #temp.append(temp_ssd)
        temp.append(temp_frcnn)
        temp.append(temp_expert)
        temp.append(temp_ret)
        input_ensemble.append(temp)

    return input_ensemble


def write_csv_from_npy(path):
    """ write the npy to a csv """

    file = []

    if "ssd_expert" in path:
        file_type = "/ssd_expert"
    elif "ssd_coco" in path:
        file_type = "/ssd_coco"
    elif "ssd" in path:
        file_type = "/ssd"
    elif "frcnn" in path:
        file_type = "/frcnn"
    elif "retina" in path:
        file_type = "/retina"

    for filename in sorted(os.listdir(path)):
        if ".npy" in filename:
            formatted_op_loaded = np.load(path+'/'+filename)
            file.extend(formatted_op_loaded)
    npy = file

    with open('../data_set_files/image_shapes', 'rb') as fp:
        shapes = pickle.load(fp)

    if "ssd" in file_type:
        npy = rescale_detections(npy, shapes)

    npy = add_image_names_to_detections(npy, image_names)

    print("Replacing integer class ids with class names")
    if "ssd_expert" in file_type:
        npy = replace_id_with_name(npy, True)
    else:
        npy = replace_id_with_name(npy, False)
    result_format = eval_format(npy)
    final_det = transform_detections(result_format)

    print(len(final_det))

    print("Writing results to csv")
    print(path + file_type + BATCH_NO + ".csv")
    with open(path + file_type + BATCH_NO + ".csv", 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(final_det)

    return npy


image_names = read_image_names_from_list(VAL_IMAGE_NAMES)

"""
uncomment this if you want to use retina model in ensemble
shapes = read_retina_image_shapes(VALID)
with open('image_shapes', 'wb') as fp:
    pickle.dump(shapes, fp)

with open ('image_shapes', 'rb') as fp:
    shapes = pickle.load(fp)
"""

formatted_op_ssd_expert_npy = write_csv_from_npy(DET_PATH_SSD_EXPERT)
formatted_op_ssd_npy = write_csv_from_npy(DET_PATH_SSD)
formatted_op_frcnn_npy = write_csv_from_npy(DET_PATH_FRCNN)
ret_detection_npy = write_csv_from_npy(DET_PATH_RETINA)

# print(len(formatted_op_ssd_npy))
# print(len(formatted_op_frcnn_npy))
# print(len(formatted_op_ssd_expert_npy))
# print(len(ret_detection_npy))
input_ensemble_npy = format_list_for_ensemble(formatted_op_ssd_npy, formatted_op_frcnn_npy, formatted_op_ssd_expert_npy, ret_detection_npy)
print(len(input_ensemble_npy))
np.save('./dets_numpy/input_ensembles/input_ensembles' + BATCH_NO + '.npy', input_ensemble_npy)

# first half of validation set was used for tuning the weight parameters and second half is being used here
# for evaluating the selected optimal weight parameters
input_ensemble_npy = input_ensemble_npy[len(input_ensemble_npy) / 2:]
print(len(input_ensemble_npy))

print("Starting model ensembling")
final_predictions =[]
for item in input_ensemble_npy:
    ens = GeneralEnsemble(item, weights=[0.16, 0.84, 0.35, 0.5])
    list_ens = eval_format_final(np.asarray(ens))
    # list_ens = [[((im_fname.split('/')[-1]).split('.')[0]).lstrip("0")] + x for x in list_ens]
    final_predictions.extend(keep_ensembled_confident(list_ens))
print("Finished ensembling.")

print("Writing ensembled results to csv")
with open("./dets_numpy/results/result_ensemble_sfe.csv", 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(final_predictions)


print("Finished.")
exit(1)
