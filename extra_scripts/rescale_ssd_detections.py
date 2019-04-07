import numpy as np
import os
import csv
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import pickle

""" 
Its a utility file that can be used to rescale the SSD detection.
This rescaling is required because the detection outputs given by SSD model are scaled based on the width and height 
of the images, hence we need to rescale the detections back to original scale for plotting the bounding boxes
"""

DET_PATH_SSD = "../ensemble_scripts/dets_numpy/ssd"
DET_PATH_FRCNN = "../ensemble_scripts/dets_numpy/frcnn"
DET_PATH_RETINA = "../ensemble_scripts/dets_numpy/retina"
DET_PATH_COMBINED = "../ensemble_scripts/dets_numpy/input_ensembles"
VALID = "../data_set_files/valid.txt"
VAL_IMAGE_NAMES = "../data_set_files/val_image_names.txt"
ssd = []
frcnn = []
retina = []
combined = []


def eval_format(ens_detections):
    """ change format of detections to input_ensemble format """

    permutations = [6, 4, 5, 0, 1, 2, 3]
    new = [[[x[r]for r in permutations] for x in img] for img in ens_detections]
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
    return frcnn


def extract_retina(lst):
    """ reformats retina detections """

    retina = []
    for item in lst:
        for item1 in item[2]:
            retina.append(eval_format_single(np.asarray(item1)))
    with open("retina.csv", 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(retina)
    return frcnn


def read_retina_image_shapes(path):
    """ read validation images from the path provided """

    file_obj = open(path, "r")
    shapes = []
    print("Reading image sizes")
    i=0
    for item in file_obj:
        item = item.strip()
        if item:
            if (i % 500 == 0) & (i > 0) :
                print("Read %d/20,000 images"% i)
            image = read_image_bgr(item)
            shapes.append(image.shape)
        i+=1
    print("Finished reading images")
    return shapes


def rescale_detections(ssd, shapes):
    """ rescale the detections given by ssd model """

    rescaled_boxes = []
    for i , item in enumerate(ssd):

        height = shapes[i][0]
        width = shapes[i][1]

        height_scale = float(height) / 512
        width_scale = float(width) / 512

        new_item = []
        for det in item:
            det_item=[]
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
    for i in range(0, len(image_names)):
        name = ((image_names[i].split('/')[-1]).split('.')[0]).lstrip("0")
        if name == "":
            name = "0"
        [j.insert(len(j), name.strip()) for j in rescaled[i]]
    return rescaled


def read_image_names_from_list(path):
    """ read the image names from given file """

    f = open(path, 'r')
    x = f.readlines()
    f.close()
    return x


def replace_id_with_name(lst):
    """ Replacing integer class ids in detection list with class names """

    retina_labels = {0: 'car', 1: 'articulated_truck', 2: 'bus', 3: 'bicycle', 4: 'motorcycle',
                     5: 'motorized_vehicle', 6: 'pedestrian', 7: 'single_unit_truck', 8: 'work_van',
                     9: 'pickup_truck', 10: 'non-motorized_vehicle'}
    for item in lst:
        for j in item:
            j[4] = retina_labels[j[4]]
    return lst

image_names = read_image_names_from_list(VAL_IMAGE_NAMES)

""" reading the shapes of validation images """
# shapes = read_retina_image_shapes(VALID)
""" saving the result in pickle file, which can be used later on """
# with open('image_shapes', 'wb') as fp:
#    pickle.dump(shapes, fp)

""" loading image shapes from pickle file """
with open ('../data_set_files/image_shapes', 'rb') as fp:
    shapes = pickle.load(fp)

for filename in sorted(os.listdir(DET_PATH_SSD)):
    if ".npy" in filename:
        formatted_op_loaded = np.load(DET_PATH_SSD+'/'+filename)
        ssd.extend(formatted_op_loaded)

rescaled = rescale_detections(ssd, shapes)
rescaled = add_image_names_to_detections(rescaled, image_names)
print("Replacing integer class ids with class names")
rescaled = replace_id_with_name(rescaled)
rescaled = eval_format(rescaled)
final_det = transform_detections(rescaled)
print(final_det)
print(len(final_det))

print("Writing ssd results to csv")
with open("ssd.csv", 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(final_det)
print("Finished.")

exit(1)
