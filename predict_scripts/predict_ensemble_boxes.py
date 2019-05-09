from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
from gluoncv.utils import viz
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet import autograd, gluon
import sys
sys.path.insert(0, '../ensemble_scripts/ensemble')
from ensemble import GeneralEnsemble
import cv2
import csv
import os
import collections
import numpy as np
import mxnet as mx
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultValTransform
import gluoncv as gcv
from gluoncv.data import batchify
import pickle
import matplotlib.image as mpimg
#from predict_retinanet import predict, plot_pred

BATCH_NO = '_24000-27743'
TEST_PATH = "../data_set_files/record_format_files/data_set_test/test_mini.txt"

"""
This file can be used to make predictions on new images using our Adaptive Weight based model ensemble

Input:
    image or a list of images
Output:
    images are plotted with bounding boxes around the vehicles present in each image
    
In the output you'll see the plots of 4 images:
    Figure 1: output of SSD model
    Figure 2: output of Faster-RCNN model
    Figure 3: output of SSD-Subset model
    Figure 4: output of our Adaptive weight based ensemble model
"""


def get_class_labels():
    """ return class labels according to retina-net and gluon-cv model formats """

    retina_labels = {0: 'car', 1: 'articulated_truck', 2: 'bus', 3: 'bicycle', 4: 'motorcycle',
                     5: 'motorized_vehicle', 6: 'pedestrian', 7: 'single_unit_truck', 8: 'work_van',
                     9: 'pickup_truck', 10: 'non-motorized_vehicle'}
    gluon_labels = ['car', 'articulated_truck', 'bus', 'bicycle', 'motorcycle', 'motorized_vehicle', 'pedestrian',
                    'single_unit_truck', 'work_van', 'pickup_truck', 'non-motorized_vehicle']

    filter_classes = ["bicycle", "motorcycle", "motorized_vehicle", "pedestrian", "work_van",
                      "non-motorized_vehicle"]

    return retina_labels, gluon_labels, filter_classes


def get_model_dicts():
    """ define a dictionary of trained models with their names and weight file names """

    models_gcv = collections.defaultdict(dict)
    models_gcv['SSD']['model'] = 'ssd_512_resnet50_v1_voc'
    models_gcv['SSD']['weights'] = '../models_for_ensemble/epoch_22_ssd_512_resnet50_v1_voc_mio_tcd.params'
    models_gcv['FASTER-RCNN']['model'] = 'faster_rcnn_resnet50_v1b_voc'
    models_gcv['FASTER-RCNN']['weights'] = '../models_for_ensemble/faster_rcnn_resnet50_v1b_voc_0061_0.7285.params'
    models_gcv['SSD-EXPERT']['model'] = 'ssd_512_resnet50_v1_voc'
    models_gcv['SSD-EXPERT']['weights'] = '../models_for_ensemble/epoch_29_ssd_512_resnet50_v1_voc_mio_tcd.params'
    models_gcv = dict(models_gcv)

    return models_gcv


def get_model(model, weights, classes):
    """ get model from gluoncv """

    net = model_zoo.get_model(model, pretrained=True)
    if weights is not 'NONE':
        net.reset_class(classes)
        net.load_parameters(weights)

    return net


def non_zero(lst, thresh):
    """ return indexes of items which are not -1 and value is greater than 0.40 """

    return [i for i, e in enumerate(lst) if e > thresh]


def slice_frcnn_list(cid, score, bbox, thresh):
    """ ignore detections where score < thresh """

    cid_new = []
    score_new = []
    bbox_new = []

    for c, s, b in zip(cid, score, bbox):
        index = non_zero(s, thresh)
        if len(index) is not 0:
            cid_new.append(c[index])
            score_new.append(s[index])
            bbox_new.append(b[index])
        else:
            cid_new.append(np.asarray([0]))
            score_new.append(np.asarray([0]))
            bbox_new.append(np.asarray([[0, 0, 0, 0]]))

    return cid_new, score_new, bbox_new


def get_ssd_detections(net, image_names):
    """ Feed data to SSD model and get its detections """
    formatted_ssd = []
    x, orig_img = gcv.data.transforms.presets.ssd.load_test(image_names, 512)
    for img, org_img in zip(x, orig_img):
        cid, score, bbox = net(img)
        cid, score, bbox = slice_frcnn_list(cid, score, bbox, 0.20)
        # ax = viz.plot_bbox(org_img, bbox[0], score[0], cid[0], class_names=classes)
        formatted_ssd.append(format_output(cid, score, bbox))

    return formatted_ssd


def get_frcnn_detections(net, image_names):
    """ Feed data to SSD model and get its detections """
    print("total images:", len(image_names))
    i = 0
    formatted_frcnn = []
    ctx = set_context_to_gpu()
    net.hybridize(static_alloc=True)
    net.collect_params().reset_ctx(ctx)
    x, orig_img = gcv.data.transforms.presets.rcnn.load_test(image_names, 512)
    print("done loading")

    for img, org_img in zip(x, orig_img):
        if i % 50 == 0:
            print("Image No.: %d out of %d images(Faster R-CNN)"% (i, len(image_names)))
        img = img.copyto(mx.gpu())
        cid, score, bbox = net(img)
        cid, score, bbox = slice_frcnn_list(cid, score, bbox, 0.20)
        # ax = viz.plot_bbox(org_img, bbox[0], score[0], cid[0], class_names=classes)
        formatted_frcnn.append(format_output(cid, score, bbox))
        i+=1

    np.save('../ensemble_scripts/dets_numpy/test/frcnn/formatted_frcnn' + BATCH_NO + '.npy', formatted_frcnn)
    formatted_frcnn_loaded = np.load('../ensemble_scripts/dets_numpy/test/frcnn/formatted_frcnn' + BATCH_NO + '.npy')

    print(np.array_equal(formatted_frcnn, formatted_frcnn_loaded))

    return formatted_frcnn


def get_ssd_expert_detections(net, image_names):
    """ Feed data to SSD model and get its detections """
    i = 0
    print("inside ssd")

    formatted_ssd_expert = []
    ctx = set_context_to_gpu()
    net.hybridize(static_alloc=True)
    net.collect_params().reset_ctx(ctx)

    x, orig_img = gcv.data.transforms.presets.ssd.load_test(image_names, 512)
    print("done loading")
    for img, org_img in zip(x, orig_img):
        if (i % 50 == 0):
            print("Image No.: %d out of %d images(SSD-Subset)"% (i, len(image_names)))
        img = img.copyto(mx.gpu())
        cid, score, bbox = net(img)
        cid, score, bbox = slice_frcnn_list(cid, score, bbox, 0.20)
        # ax = viz.plot_bbox(org_img, bbox[0], score[0], cid[0], class_names=filter_classes)
        formatted_ssd_expert.append(format_output(cid, score, bbox))
        i += 1
    np.save('../ensemble_scripts/dets_numpy/test/ssd_expert/formatted_ssd_expert' + BATCH_NO + '.npy', formatted_ssd_expert)
    formatted_ssd_expert_loaded = np.load('../ensemble_scripts/dets_numpy/test/ssd_expert/formatted_ssd_expert' + BATCH_NO + '.npy')

    print(np.array_equal(formatted_ssd_expert, formatted_ssd_expert_loaded))

    return formatted_ssd_expert


def set_context_to_gpu():
    """ set context to gpu """

    try:
        a = mx.nd.zeros((1,), ctx=mx.gpu(0))
        ctx = [mx.gpu(0)]
    except:
        ctx = [mx.cpu()]

    return ctx


def plot_bbox(img, bbox, score, cid, classes, model):
    """ plots bounding boxes on the given image """
    print("Plotting")
    ax = viz.plot_bbox(img, bbox, score, cid, thresh=0.5, class_names=classes)
    plt.suptitle(model)
    plt.show()


def convert_to_plot_box_format(lst):
    """ reformats detection list to send it to plotting function """

    bbox = []
    scores = []
    cids = []
    retina, _, _ = get_class_labels()
    inv_map = {v: k for k, v in retina.items()}

    for item in lst:
        box = []
        box.append(float(item[3]))
        box.append(float(item[4]))
        box.append(float(item[5]))
        box.append(float(item[6]))
        score = float(item[2])
        cid = inv_map[item[1]]
        bbox.append(box)
        scores.append(score)
        cids.append(cid)

    return bbox, cids, scores


def format_output(cid, score, bbox):
    """ format output by ssd/frcnn model in ensemble input format """
    formatted_detections = []

    # print("Formatting bounding boxes for ensembling")
    if type(cid[0]) is not np.ndarray:
        cid_t = cid[0].asnumpy()
        score_t = score[0].asnumpy()
        bbox_t = bbox[0].asnumpy()
    else:
        bbox_t = bbox[0]
        cid_t = cid[0]
        score_t = score[0]

    temp = [(np.hstack((bbox_t[b], cid_t[b].ravel(), score_t[b].ravel()))).tolist() for b in range(len(bbox_t))]
    formatted_detections.append(temp)
    # print("Completed formatting of bounding boxes")
    return formatted_detections[0]


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


def add_image_names_to_detections(formatted_op_frcnn, formatted_op_ssd_expert, ret_detection, names):
    """ Adding image names to the detection result lists """
    # [j.insert(len(j), name.strip()) for j in formatted_op_ssd]
    for i in range(0, len(formatted_op_frcnn)):
        [j.insert(len(j), names[i].strip()) for j in formatted_op_frcnn[i]]
        [j.insert(len(j), names[i].strip()) for j in formatted_op_ssd_expert[i]]
        [j.insert(len(j), names[i].strip()) for j in ret_detection[i]]

    return formatted_op_frcnn, formatted_op_ssd_expert, ret_detection


def format_list_for_ensemble(formatted_op_frcnn, formatted_op_ssd_expert, ret_detection):
    """ creating a new list with detections from all models to feed into the model ensemble method """

    input_ensemble = []

    for item in range(0, len(formatted_op_frcnn)):
        temp = []
        # temp_ssd = formatted_op_ssd
        temp_frcnn = formatted_op_frcnn[item]
        temp_expert = formatted_op_ssd_expert[item]
        temp_retina = ret_detection[item]
        # temp.append(temp_ssd)
        temp.append(temp_frcnn)
        temp.append(temp_expert)
        temp.append(temp_retina)
        input_ensemble.append(temp)

    return input_ensemble


def eval_format(ens_detections, permutations):
    """ rearrange columns of ens-detections according to permutations """

    new = [[r[x] for x in permutations] for r in ens_detections]
    return new


def keep_ensembled_confident(det):
    """ keep ensembled detections which have confidence score > 0.5 """

    conf = []
    for item in det:
        if item[2] > 0.5:
            conf.append(item)
    return conf


if __name__ == "__main__":

    print("Starting to predict using ensemble model.")
    retina_labels_to_names, classes, filter_classes = get_class_labels()
    image_names = [
                    '../sample_data/00000257.jpg',
                    '../sample_data/00006992.jpg',
                    '../sample_data/00110662.jpg'
                    ]
    image_names = [line.strip() for line in open(TEST_PATH)]

    models_gcv = get_model_dicts()
    names = []
    # read_image = mpimg.imread(image)
    for img in image_names:
        name = ((img.split('/')[-1]).split('.')[0]).lstrip("0")
        if name == "":
            name = "0"
        names.append(name)
    # getting detections from SSD and FRCNN
    for dicts in models_gcv.items():
        print("STARTED DETECTIONS FOR " + dicts[0] + "")
        dict_n = dict(dicts[1])

        if dicts[0] == 'SSDs':
            net = get_model(dict_n['model'], dict_n['weights'], classes)
            formatted_ssd = get_ssd_detections(net, image_names)

        elif dicts[0] == 'FASTER-RCNN':
            net = get_model(dict_n['model'], dict_n['weights'], classes)
            formatted_frcnn = get_frcnn_detections(net, image_names)

        elif dicts[0] == 'SSD-EXPERT':
            net = get_model(dict_n['model'], dict_n['weights'], filter_classes)
            formatted_ssd_expert = get_ssd_expert_detections(net, image_names)

    from predict_retinanet import *
    ret_detection, draw = predict(image_names)

    np.save('../ensemble_scripts/dets_numpy/test/retina/ret_detection' + BATCH_NO + '.npy', ret_detection)
    ret_detection_loaded = np.load('../ensemble_scripts/dets_numpy/test/retina/ret_detection' + BATCH_NO + '.npy')

    print(np.array_equal(ret_detection, ret_detection_loaded))

    print("Adding image names to the result list")
    formatted_frcnn, formatted_ssd_expert, ret_detection = add_image_names_to_detections(
        formatted_frcnn, formatted_ssd_expert, ret_detection, names)

    print("Replacing integer class ids with class names")
    # formatted_ssd = replace_id_with_name(formatted_ssd, False)
    formatted_frcnn = replace_id_with_name(formatted_frcnn, False)
    formatted_ssd_expert = replace_id_with_name(formatted_ssd_expert, True)
    retina = replace_id_with_name(ret_detection, False)

    print("Formatting list to feed into the model ensemble method")
    input_ensemble = format_list_for_ensemble(formatted_frcnn, formatted_ssd_expert, ret_detection)

    np.save('../ensemble_scripts/dets_numpy/test/input_ensembles/input_ensembles' + BATCH_NO + '.npy', input_ensemble)

    final_predictions = []
    print("Starting model ensembling")
    for item in input_ensemble:
        ens = GeneralEnsemble(item, weights=[0.8, 0.1, 0.1])
        permutations = [6, 4, 5, 0, 1, 2, 3]
        list_ens = eval_format(np.asarray(ens), permutations)
        # list_ens = [[((im_fname.split('/')[-1]).split('.')[0]).lstrip("0")] + x for x in list_ens]
        final_predictions.extend(keep_ensembled_confident(list_ens))
    print("Finished ensembling.")
    print("Final detections given by ensemble model:")

    print("Writing ensembled results to csv")
    with open("../ensemble_scripts/dets_numpy/test/results/result_ensemble" + BATCH_NO + ".csv", 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(final_predictions)

    print(len(final_predictions))
    exit()


    bbox, cid, score = convert_to_plot_box_format(final_predictions)
    x, image = gcv.data.transforms.presets.ssd.load_test(image, 512)

    plot_bbox(image, mx.nd.array(bbox), mx.nd.array(score), mx.nd.array(cid), classes, "Ensemble")
