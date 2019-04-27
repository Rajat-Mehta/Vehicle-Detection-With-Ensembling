from gluoncv import model_zoo
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet import autograd, gluon
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
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
import fnmatch

"""
Predict the detections for the input (validation) images from all of the trained models
SSD, SSD-Subset and FRCNN and pass them to GeneralEnsemble method of ensemble.py
for getting the final ensemble detections.
"""
TEST = True
VAL_PATH = "../data_set_files/valid_mini.txt"
TEST_PATH = "../data_set_files/record_format_files/data_set_test/test_mini.txt"
validation_dataset = gcv.data.RecordFileDetection('../data_set_files/record_format_files/data-set_min/val.rec',
                                           coord_normalized=True)
test_dataset = gcv.data.RecordFileDetection('../data_set_files/record_format_files/data_set_test/test_min.rec',
                                            coord_normalized=True)
if TEST:
    val_dataset = test_dataset
    VAL_PATH = TEST_PATH

BATCH_SIZE = 1
BATCH_NO = '_0-1500'


def get_ssd_data_loader(val_dataset):
    """ load data in batches for ssd model """

    batch_size = BATCH_SIZE
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(512, 512)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=0)
    return val_loader


def get_frcnn_data_loader(val_dataset, net):
    """ load data in batches for frcnn model """

    batch_size = BATCH_SIZE
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(FasterRCNNDefaultValTransform(net.short, net.max_size)),
        batch_size, False, batchify_fn=val_bfn, last_batch='keep', num_workers=0)
    return val_loader


def preprocess_image_retina(im_fname):
    """ preprocess image: will scale pixels between -1 and 1, sample-wise."""

    # load image
    image = read_image_bgr(im_fname)
    # preprocess image for network
    image = preprocess_image(image)
    return image


def get_detections(net, x):
    """ bounding box detections using SSD """

    cid, score, bbox = net(x)
    return cid, score, bbox


def get_model(model, weights, classes):
    """ get model from gluoncv """

    net = model_zoo.get_model(model, pretrained=True)
    if weights is not 'NONE':
        net.reset_class(classes)
        net.load_parameters(weights)
    return net


def keep_confident_detections_batch(cid, score, bbox):
    """ keep confident detections and ignore others (for batch)"""

    cid_n = []
    score_n = []
    bbox_n = []
    i = 0

    for batch_c, batch_s, batch_b in zip(cid, score, bbox):
        if (i % 50 == 0) & (i > 0):
            print("Processed %d/%d batches" % (i, len(cid)))
        for c, s, b in zip(batch_c, batch_s, batch_b):
            scores = np.where(s > 0.5)
            if not len(scores[0]) == 0:
                cid_n.append(c[scores[0]])
                score_n.append(s[scores[0]])
                bbox_n.append(b[scores[0]])
            else:
                cid_n.append(np.asarray([0]))
                score_n.append(np.asarray([0]))
                bbox_n.append(np.asarray([[0, 0, 0, 0]]))
        i += 1
    return cid_n, score_n, bbox_n


def slice_frcnn_list(cid, score, bbox, thresh):
    """ ignore detections where score < thresh """

    cid_new = []
    score_new = []
    bbox_new = []
    i = 0

    for c, s, b in zip(cid, score, bbox):
        if (i % 50 == 0) & (i > 0):
            print("Processed %d/%d batches" % (i, len(cid)))
        index = non_zero(s, thresh)
        if len(index) is not 0:
            cid_new.append(c[index])
            score_new.append(s[index])
            bbox_new.append(b[index])
        else:
            cid_new.append(np.asarray([0]))
            score_new.append(np.asarray([0]))
            bbox_new.append(np.asarray([[0, 0, 0, 0]]))
        i += 1

    return cid_new, score_new, bbox_new


def slice_frcnn_list_batch(cid, score, bbox):
    """ ignore detections where score < thresh """

    print("Slicing detection outputs.")
    cid_new_b = []
    score_new_b = []
    bbox_new_b = []
    for batch_c, batch_s, batch_b in zip(cid, score, bbox):
        cid_new = []
        score_new = []
        bbox_new = []
        for c, s, b in zip(batch_c, batch_s, batch_b):
            cid_new.append(c[0:50])
            score_new.append(s[0:50])
            bbox_new.append(b[0:50])
        cid_new_b.append(cid_new)
        score_new_b.append(bbox_new)
        bbox_new_b.append(cid_new)
    print("Slicing finished")

    return cid_new_b, score_new_b, bbox_new_b


def keep_confident_detections(cid, score, bbox):
    """ keep confident detections and ignore others """

    """ keep only confident detections with score > 0.5 and remove others """
    cid_n = []
    score_n = []
    bbox_n = []
    i=0

    for c, s, b in zip(cid, score, bbox):
        scores = np.where(s > 0.5)
        if (i % 50 == 0) & (i > 0):
            print("Processed %d/%d images"% (i, len(cid)))
        if not len(scores[0]) == 0:
            cid_n.append(c[scores[0]])
            score_n.append(s[scores[0]])
            bbox_n.append(b[scores[0]])
        else:
            cid_n.append(np.asarray([0]))
            score_n.append(np.asarray([0]))
            bbox_n.append(np.asarray([[0, 0, 0, 0]]))
        i+=1

    return cid_n, score_n, bbox_n


def keep_ensembled_confident(det):
    """ keep ensembled detections which have confidence score > 0.5 """

    conf = []
    for item in det:
        if item[2] > 0.5:
            conf.append(item)

    return conf


def format_retina_output(cid, score, bbox):
    """ format retina model detections the same way as ssd and frcnn """

    formatted_detections = []
    # cid, score, bbox = keep_confident_detections(cid, score, bbox)

    for item in range(0, len(cid[0])):

        bbox_t = np.ravel(np.asarray(bbox[0][item]))
        score_t = np.ravel(np.asarray([score[0][item]]))
        cid_t = np.ravel(np.asarray([cid[0][item]]))

        temp = np.hstack((bbox_t, cid_t[0], score_t[0])).ravel()
        formatted_detections.append(temp)

    return formatted_detections


def format_batch_output(cid, score, bbox):
    """ format output by ssd/frcnn model in ensemble input format (for batch) """

    formatted_detections = []
    # print("Removing unconfident batch detections")
    # print(len(cid))
    # cid, score, bbox = keep_confident_detections_batch(cid, score, bbox)
    # print("Removed unconfident batch detections")

    print("Formatting bounding boxes for ensembling")
    i = 0

    for item in range(0, len(cid)):
        if (i % 50 == 0) and (i > 0):
            print("Formatted %d/%d image detections" % (i, len(cid)))
        if type(cid[item]) is not np.ndarray:
            bbox_t = bbox[item].asnumpy()
            cid_t = cid[item].asnumpy()
            score_t = score[item].asnumpy()
        else:
            bbox_t = bbox[item]
            cid_t = cid[item]
            score_t = score[item]
        temp = [(np.hstack((bbox_t[b], cid_t[b].ravel(), score_t[b].ravel()))).tolist() for b in range(len(bbox_t))]
        formatted_detections.append(temp)
        i += 1
    print("Completed formatting of bounding boxes")

    return formatted_detections


def format_output(cid, score, bbox):
    """ format output by ssd/frcnn model in ensemble input format """
    formatted_detections = []

    # print("Removing unconfident detections")
    # cid,score,bbox = keep_confident_detections(cid, score, bbox)
    # print("Removed unconfident detections")

    print("Formatting bounding boxes for ensembling")
    i = 0
    for item in range(0, len(cid)):
        if (i % 50 == 0) & (i > 0):
            print("Formatted %d/%d image detections" % (i, len(cid)))
        if type(cid[item]) is not np.ndarray:
            cid_t = cid[item].asnumpy()
            score_t = score[item].asnumpy()
            bbox_t = bbox[item].asnumpy()

        else:
            bbox_t = bbox[item]
            cid_t = cid[item]
            score_t = score[item]

        temp = [(np.hstack((bbox_t[b], cid_t[b].ravel(), score_t[b].ravel()))).tolist() for b in range(len(bbox_t))]
        formatted_detections.append(temp)
        i+=1
    print("Completed formatting of bounding boxes")

    return formatted_detections


def format_ensemble_output(ens_detections):
    """ formatting the output given by ensemble function according to result format """

    ens_detections = np.asarray(ens_detections)
    bbox =[]
    cid=[]
    score=[]
    bbox.append(ens_detections[:, :4])
    cid.append(ens_detections[:, 4])
    score.append(ens_detections[:, 5])

    return np.asarray(bbox), np.asarray(score), np.asarray(cid)


def eval_format(ens_detections, permutations):
    """ rearrange columns of ens-detections according to permutations """

    new = [[r[x] for x in permutations] for r in ens_detections]

    return new


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device """

    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)

    return new_batch


def validate_frcnn(net, val_data, ctx):
    """ validate faster r-cnn model on validation set """

    clipper = gcv.nn.bbox.BBoxClipToImage()
    net.hybridize(static_alloc=True)
    net.collect_params().reset_ctx(ctx)
    det_bboxes = []
    det_ids = []
    i = 0
    det_scores = []
    for batch in val_data:
        if (i % 500 == 0) & (i > 0):
            print("Batch No.: %d out of %d batches"% (i, len(val_data)))
        # batch = split_and_load(batch, ctx_list=ctx)

        for x, y, im_scale in zip(*batch):
            # get prediction results
            x = x.copyto(mx.gpu())
            ids, scores, bboxes = net(x)
            det_ids.append(np.squeeze(ids))
            det_scores.append(np.squeeze(scores))
            # clip to image size
            det_bboxes.append(np.squeeze(clipper(bboxes, x)))
            # rescale to original resolution
            im_scale = im_scale.reshape((-1)).asscalar()
            det_bboxes[-1] *= im_scale
        i += 1

    return det_ids, det_scores, det_bboxes


def validate(net, val_data, ctx):
    """ Test on validation dataset """

    i = 1
    det_bboxes = []
    det_ids = []
    det_scores = []

    net.collect_params().reset_ctx(ctx)
    net.hybridize(static_alloc=True, static_shape=True)

    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize()

    for batch in val_data:
        if (i % 500 == 0) & (i > 0):
            print("Batch No.: %d out of %d batches"% (i, len(val_data)))
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        # label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

        # get prediction results
        ids, scores, bboxes = net(data[0])
        det_ids.append(np.squeeze(ids))
        det_scores.append(np.squeeze(scores))
        det_bboxes.append(np.squeeze(bboxes.clip(0, batch[0].shape[2])))

        i += 1

    return det_ids, det_scores, det_bboxes


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
    model_retina_weights = '../models_for_ensemble/retina_weights.70-0.76.hdf5'
    model_retina = models.load_model(model_retina_weights, backbone_name='resnet50')
    model_retina = models.convert_model(model_retina)

    return models_gcv, model_retina


def read_retina_images(path):
    """ read validation images from the path provided """

    file_obj = open(path, "r")
    images = []
    image_names = []
    num_lines = sum(1 for line in open(path))

    print("Reading images")
    i = 0
    for item in file_obj:
        item = item.strip()
        if item:
            if (i % 500 == 0) & (i > 0) :
                print("Read %d/%d images"% (i, num_lines))
            image_names.append(item)

            """
            image = read_image_bgr(item)
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
            """

            image = preprocess_image_retina(item)
            images.append(image)
        i += 1
    print("Finished reading images")

    return images, image_names


def convert_list_array_to_list_list(list):
    """ convert a list of arrays to list of list """

    new_list = [l.tolist() for l in list]

    return new_list


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


def string_arr_to_int(lst):
    """ converts string array to int array """

    columns = zip(lst)
    cols = [0, 2, 3, 4, 5, 6]
    new = [[int(n) for n in columns[col]] for col in cols]

    return new


def set_context_to_gpu():
    """ set context to gpu """

    try:
        a = mx.nd.zeros((1,), ctx=mx.gpu(0))
        ctx = [mx.gpu(0)]
    except:
        ctx = [mx.cpu()]

    return ctx


def get_ssd_detections(net):
    """ Feed data to SSD model and get its detections """

    val_data = get_ssd_data_loader(val_dataset)
    # set context to gpu
    ctx = set_context_to_gpu()

    cid, score, bbox = validate(net, val_data, ctx)
    if BATCH_SIZE > 1:
        cid, score, bbox = slice_frcnn_list_batch(cid, score, bbox)
        formatted_op_ssd = format_batch_output(cid, score, bbox)
    else:
        print("Ignoring low confidence detections.")
        cid, score, bbox = slice_frcnn_list(cid, score, bbox, 0.30)
        print("Removed low confidence detections")

        formatted_op_ssd = format_output(cid, score, bbox)

    with open('../data_set_files/image_shapes', 'rb') as fp:
        shapes = pickle.load(fp)

    # formatted_op_ssd = rescale_detections(formatted_op_ssd, shapes)

    return formatted_op_ssd


def filter_coco_classes(cid):
    """ filter coco detections with mio-tcd classes """
    coco = {0: 'pedestrian', 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus"}
    mio, _, _ = get_class_labels()
    filtered_coco = [[j for j, k in enumerate(e) if k in coco.keys()] for e in cid]

    return filtered_coco


def coco_to_mio_class_map(cid):
    """ map coco detections with mio-tcd classes """

    coco = {0: 'pedestrian', 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus"}
    mio, _, _ = get_class_labels()
    inv_map = {v: k for k, v in mio.items()}
    new_cid = []
    for cls in cid:
        if type(cls) is not np.ndarray:
            cls = cls.asnumpy()
        new_c = []
        for c in cls:
            val = coco[c]
            new_id = inv_map[val]
            new_c.append(new_id)
        new_cid.append(mx.nd.array(new_c, mx.gpu(0)))

    return new_cid


def coco_to_mio(cid, score, bbox, index):
    """ map class ids of ssd-coco to the classes in mio dataset """
    i = 0
    cid_new = []
    score_new = []
    bbox_new = []
    for c, s, b in zip(cid, score, bbox):
        # if (i % 50 == 0) & (i > 0):
        #    print("Mapping coco to mio %d/%d batches" % (i, len(cid)))
        if len(index[i]) is not 0:
            cid_new.append(c[index[i]])
            score_new.append(mx.nd.array(s[index[i]], mx.gpu(0)))
            bbox_new.append(mx.nd.array(b[index[i]], mx.gpu(0)))
        else:
            cid_new.append(mx.nd.array([2], mx.gpu(0)))
            score_new.append(mx.nd.array([0], mx.gpu(0)))
            bbox_new.append(mx.nd.array([[0, 0, 0, 0]], mx.gpu(0)))
        i += 1
    cid_new = coco_to_mio_class_map(cid_new)

    return cid_new, score_new, bbox_new


def get_ssd_coco_detections(net):
    """ Feed data to SSD model and get formatted detections """

    val_data = get_ssd_data_loader(val_dataset)
    # set context to gpu
    ctx = set_context_to_gpu()

    cid, score, bbox = validate(net, val_data, ctx)

    if BATCH_SIZE > 1:
        cid, score, bbox = slice_frcnn_list_batch(cid, score, bbox)
        formatted_op_ssd_coco = format_batch_output(cid, score, bbox)
    else:
        print("Ignoring low confidence detections.")
        cid, score, bbox = slice_frcnn_list(cid, score, bbox, 0.30)
        index = filter_coco_classes(cid)
        cid, score, bbox = coco_to_mio(cid, score, bbox, index)

        print("Removed low confidence detections")
        formatted_op_ssd_coco = format_output(cid, score, bbox)
    # this rescaling will only work if fed with complete val set i.e. 20,000 images
    # with open('image_shapes', 'rb') as fp:
    #    shapes = pickle.load(fp)

    # formatted_op_ssd = rescale_detections(formatted_op_ssd, shapes)
    print(len(formatted_op_ssd_coco))
    np.save('./dets_numpy/formatted_op_ssd_coco'+BATCH_NO+'.npy', formatted_op_ssd_coco)
    formatted_op_ssd_coco_loaded = np.load('./dets_numpy/formatted_op_ssd_coco'+BATCH_NO+'.npy')

    print(np.array_equal(formatted_op_ssd_coco, formatted_op_ssd_coco_loaded))

    return formatted_op_ssd_coco


def non_zero(lst, thresh):
    """ return indexes of items which are not -1 and value is greater than thresh """

    return [i for i, e in enumerate(lst) if e > thresh]


def get_frcnn_detections(net):
    """ Feed data to Faster R-CNN model and get its detections """

    val_data = get_frcnn_data_loader(val_dataset, net)
    ctx = set_context_to_gpu()
    cid, score, bbox = validate_frcnn(net, val_data, ctx)

    print("Ignoring low confidence detections.")
    cid, score, bbox = slice_frcnn_list(cid, score, bbox, 0.40)
    print("Removed low confidence detections")
    formatted_op_frcnn = format_output(cid, score, bbox)
    if TEST:
        np.save('./dets_numpy/test/formatted_op_frcnn'+BATCH_NO+'.npy', formatted_op_frcnn)
        formatted_op_frcnn_loaded = np.load('./dets_numpy/test/formatted_op_frcnn'+BATCH_NO+'.npy')
    else:
        np.save('./dets_numpy/formatted_op_frcnn' + BATCH_NO + '.npy', formatted_op_frcnn)
        formatted_op_frcnn_loaded = np.load('./dets_numpy/formatted_op_frcnn' + BATCH_NO + '.npy')

    print(np.array_equal(formatted_op_frcnn, formatted_op_frcnn_loaded))

    return formatted_op_frcnn


def get_retina_detections(model_retina, processed_images):
    """ Feed data to Retina Net model and get its detections """

    ret_detection = []
    j = 0
    for image in processed_images:
        if (j % 20 == 0) and j > 0:
            print("Retina detections: Processed %d images" % j)
        image, scale = resize_image(image)
        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))
        boxes, scores, labels = model_retina.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        labels, scores, boxes = slice_frcnn_list(labels, scores, boxes, 0.35)
        labels = np.asarray(labels)
        scores = np.asarray(scores)
        boxes = np.asarray(boxes)


        ret_detection.append(np.atleast_2d(np.squeeze(format_retina_output(labels, scores, boxes))))
        j += 1
    ret_detection = convert_list_array_to_list_list(ret_detection)

    if TEST:
        np.save('./dets_numpy/test/retina/ret_detection'+BATCH_NO+'.npy', ret_detection)
        ret_detection_loaded= np.load('./dets_numpy/test/retina/ret_detection'+BATCH_NO+'.npy')
    else:
        np.save('./dets_numpy/retina/ret_detection' + BATCH_NO + '.npy', ret_detection)
        ret_detection_loaded = np.load('./dets_numpy/retina/ret_detection'+BATCH_NO+'.npy')

    print(np.array_equal(ret_detection_loaded, ret_detection))

    return ret_detection


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


def add_image_names_to_detections(formatted_op_ssd, formatted_op_frcnn, formatted_op_ssd_expert, ret_detection):
    """ Adding image names to the detection result lists """

    for i in range(0, len(formatted_op_ssd)):
        name = ((image_names[i].split('/')[-1]).split('.')[0]).lstrip("0")
        if name == "":
            name = "0"
        [j.insert(len(j), name.strip()) for j in formatted_op_ssd[i]]
        [j.insert(len(j), name.strip()) for j in formatted_op_frcnn[i]]
        [j.insert(len(j), name.strip()) for j in formatted_op_ssd_expert[i]]
        [j.insert(len(j), name.strip()) for j in ret_detection[i]]

    return formatted_op_ssd, formatted_op_frcnn, formatted_op_ssd_expert, ret_detection


def format_list_for_ensemble(formatted_op_ssd, formatted_op_frcnn, formatted_op_ssd_expert, ret_detection):
    """ creating a new list with detections from all models to feed into the model ensemble method """

    input_ensemble = []
    for item in range(0, len(formatted_op_ssd)):
        temp = []
        temp_ssd = formatted_op_ssd[item]
        temp_frcnn = formatted_op_frcnn[item]
        temp_expert = formatted_op_ssd_expert[item]
        temp_retina = ret_detection[item]
        temp.append(temp_ssd)
        temp.append(temp_frcnn)
        temp.append(temp_expert)
        temp.append(temp_retina)
        input_ensemble.append(temp)

    return input_ensemble


def rescale_detections(ssd, shapes):
    """ don't run this function in batches (of 1500), run it on full dataset so that len(dataset) == len(shapes)"""
    """ rescales the given detections back according to original width and height """

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


if __name__ == "__main__":
    data_path = TEST_PATH
    print("Start of prediction and ensemble process.")
    processed_images, imgage_names = read_retina_images(data_path)
    print(len(processed_images), len(imgage_names))

    retina_labels_to_names, classes, filter_classes = get_class_labels()

    # get models
    models_gcv, model_retina = get_model_dicts()

    # getting detections from SSD,SSD-Subset and FRCNN
    for dicts in models_gcv.items():
        print("STARTED DETECTIONS FOR "+dicts[0]+"")
        dict_n = dict(dicts[1])

        if dicts[0] == 'SSDs':
            net = get_model(dict_n['model'], dict_n['weights'], classes)
            formatted_op_ssd = get_ssd_detections(net)

            if TEST:
                np.save('./dets_numpy/test/formatted_op_ssd' + BATCH_NO + '.npy', formatted_op_ssd)
                formatted_op_ssd_loaded = np.load(
                    './dets_numpy/test/formatted_op_ssd' + BATCH_NO + '.npy')
            else:
                np.save('./dets_numpy/formatted_op_ssd' + BATCH_NO + '.npy', formatted_op_ssd)
                formatted_op_ssd_loaded = np.load(
                    './dets_numpy/formatted_op_ssd' + BATCH_NO + '.npy')

            print(np.array_equal(formatted_op_ssd, formatted_op_ssd_loaded))

        elif dicts[0] == 'FASTER-RCNN':
            net = get_model(dict_n['model'], dict_n['weights'], classes)
            formatted_op_frcnn = get_frcnn_detections(net)

        elif dicts[0] == 'SSD-EXPERT':
            net = get_model(dict_n['model'], dict_n['weights'], filter_classes)
            formatted_op_ssd_expert = get_ssd_detections(net)

            if TEST:
                np.save('./dets_numpy/test/formatted_op_ssd_expert' + BATCH_NO + '.npy', formatted_op_ssd_expert)
                formatted_op_ssd_expert_loaded = np.load(
                    './dets_numpy/test/formatted_op_ssd_expert' + BATCH_NO + '.npy')
            else:
                np.save('./dets_numpy/formatted_op_ssd_expert' + BATCH_NO + '.npy', formatted_op_ssd_expert)
                formatted_op_ssd_expert_loaded = np.load(
                    './dets_numpy/formatted_op_ssd_expert' + BATCH_NO + '.npy')

            print(np.array_equal(formatted_op_ssd_expert, formatted_op_ssd_expert_loaded))

    print("Finished SSD and FRCNN detections")
    """
    Uncomment the code lines inside "read_retina_images" function before enabling retina net 
    """
    print("Started detections for Retina Net")
    ret_detection = get_retina_detections(model_retina, processed_images)
    print("Finished retina detections")
    # retina=[]

    print(len(formatted_op_ssd))
    print(len(formatted_op_frcnn))
    print(len(formatted_op_ssd_expert))
    print(len(ret_detection))

    # print(len(retina))
    print("Adding image names to the result list")
    formatted_op_ssd, formatted_op_frcnn, formatted_op_ssd_expert, ret_detection = add_image_names_to_detections(
        formatted_op_ssd, formatted_op_frcnn, formatted_op_ssd_expert, ret_detection)

    print("Replacing integer class ids with class names")
    formatted_op_ssd = replace_id_with_name(formatted_op_ssd, False)
    formatted_op_frcnn = replace_id_with_name(formatted_op_frcnn, False)
    formatted_op_ssd_expert = replace_id_with_name(formatted_op_ssd_expert, False)
    retina = replace_id_with_name(ret_detection)

    print("Formatting list to feed into the model ensemble method")
    input_ensemble = format_list_for_ensemble(formatted_op_ssd, formatted_op_frcnn, formatted_op_ssd_expert, ret_detection)

    with open('./dets_numpy/input_ensemble'+BATCH_NO + '.npy', 'wb') as f:
        pickle.dump(input_ensemble, f)

    # with open('saved_ind_results/input_ensembles/input_ensemble_4', 'rb') as f:
    #    my_list = pickle.load(f)

    final_predictions =[]
    print("Starting model ensembling")
    for item in input_ensemble:
        ens = GeneralEnsemble(item, weights = [0.16, 0.84, 0.35])
        permutations = [6, 4, 5, 0, 1, 2, 3]
        list_ens = eval_format(np.asarray(ens), permutations)
        # list_ens = [[((im_fname.split('/')[-1]).split('.')[0]).lstrip("0")] + x for x in list_ens]
        final_predictions.extend(keep_ensembled_confident(list_ens))
    print("Finished ensembling.")
    print("Final detections given by ensemble model:")

    print("Writing ensembled results to csv")
    with open("./dets_numpy/result_ensemble" + BATCH_NO + ".csv", 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(final_predictions)

    print("Finished.")
    exit(1)
