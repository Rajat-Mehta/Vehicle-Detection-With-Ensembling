# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet import models
# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
from keras_retinanet.models import load_model

# load retinanet model
model = models.load_model('../models_for_ensemble/retina_weights.70-0.76.hdf5', backbone_name='resnet50')
model = models.convert_model(model)


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


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


def convert_list_array_to_list_list(list):
    """ convert a list of arrays to list of list """

    new_list = [l.tolist() for l in list]

    return new_list


def non_zero(lst, thresh):
    """ return indexes of items which are not -1 and value is greater than thresh """

    return [i for i, e in enumerate(lst) if e > thresh]


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
            bbox_new.append(np.asarray([[0.0, 0.0, 0.0, 0.0]]))
        i += 1

    return cid_new, score_new, bbox_new


def predict(image_list):
    ret_detection = []
    draws = []
    x = 0

    for img in image_list:
        # load image
        if (x % 50 == 0):
            print("Image No.: %d out of %d images(RetinaNet)"% (x, len(image_list)))
        image = read_image_bgr(img)

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        # print("processing time: ", time.time() - start)

        labels, scores, boxes = slice_frcnn_list(labels, scores, boxes, 0.35)

        # correct for image scale

        boxes[0] /= scale

        new = []
        for item in boxes:
            i = item.astype(int)
            new.append(i)
        labels = np.asarray(labels)
        scores = np.asarray(scores)
        boxes = np.asarray(boxes)

        # plot_pred(boxes, scores, labels, draw)

        ret_detection.append(np.atleast_2d(np.squeeze(format_retina_output(labels, scores, boxes))))
        draws.append(draw)
        x+=1
    ret_detection = convert_list_array_to_list_list(ret_detection)

    return ret_detection, draws


def plot_pred(boxes, scores, labels, draw):
    labels_to_names = {0: 'car', 1: 'articulated_truck', 2: 'bus', 3: 'bicycle', 4: 'motorcycle',
                       5: 'motorized_vehicle',
                       6: 'pedestrian', 7: 'single_unit_truck', 8: 'work_van', 9: 'pickup_truck',
                       10: 'non-motorized_vehicle'}

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.suptitle("Retina")
    plt.imshow(draw)
    # plt.show()


if __name__=="__main__":

    predict(['../sample_data/00006992.jpg'])
