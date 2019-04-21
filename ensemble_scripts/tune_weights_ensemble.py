import os
import csv
import numpy as np
import sys
import subprocess
import ternary

"""
Tuning the ensembling model on different weight value combinations for each participating model

Input: 
 - list of weight values(search space) from where we select the optimal weights
Output:
    results of our ensemble model on each of the weight combination. we select the combination which gives
    the best mAP on (a part of) validation dataset
Parts of this script are referred from https://github.com/ahrnbom/ensemble-objdet
"""


def GeneralEnsemble(dets, iou_thresh=0.5, weights=None):
    """ performs ensemble of multiple models """

    assert(type(iou_thresh) == float)

    ndets = len(dets)
    if weights is None:
        w = 1/float(ndets)
        weights = [w]*ndets
    else:
        assert(len(weights) == ndets)

        s = sum(weights)
        for i in range(0, len(weights)):
            weights[i] /= s

    out = list()
    used = list()

    for idet in range(0, ndets):
        det = dets[idet]
        for box in det:
            if is_row_in_array(box, used):
                continue

            used.append(box)
            # Search the other detectors for overlapping box of same class
            found = []
            for iodet in range(0, ndets):
                odet = dets[iodet]
                if np.array_equal(odet, det):
                    continue

                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not is_row_in_array(obox, used):
                        # Not already used
                        if box[4] == obox[4]:
                            # Same class
                            iou = computeIOU(box, obox)
                            if iou > bestiou:
                                bestiou = iou
                                bestbox = obox

                if not bestbox is None:
                    w = weights[iodet]
                    found.append((bestbox, w))
                    used.append(bestbox)
            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                if new_box[5] < 0.85:
                    new_box[5] /= ndets
                out.append(new_box)
            else:
                allboxes = [(box, weights[idet] * box[5] * box[5])]
                allboxes.extend(found)
                allboxes = normalize_weights(allboxes)
                xc = 0.0
                yc = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0

                wsum = 0.0
                for bb in allboxes:
                    w = bb[1]
                    wsum += w
                    b = bb[0]
                    xc += w*b[0]
                    yc += w*b[1]
                    bw += w*b[2]
                    bh += w*b[3]
                    conf += w*b[5]
                xc /= wsum
                yc /= wsum
                bw /= wsum
                bh /= wsum

                new_box = [xc, yc, bw, bh, box[4], (conf), box[6]]
                out.append(new_box)
    return out


def normalize_weights(allboxes):
    """ normalize weights to make their sum equal to one """

    wt_total = 0.0
    for item in allboxes:
        wt_total += item[1]

    new_allboxes = []
    for item in allboxes:
        wt = item[1]
        box = item[0]
        wt = wt / wt_total
        new_allboxes.append((box, wt))

    return new_allboxes


def is_row_in_array(row, array):
    """ checks if the given row is present in the array or not """

    return any(np.array_equal(x, row) for x in array)


def getCoords(box):
    """ gets the coordinates from the 4 bounding box values """

    x1 = float(box[0]) - float(box[2]) / 2
    x2 = float(box[0]) + float(box[2]) / 2
    y1 = float(box[1]) - float(box[3]) / 2
    y2 = float(box[1]) + float(box[3]) / 2
    return x1, x2, y1, y2


def computeIOU(box1, box2):
    """ computes intersection over union of two bounding boxes """

    x11, x12, y11, y12 = getCoords(box1)
    x21, x22, y21, y22 = getCoords(box2)

    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    if intersect_area == 0:
        return 0.0

    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou


def eval_format_final(ens_detections):
    """ rearrange vector columns """

    permutations = [6, 4, 5, 0, 1, 2, 3]
    new = [[r[x] for x in permutations] for r in ens_detections]
    return new


def keep_ensembled_confident(det):
    """ keep only confident detections """

    conf = []
    for item in det:
        if item[2] > 0.5:
            conf.append(item)
    return conf


def generate_weights_data_new(scale=20):
    d = dict()
    from ternary.helpers import simplex_iterator
    temp =[]
    for (i, j, k) in simplex_iterator(scale):
        temp.append([float(i), float(j), float(k)])

    return temp

if __name__ == "__main__":
    dets = np.load('./dets_numpy/input_ensembles/input_ensembles_0-20000_sfe.npy')

    dets = dets[0:len(dets)/2]

    data = generate_weights_data_new()

    # Toy example
    wt = np.arange(0, 1, 0.07)
    wt1 = np.arange(0, 1, 0.07)
    ndets = len(dets[0])

    if wt is None or wt1 is None:
        w = 1 / float(ndets)
        weights_norm = [w] * ndets
    else:
        weights_norm = []

        for a in wt:
            for b in wt1:
                c = 1 - b
                weight = []
                weight.append(c)
                weight.append(b)
                weight.append(a)
                weights_norm.append(weight)

    weights_norm = []
    weights_norm = data
    print(len(weights_norm))
    for i in range(len(weights_norm)):
        final_predictions = []

        if np.count_nonzero(weights_norm[i]) <= 2:
            final_predictions.append(['0', 'car', '0.0', '0.0', '0.0', '0.0', '0.0'])
            print(final_predictions)
            print("Writing ensembled results to csv")
            with open("./tuning_results/result_ensemble_weight_" + str(i).zfill(3) + ".csv", 'w') as myfile:
                wr = csv.writer(myfile)
                wr.writerows(final_predictions)
            print("Process finished for weight :" + str(i).zfill(3))
            continue
        print("started ensembling for weights : %d" % i)
        for item in dets:
            ens = GeneralEnsemble(item, weights=weights_norm[i])

            list_ens = eval_format_final(np.asarray(ens))

            # list_ens = [[((im_fname.split('/')[-1]).split('.')[0]).lstrip("0")] + x for x in list_ens]
            final_predictions.extend(keep_ensembled_confident(list_ens))

        print("Writing ensembled results to csv")
        with open("./tuning_results/result_ensemble_weight_" + str(i).zfill(3) + ".csv", 'w') as myfile:
            wr = csv.writer(myfile)
            wr.writerows(final_predictions)
        print("Process finished for weight :" + str(i).zfill(3))
    print("results saved in csv files for all weight combinations.")

    path = "./tuning_results"

    print("start evaluating mAP for all result files")
    mAP_scores = []
    j = 0
    for filename in sorted(os.listdir(path)):
        if "result_ensemble_weight" in filename:
            score = subprocess.check_output([sys.executable, '../model_evaluation/localization_evaluation.py',
                                             '../data_set_files/gt_val_0-10k.csv',
                                             '../ensemble_scripts/tuning_results/' + filename])
            score = str(weights_norm[j][:]) + ", " +score.splitlines(True)[-2]
            mAP_scores.append(score)
            j+=1

    print("write mAP results of all weight combinations to txt file")

    with open("./tuning_results/mAP_tuned_weights.txt", 'w') as f:
        for item in mAP_scores:
            f.write("%s\n" % item)
