import numpy as np
"""
Ensembling in object detection.
Input: 
 - dets : List of detections. 
 - iou_thresh: Threshold in terms of IOU where two boxes are considered the same, 
               if they also belong to the same class.
               
 - weights: A list of weights, describing how much more some detectors should
            be trusted compared to others. 
Output:
    A list of boxes, of the same format as the input. Confidences are in range 0-1.
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

    x1 = float(box[0]) - float(box[2])/2
    x2 = float(box[0]) + float(box[2])/2
    y1 = float(box[1]) - float(box[3])/2
    y2 = float(box[1]) + float(box[3])/2
    return x1, x2, y1, y2


def computeIOU(box1, box2):
    """ computes intersection over union of two bounding boxes """

    x11, x12, y11, y12 = getCoords(box1)
    x21, x22, y21, y22 = getCoords(box2)

    x_left   = max(x11, x21)
    y_top    = max(y11, y21)
    x_right  = min(x12, x22)
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


if __name__=="__main__":
    # Toy example
    dets = [
                [
                    [100, 200, 150, 250, 0, 0.5, '0'],
                    [54, 140, 300, 600, 0, 0.6, '0']
                ],
                [
                    [108, 196, 146, 240, 0, 0.8, '0']
                ],
                [
                    [110, 204, 156, 253, 0, 0.9, '0'],
                    [500, 440, 600, 700, 0, 0.5, '0']
                ]
           ]
    ens = GeneralEnsemble(dets, weights=[3.0, 1.0, 2.0])
    print(ens)


