import os
import cv2
import IPython
import os.path
import numpy as np
from gluoncv.data import LstDetection
import sys
import subprocess
from gluoncv import utils
from gluoncv.data import RecordFileDetection

"""
Generates RecordFileFormat files for SSD and Faster R-CNN model

Input:
    ground truth csv files
Output:
    .rec, .idx and .lst files required by gluon cv models
"""


def normalize(size, box):
    """ normalizes the bounding boxes """

    w = (size[0])
    h = (size[1])
    xmin = (box[0]/float(w))
    ymin = (box[1]/float(h))
    xmax = (box[2]/float(w))
    ymax = (box[3]/float(h))
    return [xmin, ymin, xmax, ymax]


def get_class(cls):
    """ gets class id from class name """

    val = -1
    
    if cls == 'car':
        val = 0
    if cls == 'articulated_truck':
        val = 1
    if cls == 'bus':
        val = 2
    if cls == 'bicycle':
        val = 3
    if cls == 'motorcycle':
        val = 4
    if cls == 'motorized_vehicle':
        val = 5
    if cls == 'pedestrian':
        val = 6
    if cls == 'single_unit_truck':
        val = 7
    if cls == 'work_van':
        val = 8
    if cls == 'pickup_truck':
        val = 9
    if cls == 'non-motorized_vehicle':
        val = 10
    return val


def format_lst(bbox, class_id, shapes, paths):
    """ formats the train and validation lists which will be used for generating rec files """

    print('started formatting')
    print(len(class_id))
    counter_formatting = 0
    formatted_lst_train = []
    formatted_lst_val = []
    count_val =0
    count_train = 0
    formatted_box = format_bbox(bbox, class_id)
    for i in range(len(class_id)):

        h, w, c = shapes[i]
        A = 4
        B = 5
        C = w
        D = h
        if "val" in paths[i]:
            str_idx = [str(count_val)]
            count_val += 1
        else:
            str_idx = [str(count_train)]
            count_train += 1

        str_header = [str(x) for x in [A, B, C, D]]
        str_bbox = [str(x) for x in formatted_box[i]]
        str_path = [paths[i]]
        line = '\t'.join(str_idx + str_header + str_bbox + str_path) + '\n'
        if "val" in paths[i]:
            formatted_lst_val.append(line)
        else:
            formatted_lst_train.append(line)
        if counter_formatting % 1000 == 0:
            print("Formatting counter:", counter_formatting)
        counter_formatting += 1
    print('finished formatting')
    return formatted_lst_train, formatted_lst_val


def format_bbox(bbox, class_id):
    """ formats the bounding box """

    new_arr = []
    for i in range(len(bbox)):
        box = []
        for j in range(len(bbox[i])):
            box.append(class_id[i][j])
            box = box + bbox[i][j]
        new_arr.append(box)
    return new_arr


def do_it():
    """ reads ground truth csv's and generates lst files which are used for generating rec files """

    with open('../data_set_files/gt_train_min.csv', 'r') as f:
        l = f.readlines()
    i=0
    reading_counter = 0
    print('started reading')
    bbox = []
    class_id = []
    shape_img = []
    path_img = []
    print("Total objects: ",len(l))
    while i < len(l):
        path = ''
        pp = ''

        ss = l[i].split(',')

        # IPython.embed()
        if os.path.isfile('../data_set_files/train/{:08d}.jpg'.format(int(ss[0]))):
            pp = '../data_set_files/train'
            path = '../data_set_files/train/{:08d}.jpg'.format(int(ss[0]))

        if os.path.isfile('../data_set_files/valid/{:08d}.jpg'.format(int((ss[0])))):
            pp = '../data_set_files/valid'
            path = '../data_set_files/valid/{:08d}.jpg'.format(int((ss[0])))

        # IPython.embed()
        if len(pp) > 2:
            # get image dims
            im = cv2.imread('{}/{:08d}.jpg'.format(pp,int(ss[0])))
            if im is not None:
                sh = im.shape
            # height, width, channel = np.shape(img)

                sz = (sh[1], sh[0])

                mm = ss
                box_class = []
                boxes = []
                shape_img.append(sh)
                path_img.append(os.getcwd()+'/'+path)

                while mm[0] == ss[0]:
                    z = [int(mm[2]), int(mm[3]), int(mm[4]), int(mm[5].rstrip())]
                    b = normalize(sz, z)
                    cls = get_class(mm[1])
                    box_class.append(cls)
                    boxes.append(b)
                    if i+1 < len(l):
                        mm = l[i+1].split(',')
                        i = i+1
                    else:
                        i = i + 1
                        break

                bbox.append(boxes)
                class_id.append(box_class)
            else:
                print('Could not load {}/{}.jpg'.format(pp, ss[0]))
                i = i+1              
                
        else:
            i = i+1
        if reading_counter % 1000 == 0:
            print("Reading counter:", reading_counter)
        reading_counter += 1

    print('finished reading')

    train, val = format_lst(bbox, class_id, shape_img, path_img)
    print("Train image count:", len(train))
    print("Val image count:", len(val))
    write(train, val)


def write_line(img_path, im_shape, boxes, ids, idx):
    """ combine the inputs into gluon cv vector format """

    h, w, c = im_shape
    # for header, we use minimal length 2, plus width and height
    # with A: 4, B: 5, C: width, D: height
    A = 4
    B = 5
    C = w
    D = h
    # concat id and bboxes
    labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
    # normalized bboxes (recommended)
    labels[:, (1, 3)] /= float(w)
    labels[:, (2, 4)] /= float(h)
    # flatten
    labels = labels.flatten().tolist()
    str_idx = [str(idx)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_labels = [str(x) for x in labels]
    str_path = [img_path]
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    return line


def write(train,val):
    """ write train and val lists """

    print('write started')
    with open('../data_set_files/record_format_files/data-set_min/train.lst', 'w') as ftrain:
        for i in range(len(train)):
            ftrain.write(train[i])
    with open('../data_set_files/record_format_files/data-set_min/val.lst', 'w') as fval:
        for i in range(len(val)):
            fval.write(val[i])
    print('write  finished')


def main():
    """ create RecordFileFormat files """

    do_it()

    """im2rec = utils.download('https://raw.githubusercontent.com/apache/incubator-mxnet/' +
                        '6843914f642c8343aaa9a09db803b6af6f5d94a2/tools/im2rec.py', 'im2rec.py')"""
    subprocess.check_output([sys.executable, 'im2rec.py', '../data_set_files/record_format_files/data-set_min/val.lst',
                             '.', '--no-shuffle', '--pass-through', '--pack-label'])
    subprocess.check_output([sys.executable, 'im2rec.py','../data_set_files/record_format_files/data-set_min/train.lst',
                             '.', '--no-shuffle', '--pass-through', '--pack-label'])
    record_dataset = RecordFileDetection('../data_set_files/record_format_files/data-set_min/val.rec', coord_normalized=True)
    record_dataset = RecordFileDetection('../data_set_files/record_format_files/data-set_min/train.rec', coord_normalized=True)

    print('length:', len(record_dataset))
    first_img = record_dataset[0][0]
    print('image shape:', first_img.shape)
    print('Label example:')
    print(record_dataset[0][1])


if __name__ == "__main__":
    main()

