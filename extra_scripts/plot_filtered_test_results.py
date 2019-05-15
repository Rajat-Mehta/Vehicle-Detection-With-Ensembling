import csv
from matplotlib import pyplot as plt
from gluoncv.utils import viz
import gluoncv as gcv
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import mxnet as mx

TEST_RESULT_PATH = "../ensemble_scripts/dets_numpy/test/results/result_ensemble_fsr.csv"
THRESH = 0.50


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


def plot_bbox(img, bbox, score, cid, classes, model):
    """ plots bounding boxes on the given image """
    print("Plotting")
    ax = viz.plot_bbox(img, bbox, score, cid, thresh=0.5, class_names=classes)
    plt.suptitle(model)
    plt.show()


def read_retina_images(path):
    """ read validation images from the path provided """

    file_obj = path
    images = []

    print("Reading images")
    i = 0
    for item in file_obj:
        item = item.strip()
        if item:
            if (i % 500 == 0) & (i > 0) :
                print("Read %d/%d images"% (i, 100))

            """
            image = read_image_bgr(item)
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
            """

            image = preprocess_image_retina(item)
            images.append(image)
        i += 1
    print("Finished reading images")

    return images


def preprocess_image_retina(im_fname):
    """ preprocess image: will scale pixels between -1 and 1, sample-wise."""

    # load image
    image = read_image_bgr(im_fname)
    # preprocess image for network
    image = preprocess_image(image)
    return image



with open(TEST_RESULT_PATH, "r") as f:
    print(f)
    reader = csv.reader(f, delimiter=",")
    filtered_results = []
    for i, line in enumerate(reader):

        if float(line[2]) > THRESH:
            line[0] = line[0].zfill(8)
            line[3] = int(float(line[3]))

            line[3] = int(float(line[3]))
            line[4] = int(float(line[4]))
            line[5] = int(float(line[5]))
            line[6] = int(float(line[6]))
            filtered_results.append(line)

print("Writing filtered results to csv")
with open("../ensemble_scripts/dets_numpy/test/results/filtered_result_ensemble_fsr_" + str(THRESH) + ".csv", 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(filtered_results)
classes = ['car', 'articulated_truck', 'bus', 'bicycle', 'motorcycle', 'motorized_vehicle', 'pedestrian',
                    'single_unit_truck', 'work_van', 'pickup_truck', 'non-motorized_vehicle']

image_names = [
    #'../data_set_files/test/00111119.jpg'
    #'../data_set_files/test/00129733.jpg'
    #'../data_set_files/test/00135693.jpg'
    #'../data_set_files/test/00122999.jpg'
    #'../data_set_files/test/00118639.jpg'
    #'../data_set_files/test/00117410.jpg'
    '../data_set_files/test/00110602.jpg'
    #'../data_set_files/test/00115203.jpg'
]

processed_images = read_retina_images(image_names)
inp = filtered_results[17:21]
print(inp)
print(len(inp))
bbox, cid, score = convert_to_plot_box_format(inp)

x, image = gcv.data.transforms.presets.ssd.load_test(image_names, 512)
plot_bbox(image, mx.nd.array(bbox), mx.nd.array(score), mx.nd.array(cid), classes, "Ensemble")
