To train Faster-RCNN or SSD model using gluoncv framework, we need to feed the data in the form of RecordFileFormat. The files in this folder contains scripts
(convert_filtered_labels_new.py and convert_labels_new.py) that can be used to read data from "data_set_files" folder and generate .rec and .lst files which will be used for training the models.

1. convert_filtered_labels_new.py: this is the script that takes the filtered ground truth file(filtered_gt_train.csv) as input and generate dataset files in RecordFileDetection format which is then fed to the gluon cv models.
Command: python convert_filtered_labels_new.py

2. convert_labels_new.py: this is the script that takes the original ground truth file(gt_train.csv) as input and generate dataset files in RecordFileDetection format which is then fed to the gluon cv models.
Command: python convert_labels_new.py

3. extract_val_gt.py: this script extracts the ground truth labels of the validation images from original ground truth file(gt_train.csv).
Command: python extract_val_gt.py

4. im2rec.py: this script is used by convert_labels_new.py script internally to generate .rec files and is provided by gluoncv framework.
