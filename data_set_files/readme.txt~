This folder contains all of the data set files including the train and validation set images. Gluoncv requires input in the form of RecordFileFormat and we have already converted the dataset to the required format using the scripts in "data_set_script" folder. You will find all of the data set files needed for training model in the sub folders here.

1. record_format_files: this folder contains data set in RecordFileDetection format which is the required format for feeding data to gluon cv models.

2. filtered_gt_train.csv: this csv file contains ground truth for filtered dataset which was used for training ssd-subset model on a subset of original classes.

3. filtered_gt_val.csv: this csv file contains ground truth for filtered (validation) dataset which was used for validating ssd-subset model.

4. gt_train.csv and gt_val.csv: ground truth labels for the whole train and validation set.

5. gt_val_0-10k.csv and gt_val_10-20k.csv: gt_val was splitted into two halves, one for tuning the weight parameters for ensembling and the other one for evaluating the tuned parameters.

6. image_shapes: the shapes of each of the validation image, used while rescaling the ssd model's detections. These shapes were calculated just once and stored in this file to save recomputation of image shapes.

7. train.txt and valid.txt: these files include the full paths of training and validation images stored locally in user drive.

8. val_image_names.txt: this file contains the names of all of the validation set images.

9. train and valid folder contains a small subset of the whole train and validation set. The complete data set can be found at in serv-2103 at: /serv-2103/kaecheledata/MIO-TCD-Localization/ in train and valid folders
