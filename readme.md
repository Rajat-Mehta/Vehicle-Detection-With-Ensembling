# Adaptive Weight based Model Ensembling

This repository contains a Vehicle Detection model built using Adaptive Weight based Ensembling approach. The ensembling was done using Faster R-CNN and Single Shot Detector models and these models were trained on MIO-TCD dataset which consists of more than 130,000 images of vehicles taken in different lighting and weather conditions. The content of different folders in this repository is explained below.

Folder "Guided_Research_Code" contains all of the scripts used for training and testing the ensemble models. It contains multiple folders which have their own readme.txt files describing each and every script file in that folder along with the commands that are required for running each of those scripts.

## Folder descriptions:
1. data_set_files: this folder contains the dataset related files like ground truth for train and validations sets, formatted data according to gluoncv RecordFileDetection Format etc. which can be directly used for training purposes.

2. dataset_scripts: this folder contains scripts for generating files present in "data_set_files" folder.

3. ensemble_scripts: this folder contains scripts for ensembling multiple models, evaluating ensemble model and tuning the weight parameters involved in ensembling.

4. extra_scripts: this folder contains miscellaneous scripts like plotting data, rescaling the detection results.

5. model_evaluation: this folder contains script for evaluating the ensemble model results.

6. model_result_numpy: this folder contains the detection results of individual models on the whole validation set(20,000 images). The individual results are then combined into a single list of 20,000 detections which is fed to the ensemble model. All of these results for individual models(FR-CNN, SSD and SSD-Subset) are computed once and stored in numpy files so that we don't need to process 20,000 images again and again.

7. models_for_ensemble: this folder contains the weights of trained individual models(Faster R-CNN, SSD and SSD-Subset).

8. predict_scripts: this folder contains scripts for predicting detections on new images using the individual and ensemble models.

9. results: this folder contains final results with some sample result images.

10. sample_data: this folder contains the sample images that can be used for testing models.

11. train_scripts: this folder contains 4 scripts for training fr-cnn, ssd, ssd-subset and fr-cnn-subset models.

12. Rajat_Guided_Research_Report.pdf: this is the Guided research report file.

NOTE: each of the above folders contains their specific readme files explaining what each script does and how to run it.


To run the whole project, you can follow the following order:

1. Generate dataset files using the scripts in "dataset_scripts" folder.

2. Train three models: Faster R-CNN, SSD and SSD-Subset using the files generated in the previous step. Use scripts in "train_script" folder to train these models.

3. You can test the these three trained models by feeding new images to these models and see how well they perform in detecting vehicles. Use scripts in "predict_script" folder for that.

4. These three models can now be combined together to make our ensemble model. Use scripts in "ensemble_script" folder. Weight parameters for each participating model can be tuned using "tune_weights_ensemble.py" script.

5. The ensembled model with optimal weight parameters can now be evaluated on the validation set using "localization_evaluation.py" script in "model_evaluation" folder.

6. Once the ensemble model is ready, "predict_ensemble_boxes.py" script in "predict_scripts" folder can be used to compute predictions on new images.

Note: For this submission purpose, we have added only a subset of our whole training set in this folder. If you want to train these models on the whole dataset, you can get it from serv-2103: "/serv-2103/kaecheledata/MIO-TCD-Localization/" in train and valid folders.
