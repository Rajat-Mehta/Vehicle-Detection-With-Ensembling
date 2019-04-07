This folders contain scripts that can be used to make predictions with the models that we trained i.e. Faster R-CNN and SSD.

1. predict_frcnn.py, predict_ssd.py, predict_ssd_subset.py: these scripts can be used for making predictions on new images using any of the pre-trained model: ssd, fr-cnn etc.
Command: python predict_frcnn.py
Command: python predict_ssd.py
Command: python predict_ssd_subset.py

2. predict_ensemble_boxes.py: this script can be used for predicting detections using the final trained ensemble model.
Command: python predict_ensemble_boxes.py
