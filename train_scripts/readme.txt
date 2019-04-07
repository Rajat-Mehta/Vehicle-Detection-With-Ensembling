This folder contains scripts for training our base line models which will be used in our ensemble setup. We trained three models: Faster R-CNN, SSD and SSD-Subset(on a subset of our dataset)

Note: For this submission purpose we added a smaller version of our original dataset. If you want to train it on the full dataset, you can copy it from serv-2103's '/kaecheledata/MIO-TCD-Localization/' folder and subfodlers: train, test, val to our "data_set_files" folder.


1. train_faster_rcnn.py: this is the script to finetune a pre-trained fr-cnn model(on VOC dataset) using on MIO-TCD dataset.
Command: python train_faster_rcnn.py

2. filtered_finetune_fr-cnn.py: this is the script to finetune a pre-trained fr-cnn model(on VOC dataset) using our filtered MIO-TCD dataset.
Command: python filtered_finetune_fr-cnn.py

3. finetune_ssd_resnet.py: this is the script to finetune an already trained ssd model(on VOC dataset) using our MIO-TCD dataset.
Command: python finetune_ssd_resnet.py

4. filtered_finetune_ssd.py: this is the script to finetune an already trained ssd model(on VOC dataset) using our filtered MIO-TCD dataset.
Command: python filtered_finetune_ssd.py
