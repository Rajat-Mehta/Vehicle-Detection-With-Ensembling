# import keras
import keras
import keras_retinanet
from keras_retinanet import models
from keras_retinanet.models import load_model
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras.callbacks import ModelCheckpoint
from keras_retinanet.utils.transform import random_transform_generator

path='./snapshots/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpointer = ModelCheckpoint(filepath=path, verbose=1, save_best_only=False)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

#model = keras_retinanet.models.backbone('resnet50').retinanet(num_classes=11)
model = load_model('./snapshots/weights.35-0.82.hdf5', backbone_name='resnet50')
batch_size = 4
epochs=150


transform_generator = random_transform_generator(
    min_rotation=-0.1,
    max_rotation=0.1,
    min_scaling=(0.9, 0.9),
    max_scaling=(1.1, 1.1),
    flip_x_chance=0.5
)

generator = CSVGenerator(
    csv_data_file='./data_set_retina/train.csv',
    csv_class_file='./data_set_retina/class_id_mapping.txt',
    batch_size=batch_size,
    transform_generator=transform_generator
)

generator_val = CSVGenerator(
    csv_data_file='./data_set_retina/val.csv',
    csv_class_file='./data_set_retina/class_id_mapping.txt',
    batch_size=batch_size,
    transform_generator=transform_generator
)
#    transform_generator=transform_generator
with open('./data_set_retina/train.csv', 'r') as f:
    l = f.readlines()
with open('./data_set_retina/val.csv', 'r') as f:
    v = f.readlines()

ts=len(l)/batch_size
vs=len(v)/batch_size
model.compile(
    loss={
        'regression'    : keras_retinanet.losses.smooth_l1(),
        'classification': keras_retinanet.losses.focal()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
)
csv_logger = keras.callbacks.CSVLogger('./logs/training_log.csv', separator=',', append=False)
history = LossHistory()
#evaluation = Evaluate(generator_val, weighted_average=True)

print(history)

model.fit_generator(generator, steps_per_epoch=10000, epochs=epochs, verbose=1, callbacks=[history, csv_logger, checkpointer],
                    validation_data=generator_val, validation_steps=5000, class_weight=None, max_queue_size=10,
                    workers=1, use_multiprocessing=True, shuffle=True, initial_epoch= 35)


"""
python evaluate.py csv ../../data_set_retina/val.csv ../../data_set_retina/class_id_mapping.txt ../../snapshots/weights.27-0.85.hdf5
"""