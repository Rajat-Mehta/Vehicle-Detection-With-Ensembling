import time
from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
from gluoncv.utils import download, viz
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOCMApMetric
import logging
import os

"""
This script can be used to train SSD model on our dataset end-to-end.
All of the dataset paths have been configured in the arguments. You can start training by running this command:
python finetune_ssd_resnet.py

This script was taken from Gluon cv framework's official website and modified to adapt to our problem.
Reference: https://gluon-cv.mxnet.io/build/examples_detection/train_ssd_voc.html
"""


def get_dataloader(net, train_dataset,val_dataset ,data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader


def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, [gt_difficults])
    return eval_metric.get()


# It require both *.rec and *.idx file in the same
# directory, where raw image and labels are stored in *.rec file for better
# IO performance, *.idx file is used to provide random access to the binary file.

dataset = gcv.data.RecordFileDetection('../data_set_files/record_format_files/data-set_min/train.rec')
val_dataset = gcv.data.RecordFileDetection('../data_set_files/record_format_files/data-set_min/val.rec')
classes = ['car','articulated_truck','bus','bicycle','motorcycle','motorized_vehicle','pedestrian','single_unit_truck',
           'work_van','pickup_truck','non-motorized_vehicle']  # only one foreground class here

# image, label = dataset[0]
# print('label:', label)
# display image and label
# ax = viz.plot_bbox(image, bboxes=label[:, :4], labels=label[:, 4:5], class_names=classes)
# plt.show()

net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)
net.reset_class(classes)
eval_metric = VOCMApMetric(iou_thresh=0.5, class_names=classes)
# net.load_parameters('../trained_model_weights/ssd_resnet/epoch_35_ssd_512_resnet50_v1_voc_mio_tcd.params')
train_data,val_data = get_dataloader(net, dataset, val_dataset, 512, 4, 0)

try:
    a = mx.nd.zeros((1,), ctx=mx.gpu(0))
    ctx = [mx.gpu(0)]
except:
    ctx = [mx.cpu()]

prefix = './train_logs/ssd_resnet/ssd_resnet'
# set up logger
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_file_path = prefix + '_train.log'
log_dir = os.path.dirname(log_file_path)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)
fh = logging.FileHandler(log_file_path)
logger.addHandler(fh)


# Returns a ParameterDict containing this Block and all of its childrens Parameters(default)
net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(
    net.collect_params(), 'sgd',
    {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9})
print(ctx)

mbox_loss = gcv.loss.SSDMultiBoxLoss()
ce_metric = mx.metric.Loss('CrossEntropy')
smoothl1_metric = mx.metric.Loss('SmoothL1')
best_map = [0]
logger.info('Training started')

for epoch in range(0, 100):
    # Resets the internal evaluation result to initial state.

    ce_metric.reset()
    smoothl1_metric.reset()
    tic = time.time()
    btic = time.time()
    net.hybridize(static_alloc=True, static_shape=True)
    for i, batch in enumerate(train_data):
        batch_size = batch[0].shape[0]
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
        with autograd.record():
            cls_preds = []
            box_preds = []
            for x in data:
                cls_pred, box_pred, _ = net(x)
                cls_preds.append(cls_pred)
                box_preds.append(box_pred)
            sum_loss, cls_loss, box_loss = mbox_loss(
                cls_preds, box_preds, cls_targets, box_targets)
            autograd.backward(sum_loss)
        # since we have already normalized the loss, we don't want to normalize
        # by batch-size anymore
        trainer.step(1)
	# Updates the internal evaluation result
        ce_metric.update(0, [l * batch_size for l in cls_loss])
        smoothl1_metric.update(0, [l * batch_size for l in box_loss])
	# Gets the current evaluation result
        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        if i % 20 == 0:
            logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
        btic = time.time()
    map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
    logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
    net.save_parameters('./train_logs/ssd_resnet/epoch_'+str(epoch)+'_ssd_512_resnet50_v1_voc_mio_tcd.params')
    



