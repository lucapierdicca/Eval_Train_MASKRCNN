import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import visiope



def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax






# CONFIG
#--------------------------------------------------------------------
config = visiope.VisiopeConfig()

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
#config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0 #OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO






# DATASET
#-------------------------------------------------------------------

 # Load validation dataset
dataset_train = visiope.VisiopeDataset()
dataset_train.load_visiope(r"./pngImages", "val")

# Must call before using the dataset
dataset_train.prepare()

print("Images: {}\nClasses: {}".format(len(dataset_train.image_ids), dataset_train.class_names))







# MODEL
#---------------------------------------------------------------------
# Create model in inference mode
MODEL_DIR=r"./logs"
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)






# SELECT IMAGE
#-----------------------------------------------------------------------------------
#image_id = random.choice(dataset_train.image_ids)

image_id =18

image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_train, config, image_id, use_mini_mask=False)
    
info = dataset_train.image_info[image_id]

print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset_train.image_reference(image_id)))




# CREATE SUBPLOTS
ax = get_ax(1,3)




# GROUND TRUTH
#-----------------------------------------------------------------------------------
visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, ax=ax[0],
                            title="Ground_Truth")





# BASE
#------------------------------------------------------------------------------------
weights_path = "mask_rcnn_coco.h5"
model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
print("Model weights path: ", weights_path)

# Run object detection
results = model.detect([image], verbose=0)

print("ROIS: %d" % len(results[0]['rois']))

# Display results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_train.class_names, r['scores'], ax=ax[1],
                            title="Predictions_base")






# LAST
#----------------------------------------------------------------------------------
weights_path = model.find_last()[1]
print("Model weights path: ", weights_path)
model.load_weights(weights_path, by_name=True)

# Run object detection
results = model.detect([image], verbose=0)

print("ROIS: %d" % len(results[0]['rois']))

# Display results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_train.class_names, r['scores'], ax=ax[2],
                            title="Predictions_last")





plt.show()