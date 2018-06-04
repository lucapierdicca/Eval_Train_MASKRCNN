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
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log

import visiope_full
import cv2


import datetime
import time
import skimage.draw
from skimage import io
import pickle



def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    
    return ax

	
	
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    
    #print(type(gray))
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    #print(mask.shape)
    # Copy color pixels from the original color image where mask is set
    
    
    if mask.shape[0] == gray.shape[0]:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        mask = np.full((gray.shape[0], gray.shape[1], gray.shape[2]), False)
        
        splash = np.where(mask, image, gray).astype(np.uint8)
            
    return splash



def detect_and_color_splash(model, dataset, video_path=None):
        
    import cv2
    # Video capture
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    n_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Tot frames: %d" % n_frames)

    
    # Define codec and create video writer
    file_name = "{:%Y%m%dT%H%M%S}.mp4".format(datetime.datetime.now())
    detection_name = "{:%Y%m%dT%H%M%S}.pickle".format(datetime.datetime.now())
    vwriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    print(fps)
    count = 0
    success = True
    detection_list = []
	
    while success:
        print("frame: %d / %d" % (count, n_frames))
        # Read next image
        success, image = vcapture.read()
        print(success)
        if success:
            # OpenCV returns images as BGR, convert to RGB
            print(image.shape)
            image = image[..., ::-1]
            print(image.shape)
            # Detect objects
            r = model.detect([image], verbose=0)[0]
            print("ROIS: %d" % len(r['rois']))
            detection_list.append(r)
			# Create a plot made of frame+masks+bboxes
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, scores=r['scores'], ax = get_ax(1))
            # Save it on the HDD
            #plt.figure(figsize=(width,height))
            plt.savefig("image" + str(count),facecolor=None, edgecolor=None)
            plt.clf()
            plt.close()
            # Load it back as a numpy array
            numpy_frame = cv2.imread("image" + str(count) + '.png')
            #numpy_frame = numpy_frame.astype(np.uint8)
            print(numpy_frame.shape)
            # RGB -> BGR to save image to video
            #numpy_frame = numpy_frame[..., ::-1]
            print(numpy_frame.shape)
            # Add image to video writer
            vwriter.write(numpy_frame)
            count += 1
            if count == 100:
                break
          
    vwriter.release()
    print("Saved to ", file_name)
    pickle.dump(detection_list, open(detection_name, 'wb'))
    

    

# CONFIG
#--------------------------------------------------------------------
config = visiope_full.VisiopeConfig()

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #DETECTION_MIN_CONFIDENCE = 0

config = InferenceConfig()
#config.display()
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0 





# MODEL
#---------------------------------------------------------------------
# Create model in inference mode
MODEL_DIR= r"./logs_VF"

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)






# LOAD CHKPT
#----------------------------------------------------------------------------------
weights_path = r"./logs_VF/mask_rcnn_visiope_0080.h5"
print("Model weights path: ", weights_path)
model.load_weights(weights_path, by_name=True)







# LOAD DATASET (we need it to obtain the mapping class_id:class_name)
#---------------------------------------------------------------------------------
selected_COCO_class_ids = [27,31,47,51,62,65,67,70,72,73,78,79,81,82,84,85,87,90]


print("\n======VALIDATION SET======")
dataset_val = visiope_full.VisiopeDataset()

dataset_val.load_visiope(sampling="val")
print("\nN tot_visiope images: %d" % len(dataset_val.b))

n_val_visiope_imgs = len(dataset_val.image_info)
print("N val_visiope images: %d" % n_val_visiope_imgs)


dataset_val.load_coco(sampling="val", class_ids=selected_COCO_class_ids)
n_val_COCO_imgs = len(dataset_val.image_info)-n_val_visiope_imgs
print("N val_COCO images: %d" % n_val_COCO_imgs)

dataset_val.prepare()
print("N tot val images (val_visiope + val_COCO): %d\n" % len(dataset_val.image_info))







# VIDEO
#---------------------------------------------------------------------
video_path= "v.mp4"

detect_and_color_splash(model, dataset_val, video_path=video_path)

