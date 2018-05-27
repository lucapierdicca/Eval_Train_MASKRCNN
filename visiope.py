"""
Mask R-CNN
Configurations and data loading code for MS COCO.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco
    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True
    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5
    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last
    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import pickle


import os
import sys
import time
import numpy as np
import imgaug  

from PIL import Image
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from PIL import Image, ImageDraw


# Path to trained weights file
COCO_MODEL_PATH = "mask_rcnn_coco.h5"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "./logs_VP"
DEFAULT_DATASET_YEAR = "2014"

PNG_IMAGES_PATH = "./pngImages_mod" 
BMP_IMAGES_PATH = "./bmpImages_mod"
JSON_PATH       = "labelbox_mod.json"

############################################################
#  Configurations
############################################################

class VisiopeConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "visiope"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1  ##TODO: your pc or alcorlab's pc

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4 #15 ###TODO your assignment

    STEPS_PER_EPOCH = 5

STAGE_1_EPOCHS = 2#40
STAGE_2_EPOCHS = 3#120
STAGE_3_EPOCHS = 4#160


############################################################
#  Dataset
############################################################

class VisiopeDataset(utils.Dataset):

    def load_visiope(self, dataset_dir, subset, class_ids=None,class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        #coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))  #file json da aprire
        #bisogna aprirlo e leggerlo

        #mettere un campo con tutte le classi elencate

        #mod class_id to a number of classes

        self.path = PNG_IMAGES_PATH  ##TODO: add the path to the dataset folder
        self.jsonName = JSON_PATH  ##TODO: add json file name
        
        jsonPath = self.jsonName
        b = json.load(open(jsonPath))
        b = [img for img in b if 'Masks' in img and 'image_problems' not in img['Label']]

        
        all_images_ids = range(len(b))

        np.random.seed(0)
        train_images_ids = sorted(np.random.choice(len(b), 
                                                   replace=False, 
                                                   size=int(len(b)*0.9)).tolist())[:3]
        val_images_ids = sorted(list(set(all_images_ids)-set(train_images_ids)))



        assert subset in ["train", "val"]

        classes = []
        image_ids = []  # riempire con gli id di tutte le immagini non skippate

        if subset == 'train':
            selected_subset = train_images_ids
        else:
            selected_subset = val_images_ids
            
        for xx in selected_subset:
            name = xx
            if b[xx]['Label'] == "Skip":
                continue
            else:
                image_ids.append(xx)
            for x in b[xx]['Label'].keys():
                name = x
                if name not in classes:
                    classes.append(name)

        classes = sorted(classes)


        #Add classes
        for i in range(len(classes)):
            self.add_class("visiope", i+1, classes[i]) #cerca add_class

        # Add images
        img_ext = ".png"
        if "mod" in JSON_PATH:
            img_ext = ".jpeg"
        for i in image_ids:
            self.add_image("visiope", image_id=i, path=dataset_dir + "/image" + str(i) + img_ext) #cerca add_image
        
        if return_coco:
            return b

    #reads a .bmp mask from HDD (black&white img)
    #returns a boolean list of lists (H x W x N)
    def bmpToBinary(self, path):
        img = Image.open(path)
        w, h = img.size
        pixels = list(img.getdata())
        #print(len(pixels))
        aux = []
        for i in range(h):
            boolean_pixel_list = [False if rgb_tuple[0]==0 else True for rgb_tuple in pixels[w*i:w*(i+1)]]
            aux.append(boolean_pixel_list)
        return aux


    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        path = BMP_IMAGES_PATH  ##TODO: add the path to the dataset folder
        self.jsonName = JSON_PATH  ##TODO: add json file name
        self.nomeBase = "image"
        
        ret1 = []
        ret2 = []

        jsonPath = self.jsonName
        b = json.load(open(jsonPath))
        b = [img for img in b if 'Masks' in img and 'image_problems' not in img['Label']]

        print(len(b))

        classes = []
        image_ids = []  # riempire con gli id di tutte le immagini non skippate
        masks_per_img = []
        for xx in range(len(b)):
            name = xx
            if b[xx]['Label'] == "Skip":
                continue
            else:
                image_ids.append(xx)
            for x in b[xx]['Label'].keys():
                name = x
                if name not in classes:
                    classes.append(name)
        
        classes = sorted(classes)

        labels = [0]*len(classes)

        #dato che ho diviso in (train & val) simple random sampling
        #il mapping indice_imgID salta e
        #sono costretto  prendere l'imgID interna all'(oggetto) dataset self
        immNum = self.image_info[image_id]['id']
        #immNum = image_id

        if len(b[immNum]['Label']) == 1:
            for x in b[immNum]['Label'].keys():
                name = x
            nameApp = name
            #TODO forse bisgna aggiungere uno 0 alla fine in questa stringa sotto
            name = path + "/image" + str(immNum) + name + '0' + ".bmp"
            aux = self.bmpToBinary(name)
            ret1.append(aux)
            ret2.append(classes.index(nameApp))
        else:
            for x in b[immNum]['Label'].keys():
                name = x
                nameApp = name
                name = path + "/image" + str(immNum) + name + str(labels[classes.index(name)]) + ".bmp"
                labels[classes.index(x)] += 1
                aux = self.bmpToBinary(name)
                ret1.append(aux)
                ret2.append(classes.index(nameApp))

        mask = np.stack(ret1, axis=2).astype(np.bool)
        class_ids = np.array(ret2, dtype=np.int32)+1

        print(mask.shape, class_ids.shape)

        return mask, class_ids


    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]['path']
        return info


############################################################
#  COCO Evaluation
############################################################


def prediction(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """

    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    print(image_ids)
    print(dataset.image_info[image_ids[0]]['path'], dataset.image_info[image_ids[0]]['id'])

    y_pred = []
    for image_id in image_ids:
       
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

        info = dataset.image_info[image_id]

        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))

        # Run detection
        r = model.detect([image], verbose=0)

        print(len(r[0]['rois']))

        y_pred = r


    pickle.dump(y_pred,open('y_pred_base','wb') ) 

    return image_ids, y_pred


def IoU(bbox_pred, bbox_true, mask_pred, mask_true):
    
    def A(bbox):
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        return abs(x0-x1)*abs(y0-y1)

    bbox_pred_A = A(bbox_pred)
    bbox_true_A = A(bbox_true)

    pdraw_pred = ImageDraw.Draw(mask_pred)
    pdraw_pred.rectangle(bbox_pred, fill=1)

    bool_matrix_pred = np.array(mask_pred)


    pdraw_true = ImageDraw.Draw(mask_true)
    pdraw_true.rectangle(bbox_true, fill=1)

    bool_matrix_true = np.array(mask_true)

    intersection = np.logical_and(bool_matrix_pred, bool_matrix_true)

    intersection_A = np.max(np.count_nonzero(intersection, axis=1))*np.max(np.count_nonzero(intersection, axis=0))


    return intersection_A / (bbox_true_A + bbox_pred_A - intersection_A)


def evaluation(image_ids, y_pred, dataset):

    import matplotlib.pyplot as plt

    def get_ax(rows=1, cols=1, size=16):
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax
        
    y_true = []

    for i in image_ids:
        mask, class_ids = dataset.load_mask(i)
        bbox = utils.extract_bboxes(mask)
        y_true.append([bbox, class_ids, mask])

    pickle.dump(y_true,open('y_true','wb') ) 

    ax = get_ax(len(image_ids), 2)



    if len(image_ids) <=3:
        for index, i in enumerate(image_ids):
            image = dataset.load_image(i)
            visualize.display_instances(image, 
                                        y_true[i][0], 
                                        y_true[i][2], 
                                        y_true[i][1], 
                                        dataset.class_names,
                                        ax=ax[0],
                                        title="True")
            visualize.display_instances(image, 
                                        y_pred[i]['rois'], 
                                        y_pred[i]['masks'], 
                                        y_pred[i]['class_ids'], 
                                        dataset.class_names, 
                                        y_pred[i]['scores'],
                                        ax=ax[1],
                                        title="Predicted")
        plt.show()


    image = dataset.load_image(i)
    w, h = image.shape[0], image.shape[1]
    print(w,h)

    mask_pred = Image.new('1', (w,h))
    mask_true = Image.new('1', (w,h))

    bah = []
    currentIoU = 0
    for index_true, i in enumerate(y_true[0][0]):
        bah.append([])
        for index_pred, j in enumerate(y_pred[0]['rois']):
            currentIoU = IoU(i, j, mask_pred, mask_true)
            bah[index_true].append((currentIoU, 
                                    y_true[0][1][index_true], 
                                    y_pred[0]['class_ids'][index_pred]))
    
    return bah


############################################################
#  Training
############################################################


if __name__ == '__main__':

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on VISIOPE')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = VisiopeConfig()
    else:
        class InferenceConfig(VisiopeConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            #DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    #config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
        
        print("Loading weights ", model_path)
        model.load_weights(model_path, 
                            by_name=True, 
                            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
    elif args.model.lower() == "last":
        model_path = model.find_last()[1]
       
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)



    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = VisiopeDataset()
        dataset_train.load_visiope(args.dataset, "train")
        dataset_train.prepare()

        print(dataset_train.class_info)

        # Validation dataset
        dataset_val = VisiopeDataset()
        dataset_val.load_visiope(args.dataset, "val")
        dataset_val.prepare()

        print(dataset_val.class_info)

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        
        #*******************************************************************
        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=STAGE_1_EPOCHS,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=STAGE_2_EPOCHS,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=STAGE_3_EPOCHS,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = VisiopeDataset()
        coco = dataset_val.load_visiope(args.dataset, "val", return_coco=True)
        dataset_val.prepare()

        print("Running COCO evaluation on {} images.".format(args.limit))
        image_ids, y_pred = prediction(model, dataset_val, coco, "bbox", image_ids=[18])
        score = evaluation(image_ids, y_pred, dataset_val)

        print(score)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
        
